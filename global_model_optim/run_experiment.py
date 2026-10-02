#!/usr/bin/env python3
"""Run one global-model fitting experiment on one exported dataset folder.

The folder is a Nextflow work directory of a finished FitGlobalRnaKineticsModel or
FitGlobalIntronCoverageModel task, exported with tools/export_work_dir.sh. It already holds the
staged model inputs under fixed names, plus the original model_parameters.tsv / test_results_raw.tsv
that the unmodified pipeline produced, which is what experiment_001 should reproduce.

This script stays the same across experiments on purpose: if the subsetting were duplicated per
experiment, a difference in results could come from the genes each one happened to see rather than
from the optimizer. Experiments differ only in the module named by --experiment.

Subsetting: genes (or introns, for the coverage model) are sampled at random under a fixed seed, so
"1000 genes" means the same 1000 genes for every experiment on a given dataset, without needing a
gene list per dataset. LRTs are capped by taking the first n, since their order is meaningful.
Omitting either option uses everything.
"""

import argparse
import importlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from rna_kinetics.data import concat_gene_data_list, load_dataset_metadata, load_gene_data_list

MODEL_TYPES = ('rna_kinetics', 'intron_coverage')


def get_git_revision(repo_dir: Path) -> str:
    """Short commit hash of the code being run, or 'unknown' outside a git checkout."""
    try:
        return subprocess.run(['git', '-C', str(repo_dir), 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def write_subset_inputs(dataset_dir: Path,
                        subset_dir: Path,
                        model_type: str,
                        num_genes: int | None,
                        num_introns: int | None,
                        num_lrt_tests: int | None,
                        seed: int) -> dict[str, Path]:
    """Write subset copies of the plain-text inputs and return the paths to use.

    Only the small TSVs are rewritten; the coverage parquet files are read from the dataset folder
    as they are, since the loaders select the rows they need by name.
    """
    subset_dir.mkdir(parents=True, exist_ok=True)
    random_generator = np.random.default_rng(seed)
    input_paths = {name: dataset_dir / f'{name}.tsv'
                   for name in ('modeled_genes', 'modeled_introns', 'design_matrix', 'lrt_metadata',
                                'library_size_factors', 'exon_counts', 'intron_counts',
                                'isoform_length_factors')}

    if model_type == 'rna_kinetics' and num_genes is not None:
        modeled_genes_df = pd.read_csv(input_paths['modeled_genes'], sep='\t')
        if num_genes < len(modeled_genes_df):
            kept_positions = random_generator.choice(len(modeled_genes_df), size=num_genes, replace=False)
            modeled_genes_df = modeled_genes_df.iloc[np.sort(kept_positions)]
        modeled_introns_df = pd.read_csv(input_paths['modeled_introns'], sep='\t')
        modeled_introns_df = modeled_introns_df[modeled_introns_df.gene_id.isin(set(modeled_genes_df.gene_id))]
        modeled_genes_df.to_csv(subset_dir / 'modeled_genes.tsv', sep='\t', index=False)
        modeled_introns_df.to_csv(subset_dir / 'modeled_introns.tsv', sep='\t', index=False)
        input_paths['modeled_genes'] = subset_dir / 'modeled_genes.tsv'
        input_paths['modeled_introns'] = subset_dir / 'modeled_introns.tsv'

    if model_type == 'intron_coverage' and (num_introns is not None or num_genes is not None):
        # This fixture has no modeled_genes.tsv, since the coverage process never receives one, so
        # a gene subset is taken from the gene_id column instead. num_introns wins when both are
        # given; without this branch, --num_genes would be silently ignored here and the run would
        # quietly go to full scale.
        modeled_introns_df = pd.read_csv(input_paths['modeled_introns'], sep='\t')
        if num_introns is not None:
            if num_introns < len(modeled_introns_df):
                kept_positions = random_generator.choice(len(modeled_introns_df), size=num_introns, replace=False)
                modeled_introns_df = modeled_introns_df.iloc[np.sort(kept_positions)]
        else:
            gene_ids = modeled_introns_df.gene_id.unique()
            if num_genes < len(gene_ids):
                kept_gene_ids = random_generator.choice(gene_ids, size=num_genes, replace=False)
                modeled_introns_df = modeled_introns_df[modeled_introns_df.gene_id.isin(set(kept_gene_ids))]
        modeled_introns_df.to_csv(subset_dir / 'modeled_introns.tsv', sep='\t', index=False)
        input_paths['modeled_introns'] = subset_dir / 'modeled_introns.tsv'

    if num_lrt_tests is not None:
        # The reduced design matrices are loaded per test_id listed here, so capping this file is
        # enough -- the unused ones are simply never read.
        lrt_metadata_df = pd.read_csv(input_paths['lrt_metadata'], sep='\t').head(num_lrt_tests)
        lrt_metadata_df.to_csv(subset_dir / 'lrt_metadata.tsv', sep='\t', index=False)
        input_paths['lrt_metadata'] = subset_dir / 'lrt_metadata.tsv'

    return input_paths


def load_coverage_tensor(dataset_dir: Path, intron_names: list[str], sample_names: list[str]) -> torch.Tensor:
    coverage_by_sample: dict[str, pd.DataFrame] = {}
    for sample_name in tqdm(sample_names, desc='Loading coverage data'):
        sample_coverage_df = pd.read_parquet(dataset_dir / f'{sample_name}.parquet').set_index('intron_name')
        coverage_by_sample[sample_name] = sample_coverage_df.loc[intron_names]
    return torch.tensor(
        np.stack([coverage_by_sample[sample_name].values for sample_name in sample_names]),
        dtype=torch.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, required=True,
                        help='Exported work directory of a finished global-model fit.')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Label for the outputs. Defaults to the dataset folder name.')
    parser.add_argument('--model_type', choices=MODEL_TYPES, required=True)
    parser.add_argument('--experiment', type=str, required=True,
                        help='Module name under experiments/, e.g. experiment_001.')
    parser.add_argument('--output_folder', type=Path, default=Path('.'))
    parser.add_argument('--num_genes', type=int, default=None,
                        help='Random subset of genes for the rna_kinetics model. Default: all.')
    parser.add_argument('--num_introns', type=int, default=None,
                        help='Random subset of introns for the intron_coverage model. Default: all.')
    parser.add_argument('--num_lrt_tests', type=int, default=None,
                        help='Use only the first n LRT contrasts. Default: all.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--torch_num_threads', type=int, default=None,
                        help='Left alone by default; only set it when a run needs to be pinned.')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    dataset_name = args.dataset_name or args.dataset_dir.resolve().name
    args.output_folder.mkdir(parents=True, exist_ok=True)

    if args.torch_num_threads is not None:
        torch.set_num_threads(args.torch_num_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    experiment_module = importlib.import_module(f'experiments.{args.experiment}')

    input_paths = write_subset_inputs(
        dataset_dir=args.dataset_dir,
        subset_dir=args.output_folder / 'subset_inputs',
        model_type=args.model_type,
        num_genes=args.num_genes,
        num_introns=args.num_introns,
        num_lrt_tests=args.num_lrt_tests,
        seed=args.seed,
    )

    dataset_metadata = load_dataset_metadata(
        design_matrix_file=input_paths['design_matrix'],
        library_size_factors_file=input_paths['library_size_factors'],
        lrt_metadata_file=input_paths['lrt_metadata'],
        reduced_matrices_folder=args.dataset_dir / 'reduced_design_matrices',
    )

    num_lrt_tests_used = len(dataset_metadata.lrt_metadata)
    print(f'{dataset_name} / {args.model_type} / {args.experiment}: fitting with '
          f'{num_lrt_tests_used} LRT contrast(s)', flush=True)

    start_time = time.monotonic()
    if args.model_type == 'rna_kinetics':
        print(f'  {len(pd.read_csv(input_paths["modeled_genes"], sep=chr(9)))} genes, '
              f'{len(pd.read_csv(input_paths["modeled_introns"], sep=chr(9)))} introns', flush=True)
        gene_data_list = load_gene_data_list(
            modeled_genes_file=input_paths['modeled_genes'],
            modeled_introns_file=input_paths['modeled_introns'],
            exon_counts_file=input_paths['exon_counts'],
            intron_counts_file=input_paths['intron_counts'],
            isoform_length_factors_file=input_paths['isoform_length_factors'],
            coverage_folder=args.dataset_dir,
            sample_names=dataset_metadata.sample_names,
        )
        global_gene_data = concat_gene_data_list(gene_data_list)
        num_genes_used = len(global_gene_data.gene_names)
        num_introns_used = len(global_gene_data.intron_names)
        model_param_df, test_results_df, training_log_df = experiment_module.fit_global_rna_kinetics(
            global_gene_data=global_gene_data,
            dataset_metadata=dataset_metadata,
            verbose=args.verbose,
        )
    else:
        intron_names = pd.read_csv(input_paths['modeled_introns'], sep='\t')['intron_id'].tolist()
        num_genes_used = pd.Series(intron_names).str.rsplit('_', n=1).str[0].nunique()
        num_introns_used = len(intron_names)
        print(f'  {num_introns_used} introns from {num_genes_used} genes', flush=True)
        coverage = load_coverage_tensor(args.dataset_dir, intron_names, dataset_metadata.sample_names)
        model_param_df, test_results_df, training_log_df = experiment_module.fit_global_intron_coverage(
            coverage=coverage,
            dataset_metadata=dataset_metadata,
            intron_names=intron_names,
            verbose=args.verbose,
        )
    wall_time_seconds = time.monotonic() - start_time

    run_metadata = {
        'experiment': args.experiment,
        'experiment_description': getattr(experiment_module, 'EXPERIMENT_DESCRIPTION', ''),
        'dataset': dataset_name,
        'model_type': args.model_type,
        'num_genes_used': int(num_genes_used),
        'num_introns_used': int(num_introns_used),
        'num_lrt_tests_used': int(num_lrt_tests_used),
        'num_samples': len(dataset_metadata.sample_names),
        'seed': args.seed,
        'git_revision': get_git_revision(Path(__file__).resolve().parent),
        'torch_version': torch.__version__,
        'torch_num_threads': torch.get_num_threads(),
        'hostname': platform.node(),
        'wall_time_seconds': round(wall_time_seconds, 2),
    }

    # Fixed schema, one row per LRT result. This is the file that gets compared across experiments,
    # so nothing experiment-specific belongs in it; free-form logging goes to training_log.tsv.
    summary_df = test_results_df.copy()
    for column, value in run_metadata.items():
        if column not in ('experiment_description',):
            summary_df[column] = value
    summary_df['chi2_is_negative'] = summary_df['chi2_test_statistics'] < 0

    summary_columns = [
        'experiment', 'dataset', 'model_type', 'test_id', 'tested_parameter', 'variable',
        'group_1', 'group_2', 'lrt_df', 'lfc', 'loss_full_model', 'loss_reduced_model',
        'chi2_test_statistics', 'chi2_is_negative', 'p_value',
        'training_diverged_reduced_model', 'training_converged_within_max_epochs_reduced_model',
        'num_epochs_full_model', 'num_epochs_reduced_model', 'num_full_model_restarts',
        'num_genes_used', 'num_introns_used', 'num_lrt_tests_used', 'num_samples',
        'seed', 'git_revision', 'torch_version', 'torch_num_threads', 'hostname',
        'wall_time_seconds',
    ]
    summary_df = summary_df[[column for column in summary_columns if column in summary_df.columns]]

    summary_df.to_csv(args.output_folder / 'summary.tsv', sep='\t', index=False)
    model_param_df.to_csv(args.output_folder / 'model_parameters.tsv', sep='\t', index=False)
    test_results_df.to_csv(args.output_folder / 'test_results.tsv', sep='\t', index=False)
    training_log_df.to_csv(args.output_folder / 'training_log.tsv', sep='\t', index=False)
    with open(args.output_folder / 'run_metadata.json', 'w') as metadata_file:
        json.dump(run_metadata, metadata_file, indent=2)

    num_negative = int(summary_df['chi2_is_negative'].sum())
    print(f'{dataset_name} / {args.model_type} / {args.experiment}: '
          f'{len(summary_df)} LRT results, {num_negative} with negative chi2, '
          f'{wall_time_seconds:.1f} s', flush=True)


if __name__ == '__main__':
    main()
