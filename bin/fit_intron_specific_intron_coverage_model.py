#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from rna_kinetics.data import load_dataset_metadata, load_intron_coverage_list
from rna_kinetics.estimation import (get_intron_coverage_model_results, CacheForRegularization,
                                     PlugInCoverageFit, concat_plug_in_coverage_fits)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeled_introns', type=Path, required=True,
                        help='TSV with intron IDs and gene IDs to model.')
    parser.add_argument('--modeled_genes', type=Path, default=None,
                        help='Optional TSV file with modeled genes. When given, only introns of '
                             'those genes are fitted, which lets this fit be driven by the same '
                             'gene chunk file as the per-gene models.')
    parser.add_argument('--design_matrix', type=Path, required=True)
    parser.add_argument('--library_size_factors', type=Path, required=True)
    parser.add_argument('--coverage_data_folder', type=Path, required=True)
    parser.add_argument('--lrt_metadata', type=Path, required=True)
    parser.add_argument('--reduced_matrices_folder', type=Path, required=True)
    parser.add_argument('--output_folder', type=Path, default=Path('.'))
    parser.add_argument('--output_name_suffix', type=str, default='')
    args = parser.parse_args()

    args.output_folder.mkdir(exist_ok=True, parents=True)

    dataset_metadata = load_dataset_metadata(
        design_matrix_file=args.design_matrix,
        library_size_factors_file=args.library_size_factors,
        lrt_metadata_file=args.lrt_metadata,
        reduced_matrices_folder=args.reduced_matrices_folder,
    )

    intron_data_list = load_intron_coverage_list(
        modeled_introns_file=args.modeled_introns,
        modeled_genes_file=args.modeled_genes,
        coverage_folder=args.coverage_data_folder,
        sample_names=dataset_metadata.sample_names,
    )

    model_params_list: list[pd.DataFrame] = []
    ratio_lfcs_list: list[pd.DataFrame] = []
    test_results_list: list[pd.DataFrame] = []
    training_inputs: list[tuple] = []
    plug_in_coverage_fits: list[PlugInCoverageFit] = []

    for intron_data in tqdm(intron_data_list):
        results = get_intron_coverage_model_results(
            coverage=intron_data.coverage,
            dataset_metadata=dataset_metadata,
            intron_names=intron_data.intron_names,
        )
        model_param_df = results.model_parameters.reset_index(drop=True)
        model_param_df['gene_name'] = intron_data.gene_name
        results.ratio_lfcs['gene_name'] = intron_data.gene_name
        results.test_results['gene_name'] = intron_data.gene_name

        model_params_list.append(model_param_df)
        ratio_lfcs_list.append(results.ratio_lfcs)
        test_results_list.append(results.test_results)
        training_inputs.append((intron_data, results.state_dict_full_model))
        plug_in_coverage_fits.append(results.plug_in_coverage_fit)

    pd.concat(model_params_list).reset_index(drop=True).to_csv(
        args.output_folder / f'model_parameters{args.output_name_suffix}.tsv', sep='\t', index=False
    )
    pd.concat(ratio_lfcs_list).reset_index(drop=True).to_csv(
        args.output_folder / f'ratio_lfcs{args.output_name_suffix}.tsv', sep='\t', index=False
    )
    pd.concat(test_results_list).reset_index(drop=True).to_csv(
        args.output_folder / f'test_results{args.output_name_suffix}.tsv', sep='\t', index=False
    )

    cache_for_regularization = CacheForRegularization(
        training_input_per_gene=training_inputs,
        dataset_metadata=dataset_metadata,
    )
    torch.save(cache_for_regularization,
               args.output_folder / f'cache_for_regularization{args.output_name_suffix}.pt')
    torch.save(concat_plug_in_coverage_fits(plug_in_coverage_fits,
                                            dataset_metadata.feature_names),
               args.output_folder / f'plug_in_coverage_fits{args.output_name_suffix}.pt')