#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from rna_kinetics.data import load_dataset_metadata, load_gene_data_list
from rna_kinetics.estimation import (get_intron_coverage_model_results, CacheForRegularization, StateDict,
                                     PlugInCoverageFit, concat_plug_in_coverage_fits)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeled_genes',
                        type=Path,
                        required=True,
                        help='Path to the input TSV file with modeled genes.')
    parser.add_argument('--modeled_introns',
                        type=Path,
                        required=True,
                        help='Path to the input TSV file with modeled introns.')
    parser.add_argument('--design_matrix',
                        type=Path,
                        required=True,
                        help='Path to the input TSV file with design matrix.')
    parser.add_argument('--exon_counts',
                        type=Path,
                        required=True,
                        help='Path to the input TSV file with exon counts.')
    parser.add_argument('--intron_counts',
                        type=Path,
                        required=True,
                        help='Path to the input TSV file with intron counts.')
    parser.add_argument('--library_size_factors',
                        type=Path,
                        required=True,
                        help='Path to the input TSV file with library size factors.')
    parser.add_argument('--isoform_length_factors',
                        type=Path,
                        required=True,
                        help='Path to the input TSV file with isoform length factors.')
    parser.add_argument('--coverage_data_folder',
                        type=Path,
                        required=True,
                        help='Path to the folder with .parquet files with intron coverage.')
    parser.add_argument('--lrt_metadata',
                        type=Path,
                        required=True,
                        help='Path to the TSV file with LRT metadata.')
    parser.add_argument('--reduced_matrices_folder',
                        type=Path,
                        required=True,
                        help='Folder with TSV files with reduced matrices for LRT.')
    parser.add_argument('--output_folder',
                        type=Path,
                        default=Path('.'),
                        help='Output folder.')
    parser.add_argument('--output_name_suffix',
                        type=str,
                        default='',
                        help='Suffix for the output TSV files.')
    args = parser.parse_args()

    args.output_folder.mkdir(exist_ok=True, parents=True)

    dataset_metadata = load_dataset_metadata(
        design_matrix_file=args.design_matrix,
        library_size_factors_file=args.library_size_factors,
        lrt_metadata_file=args.lrt_metadata,
        reduced_matrices_folder=args.reduced_matrices_folder,
    )

    gene_data_list = load_gene_data_list(
        modeled_genes_file=args.modeled_genes,
        modeled_introns_file=args.modeled_introns,
        exon_counts_file=args.exon_counts,
        intron_counts_file=args.intron_counts,
        isoform_length_factors_file=args.isoform_length_factors,
        coverage_folder=args.coverage_data_folder,
        sample_names=dataset_metadata.sample_names,
    )

    model_params_list: list[pd.DataFrame] = []
    ratio_lfcs_list: list[pd.DataFrame] = []
    test_results_list: list[pd.DataFrame] = []
    state_dicts_list: list[StateDict] = []
    plug_in_coverage_fits: list[PlugInCoverageFit] = []

    for gene_data in tqdm(gene_data_list):
        results = get_intron_coverage_model_results(
            coverage=gene_data.coverage,
            dataset_metadata=dataset_metadata,
            intron_names=gene_data.intron_names,
        )
        model_param_df = results.model_parameters.reset_index(drop=True)
        model_param_df['gene_name'] = gene_data.gene_name
        results.ratio_lfcs['gene_name'] = gene_data.gene_name
        results.test_results['gene_name'] = gene_data.gene_name
        model_params_list.append(model_param_df)
        ratio_lfcs_list.append(results.ratio_lfcs)
        test_results_list.append(results.test_results)
        state_dicts_list.append(results.state_dict_full_model)
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
        training_input_per_gene=list(zip(gene_data_list, state_dicts_list)),
        dataset_metadata=dataset_metadata,
        # One fit over the gene's introns, so its LFC is reported under no intron name.
        intron_specific_lfc=False,
    )
    torch.save(cache_for_regularization,
               args.output_folder / f'cache_for_regularization{args.output_name_suffix}.pt')
    torch.save(concat_plug_in_coverage_fits(plug_in_coverage_fits,
                                            dataset_metadata.feature_names),
               args.output_folder / f'plug_in_coverage_fits{args.output_name_suffix}.pt')