#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from rna_kinetics.data import load_dataset_metadata, load_intron_coverage_list
from rna_kinetics.estimation import get_intron_coverage_model_results, CacheForRegularization

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeled_introns', type=Path, required=True,
                        help='TSV with intron IDs and gene IDs to model.')
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
        coverage_folder=args.coverage_data_folder,
        sample_names=dataset_metadata.sample_names,
    )

    model_params_list: list[pd.DataFrame] = []
    test_results_list: list[pd.DataFrame] = []
    training_inputs: list[tuple] = []

    for intron_data in tqdm(intron_data_list):
        model_param_df, test_results_df, state_dict = get_intron_coverage_model_results(
            coverage=intron_data.coverage,
            dataset_metadata=dataset_metadata,
            intron_names=intron_data.intron_names,
        )
        model_param_df = model_param_df.reset_index(drop=True)
        model_param_df['gene_name'] = intron_data.gene_name
        test_results_df['gene_name'] = intron_data.gene_name

        model_params_list.append(model_param_df)
        test_results_list.append(test_results_df)
        training_inputs.append((intron_data, state_dict))

    pd.concat(model_params_list).reset_index(drop=True).to_csv(
        args.output_folder / f'model_parameters{args.output_name_suffix}.tsv', sep='\t', index=False
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