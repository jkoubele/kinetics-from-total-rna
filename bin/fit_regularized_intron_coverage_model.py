#!/usr/bin/env python3

import argparse
import pandas as pd
import torch
from pathlib import Path

from tqdm import tqdm

from rna_kinetics.estimation import (get_regularized_intron_coverage_model_results,
                                     PlugInCoverageFit, concat_plug_in_coverage_fits)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_for_regularization',
                        type=Path,
                        required=True,
                        help='Path to the pickle file with CacheForRegularization object.')
    parser.add_argument('--regularization_coefficients',
                        type=Path,
                        required=True,
                        help='Path to the TSV file with regularization coefficients.')
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

    regularization_coefficients_df = pd.read_csv(args.regularization_coefficients, sep='\t')

    device = 'cpu'
    cache_for_regularization = torch.load(args.cache_for_regularization,
                                          weights_only=False,
                                          map_location=device)
    dataset_metadata = cache_for_regularization.dataset_metadata

    regularized_model_params_list = []
    regularized_ratio_lfcs_list = []
    plug_in_coverage_fits: list[PlugInCoverageFit] = []
    for input_data, model_state_dict in tqdm(cache_for_regularization.training_input_per_gene):
        results = get_regularized_intron_coverage_model_results(
            input_data=input_data,
            dataset_metadata=dataset_metadata,
            hot_start_state_dict=model_state_dict,
            regularization_coefficients_df=regularization_coefficients_df,
            intron_specific_lfc=cache_for_regularization.intron_specific_lfc,
            device=device,
        )
        regularized_model_params_list.append(results.model_parameters)
        regularized_ratio_lfcs_list.append(results.ratio_lfcs)
        plug_in_coverage_fits.append(results.plug_in_coverage_fit)

    pd.concat(regularized_model_params_list).reset_index(drop=True).to_csv(
        args.output_folder / f'regularized_model_parameters{args.output_name_suffix}.tsv',
        sep='\t',
        index=False,
    )
    pd.concat(regularized_ratio_lfcs_list).reset_index(drop=True).to_csv(
        args.output_folder / f'regularized_ratio_lfcs{args.output_name_suffix}.tsv',
        sep='\t',
        index=False,
    )
    torch.save(concat_plug_in_coverage_fits(plug_in_coverage_fits,
                                            dataset_metadata.feature_names,
                                            is_regularized=True),
               args.output_folder /
               f'regularized_plug_in_coverage_fits{args.output_name_suffix}.pt')