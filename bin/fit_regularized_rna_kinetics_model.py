#!/usr/bin/env python3

import argparse
import pandas as pd
import torch
from pathlib import Path

from tqdm import tqdm

from rna_kinetics.estimation import get_regularized_rna_kinetics_model_results

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
                        help='Suffix for the output CSV files.')

    parser.add_argument('--plug_in_coverage_fits',
                        type=Path,
                        default=None,
                        help='Path to the merged intron coverage fits, whose '
                             'lfc_elong_over_splice and intercept_pi_logit are then held fixed. '
                             'Omitting it fits the joint likelihood over the counts and the '
                             'coverage shape together.')
    args = parser.parse_args()
    output_folder = args.output_folder
    output_folder.mkdir(exist_ok=True, parents=True)

    regularization_coefficients_df = pd.read_csv(args.regularization_coefficients, sep='\t')

    device = 'cpu'
    cache_for_regularization = torch.load(args.cache_for_regularization,
                                          weights_only=False,
                                          map_location=device)
    dataset_metadata = cache_for_regularization.dataset_metadata
    intron_specific_lfc = cache_for_regularization.intron_specific_lfc

    # `None` where no fits were given, which is what turns the plug-in off: the model then
    # estimates `lfc_elong_over_splice` and `intercept_pi_logit` from the joint likelihood.
    plug_in_coverage_fits = None
    if args.plug_in_coverage_fits is not None:
        plug_in_coverage_fits = torch.load(args.plug_in_coverage_fits,
                                           weights_only=False).fits

    result_list = [
        get_regularized_rna_kinetics_model_results(gene_data=gene_data,
                                                   dataset_metadata=dataset_metadata,
                                                   intron_specific_lfc=intron_specific_lfc,
                                                   hot_start_state_dict=model_state_dict,
                                                   regularization_coefficients_df=regularization_coefficients_df,
                                                   plug_in_coverage_fits=plug_in_coverage_fits,
                                                   device=device)
        for gene_data, model_state_dict in tqdm(cache_for_regularization.training_input_per_gene)]

    pd.concat([result.model_parameters for result in result_list]).reset_index(drop=True).to_csv(
        output_folder / f"regularized_model_parameters{args.output_name_suffix}.tsv",
        sep='\t', index=False)
    pd.concat([result.ratio_lfcs for result in result_list]).reset_index(drop=True).to_csv(
        output_folder / f"regularized_ratio_lfcs{args.output_name_suffix}.tsv",
        sep='\t', index=False)
