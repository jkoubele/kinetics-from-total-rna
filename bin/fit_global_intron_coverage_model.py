#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from rna_kinetics.data import MODEL_DTYPE, load_dataset_metadata
from rna_kinetics.estimation import (get_intron_coverage_model_results,
                                     concat_plug_in_coverage_fits, get_jackknife_design_status,
                                     get_jackknife_ratio_lfc_df, JackknifeStatus)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeled_introns',
                        type=Path,
                        required=True,
                        help='Path to the input TSV file with modeled introns.')
    parser.add_argument('--design_matrix',
                        type=Path,
                        required=True,
                        help='Path to the input TSV file with design matrix.')
    parser.add_argument('--library_size_factors',
                        type=Path,
                        required=True,
                        help='Path to the input TSV file with library size factors.')
    parser.add_argument('--lrt_metadata',
                        type=Path,
                        required=True,
                        help='Path to the TSV file with LRT metadata.')
    parser.add_argument('--reduced_matrices_folder',
                        type=Path,
                        required=True,
                        help='Folder with TSV files with reduced matrices for LRT.')
    parser.add_argument('--coverage_data_folder',
                        type=Path,
                        required=True,
                        help='Path to the folder with .parquet files with intron coverage.')
    parser.add_argument('--output_folder',
                        type=Path,
                        default=Path('.'),
                        help='Output folder.')
    parser.add_argument('--jackknife_drop_sample',
                        default=None,
                        help='Fit one jackknife replicate: the full model only, with this sample '
                             'left out. Writes its rate ratio LFCs and its coverage fit for the '
                             'matching RNA kinetics replicate instead of the usual outputs.')
    args = parser.parse_args()

    args.output_folder.mkdir(exist_ok=True, parents=True)

    dataset_metadata = load_dataset_metadata(
        design_matrix_file=args.design_matrix,
        library_size_factors_file=args.library_size_factors,
        lrt_metadata_file=args.lrt_metadata,
        reduced_matrices_folder=args.reduced_matrices_folder,
    )
    if args.jackknife_drop_sample is not None:
        # The workflow launches the replicates only for a design that passes this check.
        if (get_jackknife_design_status(dataset_metadata.design_matrix.numpy())
                != JackknifeStatus.OK):
            raise ValueError('Leaving out a sample makes this design rank deficient, so it has no '
                             'jackknife replicates.')
        # The replicate is fitted from scratch rather than warm-started from the fit to all
        # samples. Removing a sample changes the data, so a parameter on a flat region of the loss
        # in the fit to all samples (an intron's pi logit saturated at 0 or 1) can become
        # informative in the replicate while its gradient is still zero, leaving a warm-started
        # replicate away from its optimum.
        dataset_metadata = dataset_metadata.without_sample(args.jackknife_drop_sample)

    modeled_introns_df = pd.read_csv(args.modeled_introns, sep='\t')
    intron_names = modeled_introns_df['intron_id'].tolist()

    coverage_df_by_sample: dict[str, pd.DataFrame] = {}
    for sample in tqdm(dataset_metadata.sample_names, desc='Loading coverage data'):
        sample_coverage_df = pd.read_parquet(
            args.coverage_data_folder / f'{sample}.parquet'
        ).set_index('intron_name')
        coverage_df_by_sample[sample] = sample_coverage_df.loc[intron_names]

    coverage = torch.tensor(
        np.stack([coverage_df_by_sample[sample].values for sample in dataset_metadata.sample_names]),
        dtype=MODEL_DTYPE,
    )

    # No ratio_lfcs here: the global model is not regularized, and with no Fisher information
    # there would be no standard error for the prior fit to use anyway.
    results = get_intron_coverage_model_results(
        coverage=coverage,
        dataset_metadata=dataset_metadata,
        intron_names=intron_names,
        is_global_model=True,
        verbose=True,
    )
    plug_in_coverage_fits = concat_plug_in_coverage_fits([results.plug_in_coverage_fit],
                                                         dataset_metadata.feature_names)

    if args.jackknife_drop_sample is not None:
        get_jackknife_ratio_lfc_df(results, args.jackknife_drop_sample).to_csv(
            args.output_folder / f'jackknife_ratio_lfcs_{args.jackknife_drop_sample}.tsv',
            sep='\t', index=False)
        # Frozen by the RNA kinetics replicate leaving out the same sample, so that the
        # jackknife covers the uncertainty of both stages of the plug-in estimator.
        torch.save(plug_in_coverage_fits, args.output_folder / 'plug_in_coverage_fits.pt')
        raise SystemExit(0)

    results.model_parameters.to_csv(
        args.output_folder / 'model_parameters.tsv', sep='\t', index=False
    )
    results.test_results.to_csv(args.output_folder / 'test_results.tsv', sep='\t', index=False)
    # One fit over every intron in the dataset, so one record, and no covariance: this model
    # computes no Fisher information.
    torch.save(plug_in_coverage_fits, args.output_folder / 'plug_in_coverage_fits.pt')
