#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd

from rna_kinetics.inference import add_jackknife_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add jackknife standard errors and t tests to a global model's test results.")
    parser.add_argument('--test_results', type=Path, required=True)
    parser.add_argument('--design_matrix', type=Path, required=True)
    parser.add_argument('--jackknife_ratio_lfcs', type=Path, nargs='*', default=[],
                        help='One TSV file per jackknife replicate. None where the design has no '
                             'jackknife, which the results then record in jackknife_status.')
    parser.add_argument('--output_folder', type=Path, default=Path('.'))
    args = parser.parse_args()

    design_matrix_df = pd.read_csv(args.design_matrix, sep='\t',
                                   dtype={'sample': str}).set_index('sample')
    test_results = pd.read_csv(args.test_results, sep='\t')
    jackknife_ratio_lfcs = (
        pd.concat([pd.read_csv(file, sep='\t', dtype={'dropped_sample': str})
                   for file in args.jackknife_ratio_lfcs], ignore_index=True)
        if args.jackknife_ratio_lfcs else pd.DataFrame())

    add_jackknife_results(
        test_results=test_results,
        jackknife_ratio_lfcs=jackknife_ratio_lfcs,
        design_matrix=design_matrix_df.values.astype(float),
        sample_names=design_matrix_df.index.tolist(),
    ).to_csv(args.output_folder / 'test_results_with_jackknife.tsv', sep='\t', index=False)

    if args.jackknife_ratio_lfcs:
        jackknife_ratio_lfcs.to_csv(args.output_folder / 'jackknife_ratio_lfcs.tsv', sep='\t',
                                    index=False)
