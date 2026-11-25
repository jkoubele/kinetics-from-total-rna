#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd

from rna_kinetics.estimation import (get_jackknife_design_status, get_jackknife_degrees_of_freedom,
                                     JackknifeStatus)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Write the samples the jackknife of the global models removes, one per row '
                    'in the column dropped_sample: every sample of the design, or none if removing '
                    'a sample makes the design rank deficient, in which case a warning file is '
                    'written as well.')
    parser.add_argument('--design_matrix', type=Path, required=True)
    parser.add_argument('--output_folder', type=Path, default=Path('.'))
    args = parser.parse_args()

    design_matrix_df = pd.read_csv(args.design_matrix, sep='\t',
                                   dtype={'sample': str}).set_index('sample')
    design_matrix = design_matrix_df.values.astype(float)
    status = get_jackknife_design_status(design_matrix)

    samples = design_matrix_df.index.tolist() if status == JackknifeStatus.OK else []
    pd.DataFrame({'dropped_sample': samples}).to_csv(
        args.output_folder / 'jackknife_samples.tsv', sep='\t', index=False)

    if status != JackknifeStatus.OK:
        (args.output_folder / 'jackknife_warning.txt').write_text(
            'No jackknife for the global models: leaving out some sample makes the design rank '
            'deficient, typically because a group has a single sample. Their test results '
            'report the likelihood ratio test only, which treats genes as independent '
            'replicates and so overstates the evidence for a global effect.\n')
    elif get_jackknife_degrees_of_freedom(design_matrix) < 3:
        (args.output_folder / 'jackknife_warning.txt').write_text(
            f'The jackknife of the global models has '
            f'{get_jackknife_degrees_of_freedom(design_matrix)} degree(s) of freedom, so its '
            f'p-values can only detect effects that are large relative to the variation between '
            f'replicates.\n')
