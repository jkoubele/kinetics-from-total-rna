#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch

from rna_kinetics.estimation import merge_plug_in_coverage_fits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Merge the intron coverage fits of several chunks into the single artifact a '
                    'plug-in fit reads.')
    parser.add_argument('--input_folder',
                        type=Path,
                        default=Path('.'),
                        help='Folder holding the per-chunk plug_in_coverage_fits_*.pt files.')
    parser.add_argument('--output_file',
                        type=Path,
                        default=Path('plug_in_coverage_fits.pt'),
                        help='Path to write the merged fits to.')
    args = parser.parse_args()

    # Sorted so that the merged artifact holds the chunks in a reproducible order. Which order
    # does not matter to a consumer, `PlugInCoverage.from_fits` addressing the fits by intron
    # name, but an arbitrary one would make two runs of the same input differ byte for byte.
    chunk_files = sorted(args.input_folder.glob('plug_in_coverage_fits_*.pt'))
    if not chunk_files:
        raise FileNotFoundError(
            f'No plug_in_coverage_fits_*.pt found in {args.input_folder.resolve()}.')

    merged = merge_plug_in_coverage_fits(
        [torch.load(chunk_file, weights_only=False) for chunk_file in chunk_files])
    torch.save(merged, args.output_file)
    print(f'Merged {len(chunk_files)} chunks into {len(merged.fits)} coverage fits over '
          f'{sum(len(fit.intron_names) for fit in merged.fits)} introns.')
