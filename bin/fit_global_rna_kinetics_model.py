#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch

from rna_kinetics.data import load_dataset_metadata, load_gene_data_list, concat_gene_data_list
from rna_kinetics.estimation import get_global_rna_kinetics_model_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeled_genes', type=Path, required=True)
    parser.add_argument('--modeled_introns', type=Path, required=True)
    parser.add_argument('--design_matrix', type=Path, required=True)
    parser.add_argument('--exon_counts', type=Path, required=True)
    parser.add_argument('--intron_counts', type=Path, required=True)
    parser.add_argument('--library_size_factors', type=Path, required=True)
    parser.add_argument('--isoform_length_factors', type=Path, required=True)
    parser.add_argument('--coverage_data_folder', type=Path, required=True)
    parser.add_argument('--lrt_metadata', type=Path, required=True)
    parser.add_argument('--reduced_matrices_folder', type=Path, required=True)
    parser.add_argument('--output_folder', type=Path, default=Path('.'))
    parser.add_argument('--plug_in_coverage_fits',
                        type=Path,
                        default=None,
                        help='Path to the merged intron coverage fits, whose '
                             'lfc_elong_over_splice and intercept_pi_logit are then held fixed. '
                             'Omitting it fits the joint likelihood over the counts and the '
                             'coverage shape together.')
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

    global_gene_data = concat_gene_data_list(gene_data_list)


    # `None` where no fits were given, which is what turns the plug-in off: the model then
    # estimates `lfc_elong_over_splice` and `intercept_pi_logit` from the joint likelihood.
    plug_in_coverage_fits = None
    if args.plug_in_coverage_fits is not None:
        plug_in_coverage_fits = torch.load(args.plug_in_coverage_fits,
                                           weights_only=False).fits

    model_param_df, test_results_df = get_global_rna_kinetics_model_results(
        global_gene_data=global_gene_data,
        dataset_metadata=dataset_metadata,
        plug_in_coverage_fits=plug_in_coverage_fits,
        verbose=True,
    )

    model_param_df.to_csv(args.output_folder / 'model_parameters.tsv', sep='\t', index=False)
    test_results_df.to_csv(args.output_folder / 'test_results.tsv', sep='\t', index=False)
