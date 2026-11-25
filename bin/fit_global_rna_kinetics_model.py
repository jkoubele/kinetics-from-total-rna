#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch

from rna_kinetics.data import (CountDistribution, load_dataset_metadata, load_gene_data_list,
                               concat_gene_data_list)
from rna_kinetics.estimation import (get_global_rna_kinetics_model_results,
                                     get_jackknife_design_status, get_jackknife_ratio_lfc_df,
                                     JackknifeStatus)

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
    parser.add_argument('--count_distribution',
                        type=CountDistribution,
                        choices=list(CountDistribution),
                        required=True,
                        help='Distribution of the exon and intron read counts. The negative '
                             'binomial needs both tables of count dispersions.')
    parser.add_argument('--exon_count_dispersions',
                        type=Path,
                        default=None,
                        help='Path to the TSV file with one count dispersion per gene.')
    parser.add_argument('--intron_count_dispersions',
                        type=Path,
                        default=None,
                        help='Path to the TSV file with one count dispersion per intron.')
    parser.add_argument('--jackknife_drop_sample',
                        default=None,
                        help='Fit one jackknife replicate: the full model only, with this sample '
                             'left out. Writes its rate ratio LFCs instead of the usual outputs. '
                             'Under the plug-in, the coverage fit given has to leave out the same '
                             'sample. The count dispersions are those of all samples.')
    parser.add_argument('--hot_start_state_dict',
                        type=Path,
                        default=None,
                        help='Model state to initialize the full fit from; a jackknife replicate '
                             'is initialized from the fit to all samples.')
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
        dataset_metadata = dataset_metadata.without_sample(args.jackknife_drop_sample)

    gene_data_list = load_gene_data_list(
        modeled_genes_file=args.modeled_genes,
        modeled_introns_file=args.modeled_introns,
        exon_counts_file=args.exon_counts,
        intron_counts_file=args.intron_counts,
        isoform_length_factors_file=args.isoform_length_factors,
        coverage_folder=args.coverage_data_folder,
        sample_names=dataset_metadata.sample_names,
        count_distribution=args.count_distribution,
        exon_count_dispersions_file=args.exon_count_dispersions,
        intron_count_dispersions_file=args.intron_count_dispersions,
    )

    global_gene_data = concat_gene_data_list(gene_data_list)

    # `None` where no fits were given, which is what turns the plug-in off: the model then
    # estimates `lfc_elong_over_splice` and `intercept_pi_logit` from the joint likelihood.
    plug_in_coverage_fits = None
    if args.plug_in_coverage_fits is not None:
        plug_in_coverage_fits = torch.load(args.plug_in_coverage_fits,
                                           weights_only=False).fits

    hot_start_state_dict = None
    if args.hot_start_state_dict is not None:
        hot_start_state_dict = torch.load(args.hot_start_state_dict, weights_only=False)

    results = get_global_rna_kinetics_model_results(
        global_gene_data=global_gene_data,
        dataset_metadata=dataset_metadata,
        plug_in_coverage_fits=plug_in_coverage_fits,
        hot_start_state_dict=hot_start_state_dict,
        verbose=True,
    )

    if args.jackknife_drop_sample is not None:
        get_jackknife_ratio_lfc_df(results, args.jackknife_drop_sample).to_csv(
            args.output_folder / f'jackknife_ratio_lfcs_{args.jackknife_drop_sample}.tsv',
            sep='\t', index=False)
        raise SystemExit(0)

    results.model_parameters.to_csv(args.output_folder / 'model_parameters.tsv', sep='\t',
                                    index=False)
    results.test_results.to_csv(args.output_folder / 'test_results.tsv', sep='\t', index=False)
    # The initialization of the jackknife replicates.
    torch.save(results.state_dict_full_model, args.output_folder / 'model_state_dict.pt')
