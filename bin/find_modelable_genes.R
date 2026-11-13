#!/usr/bin/env Rscript

library(argparse)
library(tidyverse)


parser <- ArgumentParser()
parser$add_argument("--exon_counts_tsv", required = TRUE)
parser$add_argument("--intron_counts_tsv", required = TRUE)
parser$add_argument("--all_genes_csv", required = TRUE)

parser$add_argument("--output_folder", default = '.')

parser$add_argument("--min_count_exons", type = "integer", default = 5)
parser$add_argument("--min_count_introns", type = "integer", default = 5)
parser$add_argument("--min_samples", type = "integer", default = 3)

args <- parser$parse_args()

output_folder <- args$output_folder
dir.create(output_folder, recursive = TRUE, showWarnings = FALSE)

min_count_exons <- args$min_count_exons
min_count_introns <- args$min_count_introns
min_samples <- args$min_samples

intron_counts_df <- read_tsv(args$intron_counts_tsv, show_col_types = FALSE)
all_genes_df <- read_csv(args$all_genes_csv, show_col_types = FALSE)

sample_columns <- setdiff(colnames(intron_counts_df), "intron_id")

intron_summary <- intron_counts_df |>
  mutate(
    gene_id = sub("_[^_]+$", "", intron_id),
    num_intron_supporting_samples = rowSums(across(all_of(sample_columns)) >= min_count_introns),
    is_modelable = num_intron_supporting_samples >= min_samples,
    num_intron_reads = rowSums(across(all_of(sample_columns)))
  ) |>
  select(-all_of(sample_columns))

intron_gene_summary <- intron_summary |>
  group_by(gene_id) |>
  summarise(
    num_introns_in_count_table = n(),
    num_modelable_introns = sum(is_modelable),
    num_intron_reads = sum(num_intron_reads),
    .groups = "drop"
  )

exon_counts_df <- read_tsv(args$exon_counts_tsv, show_col_types = FALSE)

sample_columns_exon <- setdiff(colnames(exon_counts_df), "gene_id")
if (!setequal(sample_columns_exon, sample_columns)) {
  stop("Sample columns differ between exon and intron count tables.")
}

exon_summary <- exon_counts_df |>
  mutate(
    num_exon_supporting_samples = rowSums(across(all_of(sample_columns)) >= min_count_exons),
    num_exon_reads = rowSums(across(all_of(sample_columns)))
  ) |>
  select(-all_of(sample_columns))

gene_summary <- all_genes_df |>
  distinct(gene_id, .keep_all = TRUE) |>
  left_join(
    exon_summary |>
      mutate(in_exon_count_table = TRUE),
    by = "gene_id"
  ) |>
  left_join(intron_gene_summary, by = "gene_id") |>
  mutate(
    in_exon_count_table = replace_na(in_exon_count_table, FALSE),
    num_exon_supporting_samples = replace_na(num_exon_supporting_samples, 0L),
    num_exon_reads = replace_na(num_exon_reads, 0),
    num_introns_in_count_table = replace_na(num_introns_in_count_table, 0L),
    num_modelable_introns = replace_na(num_modelable_introns, 0L),
    num_intron_reads = replace_na(num_intron_reads, 0),
    is_modelable = (
      num_exon_supporting_samples >= min_samples &
        num_modelable_introns > 0
    )
  )

modelable_genes <- gene_summary |>
  filter(is_modelable)

non_modelable_genes <- gene_summary |>
  filter(!is_modelable)

modelable_genes |>
  write_tsv(file.path(output_folder, "modelable_genes.tsv"))

non_modelable_genes |>
  write_tsv(file.path(output_folder, "non_modelable_genes.tsv"))

# A modelable intron can belong to a non-modelable gene, since a gene also has to pass the exon
# criterion. Carrying the gene's verdict on the intron row is what lets a per-intron fit tell which
# of its introns a per-gene fit will go on to use. An intron whose gene is absent from the gene
# table has no gene to be modelable, hence FALSE rather than NA.
intron_summary <- intron_summary |>
  left_join(
    gene_summary |>
      select(gene_id, gene_is_modelable = is_modelable),
    by = "gene_id"
  ) |>
  mutate(gene_is_modelable = replace_na(gene_is_modelable, FALSE))

modelable_introns <- intron_summary |>
  filter(is_modelable)

non_modelable_introns <- intron_summary |>
  filter(!is_modelable)

modelable_introns |> write_tsv(file.path(output_folder, 'modelable_introns.tsv'))
non_modelable_introns |> write_tsv(file.path(output_folder, 'non_modelable_introns.tsv'))

