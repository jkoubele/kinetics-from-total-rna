#!/usr/bin/env Rscript

library(argparse)
library(tidyverse)


parser <- ArgumentParser()
parser$add_argument("--input_gene_names", required = TRUE, help = "Path to input CSV file with gene names.")
parser$add_argument("--modelable_genes", required = TRUE, help = "Path to TSV file with modelable genes.")
parser$add_argument("--modelable_introns", required = TRUE, help = "Path to TSV file with modelable introns.")
parser$add_argument("--output_folder", required = TRUE, help = "Directory to write output chunked CSV files.")
parser$add_argument("--gene_chunk_size", type = "integer", default = 100, help = "Number of genes per chunk (default: 100).")
parser$add_argument("--intron_chunk_size", type = "integer", default = 1000, help = "Number of introns per chunk (default: 1000).")
args <- parser$parse_args()


output_folder <- args$output_folder
gene_chunk_size <- args$gene_chunk_size
intron_chunk_size <- args$intron_chunk_size

input_gene_names <- read_csv(args$input_gene_names)
modelable_genes <- read_tsv(args$modelable_genes)
modelable_introns <- read_tsv(args$modelable_introns)

if (!dir.exists(output_folder)) {
  dir.create(output_folder, recursive = TRUE)
}

chunks_output_subfolder = file.path(output_folder, '/gene_chunks')
if (!dir.exists(chunks_output_subfolder)) {
  dir.create(chunks_output_subfolder, recursive = TRUE)
}

intron_chunks_output_subfolder = file.path(output_folder, '/intron_chunks')
if (!dir.exists(intron_chunks_output_subfolder)) {
  dir.create(intron_chunks_output_subfolder, recursive = TRUE)
}

modeled_genes <- modelable_genes |>
  semi_join(input_gene_names, by = "gene_id")

# Every intron of a modeled gene is here, plus introns of genes that were dropped by the gene
# filter. gene_is_modeled separates the two, and replaces the gene_is_modelable column of the input
# because the two differ by this run's gene selection.
modeled_introns <- modelable_introns |>
  semi_join(input_gene_names, by = "gene_id") |>
  select(-any_of("gene_is_modelable")) |>
  mutate(gene_is_modeled = gene_id %in% modeled_genes$gene_id)

modeled_genes |>
  write_tsv(file.path(output_folder, "modeled_genes.tsv"))

modeled_introns |>
  write_tsv(file.path(output_folder, "modeled_introns.tsv"))


n_chunks <- ceiling(nrow(modeled_genes) / gene_chunk_size)
pad_width <- nchar(as.character(n_chunks))  # Determine left-padding

for (i in seq_len(n_chunks)) {
  start <- (i - 1) * gene_chunk_size + 1
  end <- min(i * gene_chunk_size, nrow(modeled_genes))

  chunk <- modeled_genes[start:end,]

  filename <- sprintf("modeled_genes_chunk_%0*d.tsv", pad_width, i)
  output_path <- file.path(chunks_output_subfolder, filename)

  write_tsv(chunk, output_path)
}

n_intron_chunks <- ceiling(nrow(modeled_introns) / intron_chunk_size)
intron_pad_width <- nchar(as.character(n_intron_chunks))

for (i in seq_len(n_intron_chunks)) {
  start <- (i - 1) * intron_chunk_size + 1
  end <- min(i * intron_chunk_size, nrow(modeled_introns))

  chunk <- modeled_introns[start:end,]

  filename <- sprintf("modeled_introns_chunk_%0*d.tsv", intron_pad_width, i)
  output_path <- file.path(intron_chunks_output_subfolder, filename)

  write_tsv(chunk, output_path)
}
