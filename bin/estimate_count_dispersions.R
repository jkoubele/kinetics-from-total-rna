#!/usr/bin/env Rscript

# Estimates the dispersions of the negative binomial read count model, one per gene for its exon
# count and one per intron for its intron count. The RNA kinetics models take them as known, in
# the same way DESeq2 itself takes its dispersion estimates as known when it tests a coefficient.
#
# The two count tables are fitted separately, so each has its own dispersion-mean trend to shrink
# towards. Each count gets the dispersion of exactly the quantity whose likelihood the model
# writes: a gene's introns are not summed, because summing averages out the part of the
# variability that is specific to one intron.

library(argparse)
library(tidyverse)
library(DESeq2)

parser <- ArgumentParser()
parser$add_argument("--samplesheet", required = TRUE,
                    help = "CSV with sample annotation. Must include a 'sample' column.")
parser$add_argument("--formula", required = TRUE,
                    help = "Design formula using standard R syntax (e.g. '~ age + genotype').")
parser$add_argument("--exon_counts", required = TRUE, help = "TSV file with exon counts per gene.")
parser$add_argument("--intron_counts", required = TRUE, help = "TSV file with read counts per intron.")
parser$add_argument("--library_size_factors", required = TRUE,
                    help = "TSV file with library size factors.")
parser$add_argument("--isoform_length_factors", required = TRUE,
                    help = "TSV file with isoform length factors per gene and sample.")
parser$add_argument("--modelable_genes", required = TRUE, help = "TSV file with modelable genes.")
parser$add_argument("--modelable_introns", required = TRUE, help = "TSV file with modelable introns.")
parser$add_argument("--output_folder", default = ".", help = "Where to write the dispersion tables.")
args <- parser$parse_args()

output_folder <- args$output_folder
dir.create(output_folder, recursive = TRUE, showWarnings = FALSE)

samplesheet <- read_csv(args$samplesheet, show_col_types = FALSE, col_types = cols(sample = col_character()))
if (!"sample" %in% names(samplesheet)) {
  stop("samplesheet must contain a column named 'sample'.")
}
sample_names <- samplesheet$sample

# The same call create_design_matrices.R makes, with the intercept kept. The models carry their
# own intercepts and are handed a design without one, while here the intercept has to be among
# the columns for the fitted means to span what the models fit.
design_matrix <- model.matrix(as.formula(args$formula), samplesheet)
rownames(design_matrix) <- sample_names

library_size_factors <- read_tsv(args$library_size_factors, show_col_types = FALSE,
                                 col_types = cols(sample = col_character())) |>
  deframe()
library_size_factors <- library_size_factors[sample_names]

# Every modelable gene and intron, rather than the ones a given run goes on to model. A count's
# dispersion is shrunk towards a trend fitted over all the rows, so estimating over a run's own
# gene selection would make one gene's result depend on which other genes were selected with it,
# and would leave a short custom gene list with too few rows to fit a trend to.
modelable_genes <- read_tsv(args$modelable_genes, show_col_types = FALSE)
# Introns of genes that failed the gene filter are included too. Their counts are never modeled,
# but they are counts of the same kind, so they inform the trend the other introns are shrunk to.
modelable_introns <- read_tsv(args$modelable_introns, show_col_types = FALSE)

read_count_matrix <- function(counts_file, id_column, ids) {
  counts_matrix <- read_tsv(counts_file, show_col_types = FALSE) |>
    column_to_rownames(id_column) |>
    as.matrix()
  missing_ids <- setdiff(ids, rownames(counts_matrix))
  if (length(missing_ids) > 0) {
    stop(sprintf("%d modelable rows are missing from %s, e.g. %s.",
                 length(missing_ids), counts_file, missing_ids[1]))
  }
  counts_matrix[ids, sample_names, drop = FALSE]
}

# Exon counts are Salmon estimates and so not integers, which DESeq2 requires. Rounding is what
# tximport does for the same reason.
exon_counts <- read_count_matrix(args$exon_counts, "gene_id", modelable_genes$gene_id) |> round()
intron_counts <- read_count_matrix(args$intron_counts, "intron_id", modelable_introns$intron_id) |> round()

isoform_length_factors <- read_count_matrix(args$isoform_length_factors, "gene_id", modelable_genes$gene_id)

# The offsets of the models' own means: the library size for every count, and for the exon counts
# the isoform length factor on top of it. Set rather than re-estimated, so that the dispersions
# describe the variability around the means the models will fit.
exon_normalization_factors <- t(t(isoform_length_factors) * library_size_factors)
intron_normalization_factors <- matrix(
  library_size_factors, nrow = nrow(intron_counts), ncol = length(sample_names), byrow = TRUE,
  dimnames = dimnames(intron_counts)
)

residual_degrees_of_freedom <- nrow(design_matrix) - qr(design_matrix)$rank

estimate_dispersions <- function(counts_matrix, normalization_factors, plot_file) {
  dds <- DESeqDataSetFromMatrix(
    countData = counts_matrix,
    colData = DataFrame(row.names = colnames(counts_matrix)),
    design = design_matrix
  )
  normalizationFactors(dds) <- normalization_factors
  dds <- estimateDispersions(dds, quiet = TRUE)

  png(plot_file, width = 1600, height = 1300, res = 200)
  plotDispEsts(dds)
  dev.off()

  as_tibble(mcols(dds)) |>
    transmute(
      dispersion = dispersion,
      mean_normalized_count = baseMean,
      dispersion_gene_estimate = dispGeneEst,
      dispersion_trend = dispFit,
      dispersion_is_outlier = dispOutlier
    )
}

# Without replicates there is no variability left over to estimate a dispersion from. A
# dispersion of zero is the Poisson distribution, which is what the models then fit.
get_zero_dispersions <- function(counts_matrix) {
  tibble(
    dispersion = rep(0, nrow(counts_matrix)),
    mean_normalized_count = NA_real_,
    dispersion_gene_estimate = NA_real_,
    dispersion_trend = NA_real_,
    dispersion_is_outlier = NA
  )
}

if (residual_degrees_of_freedom > 0) {
  exon_dispersions <- estimate_dispersions(
    exon_counts, exon_normalization_factors,
    file.path(output_folder, "exon_count_dispersions.png"))
  intron_dispersions <- estimate_dispersions(
    intron_counts, intron_normalization_factors,
    file.path(output_folder, "intron_count_dispersions.png"))
} else {
  warning_message <- sprintf(
    paste("The design '%s' has as many coefficients as there are samples (%d), which leaves no",
          "replicates to estimate count dispersions from. All dispersions are set to zero, so the",
          "read counts are modeled as Poisson."),
    args$formula, nrow(design_matrix))
  warning(warning_message)
  writeLines(warning_message, file.path(output_folder, "count_dispersions_warning.txt"))
  exon_dispersions <- get_zero_dispersions(exon_counts)
  intron_dispersions <- get_zero_dispersions(intron_counts)
}

exon_dispersions <- exon_dispersions |>
  mutate(gene_id = rownames(exon_counts), .before = 1)
intron_dispersions <- intron_dispersions |>
  mutate(intron_id = rownames(intron_counts), gene_id = modelable_introns$gene_id, .before = 1)

# Every modelable count has reads in some sample, which is all a dispersion estimate needs, so a
# missing one means the count tables and the modelable rows have come apart.
for (dispersions in list(exon_dispersions, intron_dispersions)) {
  if (any(is.na(dispersions$dispersion))) {
    stop(sprintf("No dispersion could be estimated for %d rows, e.g. %s.",
                 sum(is.na(dispersions$dispersion)),
                 dispersions[[1]][which(is.na(dispersions$dispersion))[1]]))
  }
}

write_tsv(exon_dispersions, file.path(output_folder, "exon_count_dispersions.tsv"))
write_tsv(intron_dispersions, file.path(output_folder, "intron_count_dispersions.tsv"))
