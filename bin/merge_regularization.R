#!/usr/bin/env Rscript

library(argparse)
library(tidyverse)

parser <- ArgumentParser()
parser$add_argument("--regularization_chunks_folder",
                    required = TRUE,
                    help = "Folder containing TSV files with regularization chunk results.")
parser$add_argument("--test_results",
                    required = TRUE,
                    help = "TSV file with test results before regularization.")
parser$add_argument("--output_folder",
                    default = '.',
                    help = 'Path to output folder')

args <- parser$parse_args()

output_folder <- args$output_folder
if (!dir.exists(output_folder)) {
  dir.create(output_folder, recursive = TRUE)
}

regularized_model_parameters_files <- list.files(
  path = args$regularization_chunks_folder,
  pattern = "^regularized_model_parameters.*\\.tsv$",
  full.names = TRUE
)

if (length(regularized_model_parameters_files) == 0) {
  stop("No matching TSV files with regularized model parameters found in the folder: ",
       args$regularization_chunks_folder)
}

regularized_model_parameters_merged_df <- sort(regularized_model_parameters_files) |>
  map(read_tsv) |>
  keep(~nrow(.x) > 0) |>
  list_rbind()

write_tsv(regularized_model_parameters_merged_df,
          file.path(output_folder, 'regularized_model_parameters.tsv'))

test_results <- read_tsv(args$test_results)

# TODO: broken for the three tested ratios that are not model parameters
# (lfc_init_over_elong, lfc_init_over_splice, lfc_elong_over_splice). They have no parameter_type
# row to join against, so the join below yields NA and the replace_na turns it into a regularized
# LFC of exactly 0, which reads as a real estimate of no effect. The intended fix is for Python to
# emit the regularized LFC per tested ratio, rather than having R rebuild it from the parameter
# table. Note the replace_na below is correct for its other case, a reference level contributing
# zero, so the two situations cannot be told apart after the join.
lfc_value_positive <- test_results |>
  left_join(
    regularized_model_parameters_merged_df,
    by = c(
      "tested_parameter" = "parameter_type",
      "gene_name",
      "intron_name",
      "lfc_column_positive" = "feature_name"
    )
  ) |>
  mutate(value = replace_na(value, 0.0)) |>
  pull(value)

lfc_value_negative <- test_results |>
  left_join(
    regularized_model_parameters_merged_df,
    by = c(
      "tested_parameter" = "parameter_type",
      "gene_name",
      "intron_name",
      "lfc_column_negative" = "feature_name"
    )
  ) |>
  mutate(value = replace_na(value, 0.0)) |>
  pull(value)

test_results$lfc_regularized <- lfc_value_positive - lfc_value_negative

write_tsv(test_results, file.path(output_folder, 'raw_test_results.tsv'))