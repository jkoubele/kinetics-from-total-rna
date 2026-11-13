#!/usr/bin/env Rscript

# Under the plug-in, lfc_elong_over_splice is held fixed while the read counts are fitted, so
# restricting it restricts nothing the RNA kinetics model is free to move and it has no likelihood
# ratio test there. Its test is the intron coverage model's own. This script row-binds those rows
# into the kinetics results, so that one table covers all six rate ratios, and adds the column
# saying which model each row came from.

library(argparse)
library(tidyverse)

COVERAGE_RATIO <- "lfc_elong_over_splice"
SOURCE_COLUMN <- "source_model"

parser <- ArgumentParser()
parser$add_argument("--test_results",
                    required = TRUE,
                    help = "TSV file with the RNA kinetics model's test results.")
parser$add_argument("--coverage_test_results",
                    required = TRUE,
                    help = "TSV file with the intron coverage model's test results.")
parser$add_argument("--output_file",
                    default = 'test_results_with_coverage_arm.tsv',
                    help = 'Path to write the combined test results to.')

args <- parser$parse_args()

kinetics <- read_tsv(args$test_results, show_col_types = FALSE)
coverage <- read_tsv(args$coverage_test_results, show_col_types = FALSE)

# Both guards are about wiring rather than data. The kinetics table holding the ratio means the
# joint likelihood was fitted, which already tests it and needs no second set of rows; the
# coverage table not holding it means the wrong file was handed over.
if (COVERAGE_RATIO %in% kinetics$rate_ratio) {
  stop("The RNA kinetics test results already cover ", COVERAGE_RATIO,
       ", so they did not come from a plug-in fit.")
}
coverage <- coverage |> filter(rate_ratio == COVERAGE_RATIO)
if (nrow(coverage) == 0) {
  stop("The intron coverage test results hold no ", COVERAGE_RATIO, " rows.")
}

# The coverage rows are projected onto the kinetics table's columns. The two schemas agree for a
# gene-specific and an intron-specific fit; the global coverage model carries an intron_name its
# kinetics counterpart does not, which is empty there, its LFC being shared across the dataset.
missing_columns <- setdiff(colnames(kinetics), colnames(coverage))
if (length(missing_columns) > 0) {
  stop("The intron coverage test results are missing columns the RNA kinetics ones have: ",
       paste(missing_columns, collapse = ", "))
}

combined <- bind_rows(
  kinetics |> mutate(!!SOURCE_COLUMN := "rna_kinetics"),
  coverage |> select(all_of(colnames(kinetics))) |> mutate(!!SOURCE_COLUMN := "intron_coverage")
)

write_tsv(combined, args$output_file)

cat("Added", nrow(coverage), COVERAGE_RATIO, "rows from the intron coverage model to",
    nrow(kinetics), "rows from the RNA kinetics model.\n")
