#!/usr/bin/env Rscript

library(tidyverse)
library(ashr)
library(argparse)


parser <- ArgumentParser()
parser$add_argument("--model_parameters",
                    required = TRUE,
                    help = "Path to TSV file with model parameters.")
parser$add_argument("--output_folder", default = ".",
                    help = "Where to write the output TSV with regularization coefficients.")
parser$add_argument("--max_abs_value", type = "double", default = 10,
                    help = paste("Entries with |value| above this (natural log scale) are excluded from the ash fit.",
                                 "They come from degenerate fits where the likelihood is flat and the optimizer ran off;",
                                 "they are not effect-size observations, and including them both overflows ashr and",
                                 "inflates the estimated prior by an order of magnitude."))
parser$add_argument("--max_se", type = "double", default = 10,
                    help = "Entries with SE above this carry no information about the prior and are excluded.")


args <- parser$parse_args()

model_parameters <- read_tsv(args$model_parameters) |>
  filter(!is.na(feature_name))

regularization_coefficients_df <- tibble(
  parameter_type = character(),
  feature_name = character(),
  prior_sd = double(),
  lambda = double()
)

for (parameter in unique(model_parameters$parameter_type)) {
  df_parameter <- model_parameters |> filter(parameter_type == parameter)
  for (feature in unique(df_parameter$feature_name)) {
    df_feature <- df_parameter |>
      filter(feature_name == feature,
             identifiable == TRUE,
             training_diverged_full_model == FALSE,
             !is.na(value),
             !is.na(SE),
             SE > 0)

    num_before_bounds <- nrow(df_feature)
    df_feature <- df_feature |>
      filter(abs(value) <= args$max_abs_value,
             SE <= args$max_se)
    num_excluded <- num_before_bounds - nrow(df_feature)
    if (num_excluded > 0) {
      message(sprintf(
        "Excluded %d of %d entries (%.1f%%) outside the bounds |value| <= %g and SE <= %g for parameter_type='%s', feature_name='%s'.",
        num_excluded, num_before_bounds, 100 * num_excluded / num_before_bounds,
        args$max_abs_value, args$max_se, parameter, feature
      ))
    }

    if (nrow(df_feature) == 0) {
      message(sprintf(
        "Warning: no identifiable entries with finite SE for parameter_type='%s', feature_name='%s'; skipping.",
        parameter,
        feature
      ))
      next
    }

    ash_results <- ash(betahat = df_feature$value, sebetahat = df_feature$SE)
    prior_sd <- calc_mixsd(ash_results$fitted_g)

    regularization_coefficients_df <- regularization_coefficients_df |> add_row(
      parameter_type = parameter,
      feature_name = feature,
      prior_sd = prior_sd,
      lambda = 1 / (2 * prior_sd^2)
    )

  }
}

write_tsv(regularization_coefficients_df, file.path(args$output_folder, "regularization_coefficients.tsv"))

