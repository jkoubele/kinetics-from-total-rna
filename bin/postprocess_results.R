#!/usr/bin/env Rscript

library(argparse)
library(tidyverse)

parser <- ArgumentParser()
parser$add_argument("--test_results", required = TRUE,
                    help = "Path to TSV with raw test results (output of merge_regularization.R).")
parser$add_argument("--model_type", required = TRUE,
                    choices = c("rna_kinetics", "intron_coverage"),
                    help = "Model type: rna_kinetics or intron_coverage.")
parser$add_argument("--output_folder", default = '.', help = "Path to output folder.")

args <- parser$parse_args()

output_folder <- args$output_folder
if (!dir.exists(output_folder)) dir.create(output_folder, recursive = TRUE)

# Keyed by parameter_type, which Python writes: these must stay in step with TestedRatio and the
# LFC attribute names in rna_kinetics/models.py, or the plot falls back to the raw name. Each is a
# log fold change of a ratio of two rates, which is why none of them is a rate change on its own.
parameter_type_subtitles <- c(
  lfc_init_over_deg     = "LFC(initiation rate) − LFC(degradation rate)",
  lfc_elong_over_deg    = "LFC(elongation speed) − LFC(degradation rate)",
  lfc_splice_over_deg   = "LFC(splicing speed) − LFC(degradation rate)",
  lfc_init_over_elong   = "LFC(initiation rate) − LFC(elongation speed)",
  lfc_init_over_splice  = "LFC(initiation rate) − LFC(splicing speed)",
  lfc_elong_over_splice = "LFC(elongation speed) − LFC(splicing speed)"
)

test_results <- read_tsv(args$test_results)

# The rna_kinetics models name the tested ratio themselves, so the column only needs renaming.
# The intron coverage models test their single LFC, which the column already names.
test_results <- test_results |>
  rename(parameter_type = tested_parameter)

test_results <- test_results |>
  mutate(
    l2fc_unregularized = lfc / log(2),
    l2fc_regularized   = lfc_regularized / log(2),
  ) |>
  rename(lfc_unregularized = lfc) |>
  select(gene_name, intron_name, parameter_type, variable, group_1, group_2,
         l2fc_unregularized, l2fc_regularized, p_value, chi2_test_statistics,
         test_id, test_type, everything())

write_tsv(test_results, file.path(output_folder, "test_results.tsv"))

min_p_value      <- 1e-20
fdr_threshold    <- 0.05
l2fc_clamp_value <- 5

make_volcano_plots <- function(test_results, base_output_subfolder, lfc_col, lfc_label) {
  for (plot_parameter_type in unique(test_results$parameter_type)) {
    output_subfolder <- file.path(base_output_subfolder, plot_parameter_type)
    if (!dir.exists(output_subfolder)) dir.create(output_subfolder, recursive = TRUE)

    for (plot_test_id in unique(test_results$test_id)) {
      plot_test_results <- test_results |>
        filter(parameter_type == plot_parameter_type, test_id == plot_test_id) |>
        mutate(
          p_value_clipped = pmax(p_value, min_p_value),
          fdr             = p.adjust(p_value_clipped, method = "fdr"),
          l2fc_plot       = .data[[lfc_col]],
          category        = case_when(
            fdr < fdr_threshold & l2fc_plot > 0 ~ "up",
            fdr < fdr_threshold & l2fc_plot < 0 ~ "down",
            TRUE                                ~ "not_significant"
          ),
          neglog10_fdr = -log10(fdr),
          l2fc_clamped = pmin(pmax(l2fc_plot, -l2fc_clamp_value), l2fc_clamp_value)
        )

      stopifnot(length(unique(plot_test_results$variable)) == 1)
      stopifnot(length(unique(plot_test_results$group_1)) <= 1)
      stopifnot(length(unique(plot_test_results$group_2)) <= 1)

      cat_counts <- table(plot_test_results$category)
      cat_text   <- paste0("down: ", cat_counts["down"], " | ",
                           "not significant: ", cat_counts["not_significant"], " | ",
                           "up: ", cat_counts["up"])

      group_1 <- plot_test_results |>
        summarize(v = if (all(is.na(group_1))) NA else first(group_1, default = NA)) |>
        pull(v)
      group_2 <- plot_test_results |>
        summarize(v = if (all(is.na(group_2))) NA else first(group_2, default = NA)) |>
        pull(v)
      tested_variable <- plot_test_results |> summarize(v = first(variable)) |> pull(v)

      subtitle_base <- parameter_type_subtitles[[plot_parameter_type]] %||%
        (str_replace_all(plot_parameter_type, "_", " ") |> str_to_sentence())
      title_nice <- paste(subtitle_base, "-", tested_variable)
      if (!is.na(group_1) && !is.na(group_2)) {
        title_nice <- paste0(title_nice, ": ", group_1, " vs. ", group_2)
      }

      volcano_plot <- ggplot(plot_test_results, aes(x = l2fc_clamped, y = neglog10_fdr, color = category)) +
        geom_point(alpha = 0.7, size = 1.5) +
        scale_color_manual(values = c(not_significant = "gray70", up = "red", down = "blue")) +
        labs(
          title    = title_nice,
          subtitle = cat_text,
          x        = lfc_label,
          y        = "-log10(FDR)",
          color    = ""
        ) +
        theme_classic() +
        theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5))

      file_name <- tested_variable
      if (!is.na(group_1) && !is.na(group_2)) {
        file_name <- paste0(plot_parameter_type, "_", file_name, "_", group_1, "_vs_", group_2)
      }

      ggsave(file.path(output_subfolder, paste0(file_name, ".png")),
             plot = volcano_plot, bg = "white", width = 10, height = 8, dpi = 300)
    }
  }
}

make_volcano_plots(test_results,
                   file.path(output_folder, "volcano_plots", "mle"),
                   lfc_col   = "l2fc_unregularized",
                   lfc_label = "MLE log2FC estimate")

make_volcano_plots(test_results,
                   file.path(output_folder, "volcano_plots", "regularized"),
                   lfc_col   = "l2fc_regularized",
                   lfc_label = "Regularized log2FC estimate")