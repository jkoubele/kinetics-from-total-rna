#!/usr/bin/env Rscript

library(argparse)
library(tidyverse)

parser <- ArgumentParser()
parser$add_argument("--test_results", required = TRUE,
                    help = "Path to TSV with raw test results (output of FitGlobal*Model).")
parser$add_argument("--model_type", required = TRUE,
                    choices = c("global_rna_kinetics", "global_intron_coverage"),
                    help = "Model type: global_rna_kinetics or global_intron_coverage.")
parser$add_argument("--output_folder", default = ".", help = "Path to output folder.")

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

# The global rna_kinetics model names the tested ratio itself, so the column only needs
# renaming. The global intron coverage model tests its single LFC, which the column already names.
test_results <- test_results |>
  rename(parameter_type = tested_parameter)

test_results <- test_results |>
  mutate(
    l2fc       = lfc / log(2),
    comparison = if_else(
      !is.na(group_1) & !is.na(group_2),
      paste0(group_1, " vs. ", group_2),
      variable
    ),
    sig_label = case_when(
      p_value < 0.001 ~ "***",
      p_value < 0.01  ~ "**",
      p_value < 0.05  ~ "*",
      TRUE            ~ "n.s."
    ),
    significant = p_value < 0.05
  )

comparison_levels <- test_results |>
  distinct(test_id, comparison) |>
  arrange(test_id) |>
  pull(comparison)

test_results <- test_results |>
  mutate(comparison = factor(comparison, levels = rev(comparison_levels)))

test_results |>
  select(-comparison, -sig_label, -significant) |>
  rename(lfc_mle = lfc, l2fc_mle = l2fc) |>
  select(parameter_type, variable, group_1, group_2, l2fc_mle, lfc_mle,
         p_value, chi2_test_statistics, test_id, test_type, everything()) |>
  write_tsv(file.path(output_folder, "test_results.tsv"))

plots_folder <- file.path(output_folder, "plots")
if (!dir.exists(plots_folder)) dir.create(plots_folder, recursive = TRUE)

sig_caption <- "* p<0.05   ** p<0.01   *** p<0.001"

make_global_plot <- function(df, subtitle) {
  p <- ggplot(df, aes(x = l2fc, y = comparison)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.5) +
    geom_point(aes(color = significant), size = 3.5) +
    geom_text(aes(label = sig_label), hjust = -0.4, size = 5, color = "grey30") +
    scale_color_manual(values = c("TRUE" = "#1f77b4", "FALSE" = "grey60"), guide = "none") +
    scale_x_continuous(expand = expansion(mult = c(0.15, 0.25))) +
    labs(x = "log2FC", y = NULL, subtitle = subtitle, caption = sig_caption) +
    theme_bw(base_size = 14) +
    theme(
      strip.background   = element_rect(fill = "grey92", color = "grey70"),
      strip.text         = element_text(face = "bold"),
      plot.subtitle      = element_text(hjust = 0.5),
      plot.caption       = element_text(size = 9, color = "grey40", hjust = 0),
      panel.grid.minor.x = element_line(color = "grey88", linewidth = 0.3),
      panel.grid.minor.y = element_blank(),
      panel.grid.major.y = element_blank(),
    )

  if (length(unique(df$variable)) > 1) {
    p <- p + facet_grid(variable ~ ., scales = "free_y", space = "free_y")
  }
  p
}

for (param_type in unique(test_results$parameter_type)) {
  df       <- test_results |> filter(parameter_type == param_type)
  subtitle <- parameter_type_subtitles[[param_type]] %||% param_type

  plot_height <- max(4, 1.5 + 0.45 * nrow(df))

  p <- make_global_plot(df, subtitle)
  ggsave(file.path(plots_folder, paste0(param_type, ".png")),
         plot = p, bg = "white", width = 9, height = plot_height, dpi = 300)
}