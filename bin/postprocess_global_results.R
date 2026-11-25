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

# Keyed by rate_ratio, which Python writes: these must stay in step with TestedRatio and the
# LFC attribute names in rna_kinetics/models.py, or the plot falls back to the raw name. Each is a
# log fold change of a ratio of two rates, which is why none of them is a rate change on its own.
rate_ratio_subtitles <- c(
  lfc_init_over_deg     = "LFC(initiation rate) − LFC(degradation rate)",
  lfc_elong_over_deg    = "LFC(elongation speed) − LFC(degradation rate)",
  lfc_splice_over_deg   = "LFC(splicing speed) − LFC(degradation rate)",
  lfc_init_over_elong   = "LFC(initiation rate) − LFC(elongation speed)",
  lfc_init_over_splice  = "LFC(initiation rate) − LFC(splicing speed)",
  lfc_elong_over_splice = "LFC(elongation speed) − LFC(splicing speed)"
)

test_results <- read_tsv(args$test_results)

# Written by AddJackknifeResults, and absent when the jackknife was switched off.
has_jackknife <- "p_value_jackknife" %in% colnames(test_results)
if (!has_jackknife) {
  test_results <- test_results |>
    mutate(lfc_se_jackknife = NA_real_, df_jackknife = NA_real_, p_value_jackknife = NA_real_)
}

# The plots show the jackknife p-value wherever there is one, and the likelihood ratio test's
# otherwise. The latter treats genes as independent replicates and so overstates the evidence for
# a global effect, which is why the caption says which one a plot shows.
test_results <- test_results |>
  mutate(
    l2fc       = lfc / log(2),
    l2fc_se_jackknife = lfc_se_jackknife / log(2),
    comparison = if_else(
      !is.na(group_1) & !is.na(group_2),
      paste0(group_1, " vs. ", group_2),
      variable
    ),
    p_value_plotted = coalesce(p_value_jackknife, p_value_lrt),
    p_value_source  = if_else(is.na(p_value_jackknife), "lrt", "jackknife"),
    sig_label = case_when(
      p_value_plotted < 0.001 ~ "***",
      p_value_plotted < 0.01  ~ "**",
      p_value_plotted < 0.05  ~ "*",
      TRUE                    ~ "n.s."
    ),
    significant = p_value_plotted < 0.05,
    # 95% confidence interval from the jackknife, on the log2 scale.
    ci_half_width = qt(0.975, df_jackknife) * l2fc_se_jackknife
  )

comparison_levels <- test_results |>
  distinct(test_id, comparison) |>
  arrange(test_id) |>
  pull(comparison)

test_results <- test_results |>
  mutate(comparison = factor(comparison, levels = rev(comparison_levels)))

output_table <- test_results |>
  select(-comparison, -sig_label, -significant, -p_value_plotted, -p_value_source,
         -ci_half_width) |>
  rename(lfc_mle = lfc, l2fc_mle = l2fc)
if (has_jackknife) {
  output_table <- output_table |>
    select(rate_ratio, variable, group_1, group_2, l2fc_mle, lfc_mle,
           p_value_jackknife, l2fc_se_jackknife, lfc_se_jackknife, t_statistic_jackknife,
           df_jackknife, jackknife_status, p_value_lrt, chi2_test_statistics, test_id, test_type,
           everything())
} else {
  output_table <- output_table |>
    select(-l2fc_se_jackknife, -lfc_se_jackknife, -df_jackknife, -p_value_jackknife) |>
    select(rate_ratio, variable, group_1, group_2, l2fc_mle, lfc_mle,
           p_value_lrt, chi2_test_statistics, test_id, test_type, everything())
}
write_tsv(output_table, file.path(output_folder, "test_results.tsv"))

plots_folder <- file.path(output_folder, "plots")
if (!dir.exists(plots_folder)) dir.create(plots_folder, recursive = TRUE)

sig_caption <- "* p<0.05   ** p<0.01   *** p<0.001"

get_p_value_caption <- function(p_value_sources) {
  sources <- unique(p_value_sources)
  if (identical(sources, "jackknife")) {
    return("p-values and 95% intervals: delete-one-sample jackknife")
  }
  lrt_caption <- paste0("likelihood ratio test, which treats genes as independent replicates\n",
                        "and so overstates the evidence for a global effect")
  if (identical(sources, "lrt")) {
    return(paste("p-values:", lrt_caption))
  }
  paste0("p-values and 95% intervals: delete-one-sample jackknife where shown;\n",
         "elsewhere ", lrt_caption)
}

make_global_plot <- function(df, subtitle) {
  caption <- paste(sig_caption, get_p_value_caption(df$p_value_source), sep = "\n")
  p <- ggplot(df, aes(x = l2fc, y = comparison)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.5) +
    geom_errorbar(data = df |> filter(!is.na(ci_half_width)),
                  aes(xmin = l2fc - ci_half_width, xmax = l2fc + ci_half_width),
                  orientation = "y", width = 0.15, color = "grey40", linewidth = 0.5) +
    geom_point(aes(color = significant), size = 3.5) +
    geom_text(aes(x = l2fc + coalesce(ci_half_width, 0), label = sig_label),
              hjust = -0.4, size = 5, color = "grey30") +
    scale_color_manual(values = c("TRUE" = "#1f77b4", "FALSE" = "grey60"), guide = "none") +
    scale_x_continuous(expand = expansion(mult = c(0.15, 0.25))) +
    labs(x = "log2FC", y = NULL, subtitle = subtitle, caption = caption) +
    theme_bw(base_size = 14) +
    theme(
      strip.background   = element_rect(fill = "grey92", color = "grey70"),
      strip.text         = element_text(face = "bold"),
      plot.subtitle      = element_text(hjust = 0.5),
      plot.caption       = element_text(size = 9, color = "grey40", hjust = 0),
      plot.caption.position = "plot",
      panel.grid.minor.x = element_line(color = "grey88", linewidth = 0.3),
      panel.grid.minor.y = element_blank(),
      panel.grid.major.y = element_blank(),
    )

  if (length(unique(df$variable)) > 1) {
    p <- p + facet_grid(variable ~ ., scales = "free_y", space = "free_y")
  }
  p
}

for (rate_ratio_name in unique(test_results$rate_ratio)) {
  df       <- test_results |> filter(rate_ratio == rate_ratio_name)
  subtitle <- rate_ratio_subtitles[[rate_ratio_name]] %||% rate_ratio_name

  plot_height <- max(4, 1.5 + 0.45 * nrow(df))

  p <- make_global_plot(df, subtitle)
  ggsave(file.path(plots_folder, paste0(rate_ratio_name, ".png")),
         plot = p, bg = "white", width = 9, height = plot_height, dpi = 300)
}