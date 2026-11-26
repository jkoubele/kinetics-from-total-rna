#!/usr/bin/env Rscript

library(argparse)
library(tidyverse)

parser <- ArgumentParser()
parser$add_argument("--test_results", required = TRUE,
                    help = "Path to TSV with raw test results (output of merge_regularization.R).")
parser$add_argument("--model_type", required = TRUE,
                    choices = c("rna_kinetics", "intron_coverage"),
                    help = "Model type: rna_kinetics or intron_coverage.")
parser$add_argument("--p_value_test", required = TRUE,
                    choices = c("wald", "lrt"),
                    help = paste("Test the volcano plots use for the model's own rows. Rows taken",
                                 "over from the intron coverage model (source_model) always use",
                                 "their likelihood ratio test."))
parser$add_argument("--output_folder", default = '.', help = "Path to output folder.")

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

test_results <- test_results |>
  mutate(
    l2fc_unregularized = lfc / log(2),
    l2fc_regularized   = lfc_regularized / log(2),
    # Python reports the Wald standard error on the natural-log scale of the LFC itself.
    l2fc_se            = lfc_se / log(2),
  ) |>
  rename(lfc_unregularized = lfc) |>
  select(gene_name, intron_name, rate_ratio, variable, group_1, group_2,
         l2fc_unregularized, l2fc_regularized, l2fc_se, p_value_lrt, p_value_wald,
         chi2_test_statistics, test_id, test_type, everything())

write_tsv(test_results, file.path(output_folder, "test_results.tsv"))

# The test behind each row's plotted p-value. Under the plug-in, the coverage arm's rows sit beside
# the kinetics model's own, but each volcano plot covers one rate_ratio and so a single test.
has_coverage_arm <- "source_model" %in% colnames(test_results)
test_results <- test_results |>
  mutate(
    from_coverage_arm = if (has_coverage_arm) coalesce(source_model == "intron_coverage", FALSE)
                        else FALSE,
    plotted_test      = if_else(from_coverage_arm, "lrt", args$p_value_test)
  )

test_names <- c(
  wald = "Wald test",
  lrt  = "likelihood ratio test"
)

get_test_caption <- function(plotted_test, from_coverage_arm) {
  if (from_coverage_arm) {
    return("Likelihood ratio test of the intron coverage model, from which this ratio is taken over.")
  }
  if (plotted_test == "wald" && has_coverage_arm) {
    return(paste("Wald test, with standard errors that account for the estimate taken over from",
                 "the intron coverage model."))
  }
  paste0(str_to_sentence(test_names[[plotted_test]]), ".")
}

min_p_value      <- 1e-20
fdr_threshold    <- 0.05
l2fc_clamp_value <- 5

make_volcano_plots <- function(test_results, base_output_subfolder, lfc_col, lfc_label) {
  for (plot_rate_ratio in unique(test_results$rate_ratio)) {
    output_subfolder <- file.path(base_output_subfolder, plot_rate_ratio)
    if (!dir.exists(output_subfolder)) dir.create(output_subfolder, recursive = TRUE)

    for (plot_test_id in unique(test_results$test_id)) {
      plot_test_results <- test_results |>
        filter(rate_ratio == plot_rate_ratio, test_id == plot_test_id)
      plotted_test <- unique(plot_test_results$plotted_test)
      stopifnot(length(plotted_test) == 1)
      test_caption <- get_test_caption(plotted_test, all(plot_test_results$from_coverage_arm))

      # A unit the test could not be computed for (no p-value, e.g. an LFC the Wald test reports
      # as not identifiable) is left out of the FDR and the plot, and counted in the subtitle.
      plot_test_results <- plot_test_results |>
        mutate(p_value_plotted = if (.env$plotted_test == "wald") p_value_wald else p_value_lrt)
      num_not_testable <- sum(is.na(plot_test_results$p_value_plotted))
      plot_test_results <- plot_test_results |>
        filter(!is.na(p_value_plotted)) |>
        mutate(
          p_value_clipped = pmax(p_value_plotted, min_p_value),
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

      count_category <- function(category) sum(plot_test_results$category == category)
      cat_text <- paste0("down: ", count_category("down"), " | ",
                         "not significant: ", count_category("not_significant"), " | ",
                         "up: ", count_category("up"))
      if (num_not_testable > 0) {
        cat_text <- paste0(cat_text, " | not testable: ", num_not_testable)
      }

      group_1 <- plot_test_results |>
        summarize(v = if (all(is.na(group_1))) NA else first(group_1, default = NA)) |>
        pull(v)
      group_2 <- plot_test_results |>
        summarize(v = if (all(is.na(group_2))) NA else first(group_2, default = NA)) |>
        pull(v)
      tested_variable <- plot_test_results |> summarize(v = first(variable)) |> pull(v)

      subtitle_base <- rate_ratio_subtitles[[plot_rate_ratio]] %||%
        (str_replace_all(plot_rate_ratio, "_", " ") |> str_to_sentence())
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
          y        = paste0("-log10(FDR), ", test_names[[plotted_test]]),
          color    = "",
          caption  = test_caption
        ) +
        theme_classic() +
        theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5),
              plot.caption = element_text(size = 9, color = "grey40", hjust = 0))

      file_name <- tested_variable
      if (!is.na(group_1) && !is.na(group_2)) {
        file_name <- paste0(plot_rate_ratio, "_", file_name, "_", group_1, "_vs_", group_2)
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