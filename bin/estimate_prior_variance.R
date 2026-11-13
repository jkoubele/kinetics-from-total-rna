#!/usr/bin/env Rscript

# Estimates the prior covariance of the rate ratio LFCs, per design column, and converts it into
# the L2 penalty weights the regularized refit applies.
#
# The penalty the refit adds is `0.5 * t(theta) %*% Lambda %*% theta` per design column, where
# `theta` holds that column's three parametrized ratios. That is the same thing as a zero-centred
# Gaussian prior on `theta` with covariance `Lambda^-1`, so what has to be estimated is that
# covariance. The matching estimator is a multivariate random-effects model over genes, fitted by
# REML: each gene's three estimates are a draw from the prior, observed with that gene's own
# sampling covariance, as `y_g ~ N(mu, Psi + V_g)`.
#
# REML is restricted, or residual, maximum likelihood (Patterson & Thompson 1971, Biometrika
# 58:545-554). It maximises the likelihood of residual contrasts, combinations of the data whose
# distribution does not involve the fixed effects, so the estimated covariance is corrected for the
# degrees of freedom spent estimating the mean; in one dimension it is the difference between
# dividing by n and by n - 1. Plain maximum likelihood is biased low for a variance component and
# can push one onto the zero boundary, which is why "reml" is passed rather than "ml".
#
# A gene whose estimates are explained by noise contributes almost nothing, which is why there is
# no bound on the standard errors: the fit discounts an imprecise estimate through that weight
# without needing a threshold, and stays stable even for standard errors many orders of magnitude
# above the rest. A finite, strictly positive standard error is still required: a zero sampling
# variance would be an infinite weight, and an infinite one makes the fit fail outright.
#
# Fitting one joint covariance rather than six separate per-ratio priors is what keeps the result
# coherent. Six independently fitted marginals need not describe any joint distribution, and on
# intron-specific fits they do not: they can imply a correlation above one. Every ratio's prior is
# instead read off the single fitted matrix, so consistency holds by construction.
#
# Notation: `Psi`, `tau`, `mu` are mixmeta's, for the between-study covariance, its standard
# deviations, and the pooled mean of a random-effects meta-analysis (Sera, Armstrong, Blangiardo &
# Gasparrini 2019, Stat. Med. 38:5429-5444); genes play the role of studies. `lambda`, `prior_sd`
# and `prior_cov_*` are this pipeline's own names and are columns of the output. Nothing here uses
# the per-rate symbols the model itself is written with.

library(argparse)
library(tidyverse)
library(mixmeta)

parser <- ArgumentParser()
parser$add_argument("--ratio_lfcs",
                    required = TRUE,
                    help = "TSV with one row per rate ratio, intron and design column.")
parser$add_argument("--output_folder", default = ".",
                    help = "Where to write the output TSV with regularization coefficients.")
parser$add_argument("--max_abs_lfc", type = "double", default = 10 * log(2),
                    help = paste("Estimates with |lfc| above this (natural log scale) do not",
                                 "contribute to the prior fit. They are still regularized: the",
                                 "fitted prior is applied to every gene regardless. The default is",
                                 "a 1024-fold change in the rate ratio, the same bound DESeq2",
                                 "applies as abs(x) < 10 on its log2 scale before estimating its",
                                 "own coefficient prior. Beyond it the estimates come from fits",
                                 "whose likelihood has no interior maximum rather than from",
                                 "measured effects."))

args <- parser$parse_args()

output_folder <- args$output_folder
if (!dir.exists(output_folder)) {
  dir.create(output_folder, recursive = TRUE)
}

# The basis the penalty acts on when the input holds the per-gene model's ratios: the other three
# are linear combinations of these, so their priors follow from the same matrix instead of being
# fitted. These are also the three the models parametrize, and the two a prior reads most directly
# in: lfc_init_over_deg is differential expression of the mature RNA, and lfc_elong_over_splice is
# how far the polymerase travels past the junction before the intron is cleaved.
#
# Those two share no rate, so no pairwise difference of this basis is itself one of the six ratios
# and the sampling covariances cannot be rebuilt from variances alone. They are read from the
# cov_with_* columns of ratio_lfcs instead, which Python writes off the same inverse Fisher
# information the standard errors come from. See LFC_PRIOR_BASIS in rna_kinetics/estimation.py.
#
# An input that does not hold all three, such as the intron coverage model's single
# lfc_elong_over_splice, has no such structure to exploit and is fitted one ratio at a time. That
# is a statement about which ratios the input table covers, not about how either model is
# parametrized: lfc_elong_over_splice is a parameter of both. Dispatching on what the input
# contains rather than assuming is what keeps a coverage model from silently receiving no prior at
# all.
BASIS <- c("lfc_init_over_deg", "lfc_elong_over_splice", "lfc_splice_over_deg")

# Every ratio as a contrast of the basis, which is what lets one fitted matrix yield all six
# priors. Because the basis is the parametrization, these are the same coefficients as
# RATIO_AS_LFC_COMBINATION in rna_kinetics/models.py, which is the single place the parametrization
# is spelled out on the Python side; keep the two in step.
CONTRAST <- list(lfc_init_over_deg     = c( 1,  0,  0),
                 lfc_elong_over_splice = c( 0,  1,  0),
                 lfc_splice_over_deg   = c( 0,  0,  1),
                 lfc_elong_over_deg    = c( 0,  1,  1),
                 lfc_init_over_splice  = c( 1,  0, -1),
                 lfc_init_over_elong   = c( 1, -1, -1))

# The ratio_lfcs columns carrying each row's covariance with each basis ratio, in BASIS order.
COVARIANCE_WITH_BASIS <- paste0("cov_with_", BASIS)

# The regularization_coefficients columns the fitted prior covariance is written to, in the same
# order. Derived from BASIS so that moving the basis is one edit rather than four.
PRIOR_COVARIANCE_COLUMNS <- paste0("prior_cov_", sub("^lfc_", "", BASIS))

# The off-diagonal pairs of the basis, as index pairs into it, and the working column each one's
# sampling covariance is carried in while a unit's block is assembled.
BASIS_PAIRS <- combn(seq_along(BASIS), 2, simplify = FALSE)
covariance_column <- function(pair) {
  paste0("cov_of_", BASIS[pair[1]], "_with_", BASIS[pair[2]])
}

# The prior covariance columns as a one-row tibble, so their names follow BASIS rather than being
# spelled out at each of the places a row is built.
prior_covariance_tibble <- function(values = rep(NA_real_, length(BASIS))) {
  as_tibble(set_names(as.list(values), PRIOR_COVARIANCE_COLUMNS))
}

# With no usable estimates there is nothing to learn a prior from, so fall back to a prior so wide
# that it imposes no shrinkage. DESeq2 does the same, returning a prior variance of 1e6 when its
# own finiteness filter empties the input.
UNINFORMATIVE_PRIOR_VARIANCE <- 1e6

# REML needs at least two genes to separate between-gene covariance from sampling covariance.
MIN_UNITS_FOR_FIT <- 2

ratio_lfcs <- read_tsv(args$ratio_lfcs, show_col_types = FALSE) |>
  mutate(usable = lfc_identifiable &
           !training_diverged_full_model &
           is.finite(lfc) &
           is.finite(lfc_se) &
           lfc_se > 0 &
           abs(lfc) <= args$max_abs_lfc)

# One row per unit of the fit, with every ratio's estimate and variance in its own column. A unit
# is a (gene, intron) pair when any ratio is intron-specific, and a gene otherwise. A ratio the
# model shares across a gene's introns is reported once with a null intron_name, so it joins on the
# gene alone and is broadcast onto each of that gene's introns.
build_fit_table <- function(df_feature) {
  intron_level <- df_feature |> filter(!is.na(intron_name))
  units <- if (nrow(intron_level) > 0) {
    distinct(intron_level, gene_name, intron_name)
  } else {
    distinct(df_feature, gene_name) |> mutate(intron_name = NA_character_)
  }
  attach_columns <- function(units, one_ratio) {
    is_shared <- all(is.na(one_ratio$intron_name))
    left_join(units,
              if (is_shared) select(one_ratio, -intron_name) else one_ratio,
              by = if (is_shared) "gene_name" else c("gene_name", "intron_name"))
  }
  for (ratio_name in names(CONTRAST)) {
    units <- attach_columns(units, df_feature |>
      filter(rate_ratio == ratio_name) |>
      transmute(gene_name,
                intron_name,
                !!paste0("lfc_of_", ratio_name) := if_else(usable, lfc, NA_real_),
                !!paste0("var_of_", ratio_name) := if_else(usable, lfc_se^2, NA_real_)))
  }
  # Each basis pair's sampling covariance, read off the first member's row. Python writes the
  # cov_with_* columns from the same inverse Fisher information the standard errors come from, so
  # a ratio's entry against itself is its own variance and only the off-diagonal pairs are needed
  # here. No usability filter is applied: it depends on both members of the pair, and the second
  # one's row is elsewhere in the input. An entry belonging to a member that was filtered out is
  # never read, mixmeta removing that outcome's row and column from the block.
  for (pair in BASIS_PAIRS) {
    units <- attach_columns(units, df_feature |>
      filter(rate_ratio == BASIS[pair[1]]) |>
      transmute(gene_name,
                intron_name,
                !!covariance_column(pair) := .data[[COVARIANCE_WITH_BASIS[pair[2]]]]))
  }
  units
}

# The estimates and their sampling covariances, in the form mixmeta takes.
#
# A basis ratio is dropped for a unit when its own estimate is unusable. mixmeta uses whichever
# outcomes remain, so a partly dropped unit still contributes, and the entries of the block
# belonging to a dropped outcome are never read.
build_estimates_and_covariances <- function(fit_table) {
  num_units <- nrow(fit_table)
  estimates <- as.matrix(fit_table[, paste0("lfc_of_", BASIS)])
  colnames(estimates) <- BASIS
  variances <- as.matrix(fit_table[, paste0("var_of_", BASIS)])
  covariances <- as.matrix(
    fit_table[, vapply(BASIS_PAIRS, covariance_column, character(1))])

  # Filling a dropped outcome's entries with zero is only to keep each block well formed.
  sampling_covariances <- lapply(seq_len(num_units), function(unit) {
    block <- diag(replace_na(variances[unit, ], 0), nrow = length(BASIS))
    for (position in seq_along(BASIS_PAIRS)) {
      pair <- BASIS_PAIRS[[position]]
      block[pair[1], pair[2]] <- block[pair[2], pair[1]] <-
        replace_na(covariances[unit, position], 0)
    }
    block
  })
  list(estimates = estimates, sampling_covariances = sampling_covariances)
}

# The fitted prior for one design column, as one row per rate ratio.
#
# A basis ratio with no usable estimates anywhere carries no information about its own direction,
# and mixmeta cannot estimate a variance for it. Those directions are dropped and the reduced model
# is fitted on the rest, so the ratios lying in the fitted subspace still get a prior and only the
# ones needing a dropped direction fall back.
estimate_prior_for_feature <- function(df_feature) {
  feature <- df_feature$feature_name[1]
  fit_table <- build_fit_table(df_feature)
  prepared <- build_estimates_and_covariances(fit_table)
  estimates <- prepared$estimates

  # mixmeta factorises every unit's sampling covariance, so one unit whose block is not positive
  # definite stops the fit for the whole design column. The block is a principal submatrix of
  # C %*% Sigma %*% t(C) and so inherits positive definiteness from the inverse Fisher information
  # it came from, which a fit that stopped away from a maximum of the likelihood does not give.
  # Those units are dropped rather than repaired, because the values they carry are not valid
  # estimates of a standard error. The test is the factorisation itself, since no bound on the
  # entries detects it: the block can have every variance comfortably positive and every pairwise
  # correlation below one and still not be a covariance matrix.
  block_is_positive_definite <- function(unit) {
    present <- which(!is.na(estimates[unit, ]))
    if (length(present) == 0) {
      return(TRUE)
    }
    block <- prepared$sampling_covariances[[unit]][present, present, drop = FALSE]
    !inherits(try(chol(block), silent = TRUE), "try-error")
  }
  keep <- vapply(seq_len(nrow(estimates)), block_is_positive_definite, logical(1))
  estimates <- estimates[keep, , drop = FALSE]
  prepared$sampling_covariances <- prepared$sampling_covariances[keep]

  is_fittable <- colSums(!is.na(estimates)) >= MIN_UNITS_FOR_FIT

  counts <- tibble(num_units_used = sum(rowSums(!is.na(estimates)) > 0),
                   num_units_total = nrow(fit_table),
                   num_estimates_used = sum(!is.na(estimates)),
                   num_units_not_positive_definite = sum(!keep))
  uninformative <- function(ratio_names, status) {
    tibble(feature_name = feature,
           rate_ratio = ratio_names,
           prior_sd = sqrt(UNINFORMATIVE_PRIOR_VARIANCE),
           tau = NA_real_,
           mu = NA_real_,
           status = status) |>
      bind_cols(prior_covariance_tibble()) |>
      bind_cols(counts)
  }
  if (!any(is_fittable)) {
    return(uninformative(names(CONTRAST), "no_usable_estimates"))
  }

  fitted_index <- which(is_fittable)
  fit <- tryCatch(mixmeta(estimates[, fitted_index, drop = FALSE] ~ 1,
                          S = lapply(prepared$sampling_covariances,
                                     function(block) block[fitted_index, fitted_index, drop = FALSE]),
                          method = "reml"),
                  error = function(e) NULL)
  if (is.null(fit)) {
    return(uninformative(names(CONTRAST), "fit_failed"))
  }

  between_gene_covariance <- as.matrix(fit$Psi)
  pooled_mean <- as.numeric(coef(fit))
  # The penalty shrinks each LFC toward zero rather than toward the fitted mean, so the prior
  # covariance is the second moment about zero, Psi + mu %*% t(mu). Shrinking toward the fitted
  # mean would break two things: the effect size is reported next to an LRT p-value for an LFC of
  # zero, so a gene with no effect would carry a non-zero regularized one, and a mean fitted per
  # ratio is not consistent with the differences between ratios. `mu` is reported because a large
  # value means either a genuine genome-wide shift or a normalisation problem.
  prior_covariance <- between_gene_covariance + outer(pooled_mean, pooled_mean)
  is_singular <- min(eigen(prior_covariance, only.values = TRUE)$values) <= 0

  rows <- map_dfr(names(CONTRAST), function(ratio_name) {
    contrast <- CONTRAST[[ratio_name]]
    # A ratio is estimable only if it puts no weight on a dropped direction.
    if (any(contrast[!is_fittable] != 0)) {
      return(uninformative(ratio_name, "not_estimable"))
    }
    fitted_contrast <- contrast[fitted_index]
    covariance_with_basis <- set_names(rep(NA_real_, length(BASIS)), BASIS)
    covariance_with_basis[fitted_index] <- as.numeric(fitted_contrast %*% prior_covariance)
    tibble(feature_name = feature,
           rate_ratio = ratio_name,
           prior_sd = sqrt(as.numeric(t(fitted_contrast) %*% prior_covariance %*% fitted_contrast)),
           tau = sqrt(as.numeric(t(fitted_contrast) %*% between_gene_covariance %*% fitted_contrast)),
           mu = sum(fitted_contrast * pooled_mean),
           # A prior of zero width in a direction asserts those effects are exactly zero, which
           # gives an infinite penalty and a non-finite loss in the refit. It is a failed fit, not
           # a finding.
           status = if (is_singular || !is.finite(prior_sd) || prior_sd <= 0)
             "degenerate_prior" else "ok") |>
      bind_cols(prior_covariance_tibble(covariance_with_basis)) |>
      bind_cols(counts)
  })
  rows
}

# One prior per ratio, for an input with no joint structure to fit over.
estimate_priors_one_ratio_at_a_time <- function(df_feature) {
  feature <- df_feature$feature_name[1]
  map_dfr(sort(unique(df_feature$rate_ratio)), function(ratio_name) {
    usable <- df_feature |> filter(rate_ratio == ratio_name, usable)
    row <- tibble(feature_name = feature,
                  rate_ratio = ratio_name,
                  prior_sd = sqrt(UNINFORMATIVE_PRIOR_VARIANCE),
                  tau = NA_real_,
                  mu = NA_real_,
                  status = "no_usable_estimates",
                  num_units_used = nrow(usable),
                  num_units_total = sum(df_feature$rate_ratio == ratio_name),
                  num_estimates_used = nrow(usable),
                  # Each unit's sampling covariance is a single positive variance here, so there is
                  # no block that can fail to be positive definite.
                  num_units_not_positive_definite = 0L) |>
      bind_cols(prior_covariance_tibble())
    if (nrow(usable) < MIN_UNITS_FOR_FIT) {
      return(row)
    }
    fit <- tryCatch(mixmeta(usable$lfc, S = usable$lfc_se^2, method = "reml"),
                    error = function(e) NULL)
    if (is.null(fit)) {
      return(row |> mutate(status = "fit_failed"))
    }
    between_gene_variance <- as.numeric(fit$Psi)
    pooled_mean <- as.numeric(coef(fit))
    # Named apart from the column it fills: inside mutate, a bare `prior_sd` would resolve to the
    # column rather than to this value.
    fitted_prior_sd <- sqrt(between_gene_variance + pooled_mean^2)
    row |> mutate(tau = sqrt(between_gene_variance),
                  mu = pooled_mean,
                  prior_sd = fitted_prior_sd,
                  status = if (!is.finite(fitted_prior_sd) || fitted_prior_sd <= 0)
                    "degenerate_prior" else "ok")
  })
}

estimate_priors_for_feature <- if (all(BASIS %in% unique(ratio_lfcs$rate_ratio))) {
  estimate_prior_for_feature
} else {
  estimate_priors_one_ratio_at_a_time
}

regularization_coefficients_df <- ratio_lfcs |>
  group_split(feature_name) |>
  map_dfr(estimate_priors_for_feature) |>
  mutate(lambda = 1 / (2 * prior_sd^2)) |>
  select(rate_ratio, feature_name, prior_sd, lambda, tau, mu,
         starts_with("prior_cov_"), status, starts_with("num_")) |>
  arrange(rate_ratio, feature_name)

degenerate <- regularization_coefficients_df |>
  filter(status == "degenerate_prior") |>
  distinct(feature_name)
if (nrow(degenerate) > 0) {
  stop("Prior covariance estimated as singular for design column(s): ",
       paste(degenerate$feature_name, collapse = ", "),
       ". This would put an infinite penalty on those coefficients.")
}

write_tsv(regularization_coefficients_df,
          file.path(output_folder, "regularization_coefficients.tsv"))

print(as.data.frame(regularization_coefficients_df), digits = 4)
