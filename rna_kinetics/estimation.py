from dataclasses import dataclass
from typing import Callable, Iterator, NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.func import functional_call

from rna_kinetics.data import MODEL_DTYPE, GeneData, IntronData, DatasetMetadata, GlobalGeneData
from rna_kinetics.models import RNAKineticsLoss, RNAKineticsModel, TestedRatio, LRTSpecification, \
    IntronCoverageModel, CoverageLoss, GlobalRNAKineticsModel, RATIO_AS_LFC_COMBINATION, \
    GLOBAL_MODEL_TESTED_RATIOS, GLOBAL_PLUG_IN_TESTABLE_RATIOS, PLUG_IN_TESTABLE_RATIOS, \
    LFCParameter, PlugInCoverage, PlugInCoverageFit, StateDict, get_lfc_of_tested_ratio, \
    get_lfc_of_column, LFC_PRIOR_BASIS
from rna_kinetics.fitting import TrainingResults, make_lbfgs_optimizer, train_model, make_closures, \
    train_global_model
from rna_kinetics.inference import get_fisher_information_matrix, ParameterCovariance, \
    is_positive_definite, try_invert, get_parameter_covariance, get_two_stage_parameter_covariance, \
    get_plug_in_coverage_fit, add_wald_test_results, get_contrast_of_ratio, \
    get_contrast_wald_results, RatioToReport, get_ratio_lfc_df, get_lrt_results


@dataclass
class CacheForRegularization:
    training_input_per_gene: list[tuple[GeneData, StateDict]]
    dataset_metadata: DatasetMetadata
    # Whether the LFC belongs to one intron. For `RNAKineticsModel` it is the LFC parameter's
    # shape; for `IntronCoverageModel` it is whether the fit was per intron, which decides the
    # name the LFC is reported under. Required, so that the refit reads the granularity the fit
    # declared rather than inferring it from the data it was handed.
    intron_specific_lfc: bool


class ModelFitResults(NamedTuple):
    """
    Everything one fit of a full model reports.

    `model_parameters` has one row per model parameter, in the order the Fisher information
    indexes them. `ratio_lfcs` has one row per rate ratio and design column, which is the shape
    the empirical Bayes step consumes, since the L2 penalty acts per column. `test_results` has
    one row per likelihood ratio test. `state_dict_full_model` is the hot start cached for the
    regularized refit.

    `plug_in_coverage_fit` is set by the coverage models and is None for the others, which have no
    coverage-only fit to hand on.
    """
    model_parameters: pd.DataFrame
    ratio_lfcs: pd.DataFrame
    test_results: pd.DataFrame
    state_dict_full_model: StateDict
    plug_in_coverage_fit: Optional[PlugInCoverageFit] = None


class RegularizedFitResults(NamedTuple):
    """
    What one regularized refit reports.

    `ratio_lfcs` matches the shape the unregularized fit emits, minus the standard errors: the
    Hessian of a penalized objective is not an information matrix. It exists so that the
    regularized effect size of every rate ratio can be joined onto the test results directly,
    rather than rebuilt downstream from the parameter table, which has no row for the three ratios
    that are not parameters.

    `plug_in_coverage_fit` is set by the coverage models only, and carries no Fisher information
    for the same reason `ratio_lfcs` carries no standard errors.
    """
    model_parameters: pd.DataFrame
    ratio_lfcs: pd.DataFrame
    plug_in_coverage_fit: Optional[PlugInCoverageFit] = None


class LRTInputs(NamedTuple):
    """
    Everything needed to run and report one likelihood ratio test.

    The tests are enumerated before any fitting so the restart loop can address them by position
    and refit the same set repeatedly, keeping the best attempt for each. `lrt_specification` is
    None for the coverage models, whose single LFC is the only thing they can restrict.
    """
    lrt_metadata_row: pd.Series
    reduced_design_matrix: torch.Tensor
    lrt_specification: Optional[LRTSpecification] = None


def get_projection_onto_design(design_matrix: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Least-squares coefficients of `target` on `[1, design_matrix]`, with the constant dropped.

    The constant column is in the projection but not in the result because the models carry their
    own intercepts: what a design can reproduce is a term up to an additive constant, and which
    intercepts absorb that constant is a per-model question answered by the absorb_* functions.
    Including it is also what makes the projection exact in both directions, since a reduced design
    is nested in the full one only once the constant is there -- relevelling a categorical variable
    leaves the raw column spaces genuinely non-nested.

    `target` is (num_samples,) or (num_samples, k); the result is (num_features,) or
    (num_features, k) to match.
    """
    is_one_dimensional = target.dim() == 1
    constant_column = design_matrix.new_ones(design_matrix.shape[0], 1)
    design_with_constant = torch.cat([constant_column, design_matrix], dim=1)
    solution = torch.linalg.lstsq(
        design_with_constant, target.unsqueeze(1) if is_one_dimensional else target).solution
    coefficients = solution[1:]
    return coefficients.squeeze(1) if is_one_dimensional else coefficients


def get_hot_start_reduced_lfc(design_matrix: torch.Tensor,
                              reduced_design_matrix: torch.Tensor,
                              lfc_of_tested_ratio: torch.Tensor) -> torch.Tensor:
    """
    The reduced model's LFC that best reproduces the full model's term for the tested ratio.

    Shaped like lfc_of_tested_ratio, which is (num_features,) for the rna_kinetics models and
    (num_features, num_introns) for the coverage models.
    """
    is_one_dimensional = lfc_of_tested_ratio.dim() == 1
    target = design_matrix @ (lfc_of_tested_ratio.unsqueeze(1) if is_one_dimensional
                              else lfc_of_tested_ratio)
    reduced_lfc = get_projection_onto_design(reduced_design_matrix, target)
    return reduced_lfc.squeeze(1) if is_one_dimensional else reduced_lfc


def absorb_read_class_shifts(model_to_adjust: nn.Module,
                             model_target: nn.Module,
                             design_matrix: torch.Tensor,
                             reduced_design_matrix: torch.Tensor) -> None:
    """
    Move `model_to_adjust`'s intercepts so its predictions match `model_target`'s as closely as its
    own design allows. Modifies `model_to_adjust` in place.

    Each read class's LFC term is identified only up to a constant, which the intercepts absorb
    exactly: the exon class sits under intercept_exon alone, the nascent class under
    intercept_unspliced + intercept_pi_logit, and the unspliced class under intercept_unspliced,
    so shifting the unspliced class also needs intercept_pi_logit to keep the nascent class put.
    Under the plug-in that last move is both forbidden and unnecessary: the two intron classes
    shift together by construction, so the correction is zero up to rounding, and writing it would
    drift the value the coverage fit supplied.

    The shift is measured rather than derived per tested ratio, which keeps one rule for all six
    and lets this serve both directions -- hot starting a reduced model from the full fit, and
    restarting the full fit from a reduced solution that beat it. Each model is asked for its own
    terms according to its own mode, so the caller need not say which is which. Taking the mean
    over samples is right because the projection's residual is orthogonal to the constant vector,
    so the mean is exactly the constant the projection dropped; when the two designs span the same
    space the residual vanishes and the adjusted model starts at the target's loss.
    """
    def get_terms(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return model.get_lfc_terms_per_read_class(
            design_matrix, reduced_design_matrix if model.lrt_specification is not None else None)

    with torch.no_grad():
        shift_exon, shift_nascent, shift_unspliced = (
            (adjusted - target).mean(dim=0)
            for adjusted, target in zip(get_terms(model_to_adjust), get_terms(model_target)))

        model_to_adjust.intercept_exon -= shift_exon
        model_to_adjust.intercept_unspliced -= shift_unspliced
        if not model_to_adjust.coverage_is_plugged_in:
            model_to_adjust.intercept_pi_logit += shift_unspliced - shift_nascent


def absorb_coverage_model_shift(model_to_adjust: IntronCoverageModel,
                                design_matrix_to_adjust: torch.Tensor,
                                model_target: IntronCoverageModel,
                                design_matrix_target: torch.Tensor) -> None:
    """
    Move `model_to_adjust`'s intercept so its predicted pi matches `model_target`'s as closely as
    its own design allows. Modifies `model_to_adjust` in place.

    The counterpart of absorb_read_class_shifts for the coverage-only models, which predict
    pi = sigmoid(intercept_pi_logit - X @ lfc_elong_over_splice) and so have a single LFC term
    under a single intercept. The term enters negatively, so the intercept takes the shift with the
    same sign. Each model is paired with its own design matrix, so this also serves both directions.
    """
    with torch.no_grad():
        shift = ((design_matrix_to_adjust @ model_to_adjust.lfc_elong_over_splice)
                 - (design_matrix_target @ model_target.lfc_elong_over_splice)).mean(dim=0)
        model_to_adjust.intercept_pi_logit += shift


def set_reduced_rna_kinetics_model_from_full(model_reduced: nn.Module,
                                             model_full: nn.Module,
                                             design_matrix: torch.Tensor,
                                             reduced_design_matrix: torch.Tensor,
                                             intron_index: int = 0) -> None:
    """
    Point `model_reduced` at the closest thing to `model_full`'s solution that its own design can
    express. Modifies `model_reduced` in place.

    Every parameter carries over except the one the test restricts, which is replaced by the
    reduced design's best fit to the tested ratio's term, and the constant that fit leaves over is
    absorbed by the intercepts. Serves RNAKineticsModel and GlobalRNAKineticsModel alike;
    `intron_index` picks the tested intron where their LFCs are intron-specific.

    The counterpart of set_full_rna_kinetics_model_from_reduced, which goes the other way when a
    reduced fit lands below the full one.
    """
    _copy_shared_parameters(model_reduced, model_full)
    if reduced_design_matrix.shape[1] > 0:
        with torch.no_grad():
            model_reduced.reduced_lfc.copy_(get_hot_start_reduced_lfc(
                design_matrix, reduced_design_matrix,
                get_lfc_of_tested_ratio(model_full.state_dict(),
                                        model_reduced.lrt_specification.tested_ratio,
                                        intron_index)))
    absorb_read_class_shifts(model_reduced, model_full, design_matrix, reduced_design_matrix)


def set_reduced_coverage_model_from_full(model_reduced: IntronCoverageModel,
                                         reduced_design_matrix: torch.Tensor,
                                         model_full: IntronCoverageModel,
                                         design_matrix: torch.Tensor) -> None:
    """
    The IntronCoverageModel counterpart of set_reduced_rna_kinetics_model_from_full.

    The intercept is copied rather than the whole state dict, because the two models' single LFC
    has a different number of features and would not load.
    """
    with torch.no_grad():
        model_reduced.intercept_pi_logit.copy_(model_full.intercept_pi_logit)
        if reduced_design_matrix.shape[1] > 0:
            model_reduced.lfc_elong_over_splice.copy_(get_hot_start_reduced_lfc(
                design_matrix, reduced_design_matrix, model_full.lfc_elong_over_splice.detach()))
    absorb_coverage_model_shift(model_reduced, reduced_design_matrix,
                                model_full, design_matrix)


def _copy_shared_parameters(model_destination: nn.Module, model_source: nn.Module) -> None:
    """Copy every parameter the two models have in common, skipping any they do not."""
    destination_state = model_destination.state_dict()
    for key, value in model_source.state_dict().items():
        if key in destination_state:
            destination_state[key] = value
    model_destination.load_state_dict(destination_state)


def set_full_rna_kinetics_model_from_reduced(model_full: nn.Module,
                                               model_reduced: nn.Module,
                                               design_matrix: torch.Tensor,
                                               reduced_design_matrix: torch.Tensor) -> None:
    """
    Point `model_full` at `model_reduced`'s solution, so that refitting resumes from there.
    Modifies `model_full` in place.

    The reverse of hot starting a reduced model, used when a reduced fit lands below the full fit:
    the full model nests the reduced one, so the reduced solution is reachable and the full fit had
    simply stopped somewhere worse.

    Needs no per-ratio knowledge. The reduced model's three read-class terms are the same linear
    combinations of the parameters as the full model's (see `get_lfc_terms_per_read_class`), so
    projecting each onto the full design recovers all three parameters, whichever ratio was
    restricted. The projection is exact because the reduced design is nested in [1, X], and
    whatever constant it drops is then absorbed by the intercepts.
    """
    _copy_shared_parameters(model_full, model_reduced)
    with torch.no_grad():
        exon_term, nascent_term, unspliced_term = model_reduced.get_lfc_terms_per_read_class(
            design_matrix, reduced_design_matrix)
        lfc_init_over_deg = get_projection_onto_design(design_matrix, exon_term)
        init_over_deg_per_intron = lfc_init_over_deg.unsqueeze(1)
        model_full.lfc_init_over_deg.copy_(lfc_init_over_deg)
        model_full.lfc_splice_over_deg.copy_(
            init_over_deg_per_intron - get_projection_onto_design(design_matrix, unspliced_term))
        if not model_full.coverage_is_plugged_in:
            model_full.lfc_elong_over_splice.copy_(
                get_projection_onto_design(design_matrix, unspliced_term - nascent_term))
    absorb_read_class_shifts(model_full, model_reduced, design_matrix, reduced_design_matrix)


def set_full_global_rna_kinetics_model_from_reduced(model_full: nn.Module,
                                                      model_reduced: nn.Module,
                                                      design_matrix: torch.Tensor,
                                                      reduced_design_matrix: torch.Tensor) -> None:
    """
    The GlobalRNAKineticsModel counterpart of set_full_rna_kinetics_model_from_reduced.

    lfc_init_over_deg is per gene here and no tested ratio involves initiation
    (GLOBAL_MODEL_TESTED_RATIOS), so it carries over unchanged and the exon class already matches.
    The other two are shared across introns, so each per-intron difference below is the same in
    every column and is averaged over them.
    """
    _copy_shared_parameters(model_full, model_reduced)
    with torch.no_grad():
        exon_term, nascent_term, unspliced_term = model_reduced.get_lfc_terms_per_read_class(
            design_matrix, reduced_design_matrix)
        init_over_deg_per_intron = exon_term[:, model_reduced.gene_idx]
        model_full.lfc_splice_over_deg.copy_(get_projection_onto_design(
            design_matrix, (init_over_deg_per_intron - unspliced_term).mean(dim=1)))
        if not model_full.coverage_is_plugged_in:
            model_full.lfc_elong_over_splice.copy_(get_projection_onto_design(
                design_matrix, (unspliced_term - nascent_term).mean(dim=1)))
    absorb_read_class_shifts(model_full, model_reduced, design_matrix, reduced_design_matrix)


def set_full_coverage_model_from_reduced(model_full: IntronCoverageModel,
                                           design_matrix: torch.Tensor,
                                           model_reduced: IntronCoverageModel,
                                           reduced_design_matrix: torch.Tensor) -> None:
    """
    The IntronCoverageModel counterpart of set_full_rna_kinetics_model_from_reduced. One LFC
    term, recovered by projecting the reduced model's onto the full design.
    """
    with torch.no_grad():
        model_full.intercept_pi_logit.copy_(model_reduced.intercept_pi_logit)
        model_full.lfc_elong_over_splice.copy_(get_projection_onto_design(
            design_matrix, reduced_design_matrix @ model_reduced.lfc_elong_over_splice))
    absorb_coverage_model_shift(model_full, design_matrix, model_reduced, reduced_design_matrix)


def get_frozen_basis_ratios(model: nn.Module) -> frozenset[TestedRatio]:
    """
    Which of `LFC_PRIOR_BASIS` the model holds fixed rather than estimates.

    A plugged-in value is registered as a buffer, so it is absent from `named_parameters()`, and a
    basis ratio counts as frozen when every parameter it is built from is. Under the current basis
    that is one parameter per ratio, so a ratio is either wholly frozen or wholly estimated. A
    basis built from both at once has no well defined prior over what is left, since the penalty
    is a quadratic form in the ratios rather than in the parameters, so it is rejected here rather
    than given one of the two readings.
    """
    estimated = {name for name, _ in model.named_parameters()}
    frozen = set()
    for basis_ratio in LFC_PRIOR_BASIS:
        attributes = [attribute for attribute, _ in RATIO_AS_LFC_COMBINATION[basis_ratio]]
        if all(attribute not in estimated for attribute in attributes):
            frozen.add(basis_ratio)
        elif any(attribute not in estimated for attribute in attributes):
            raise ValueError(
                f'{basis_ratio} mixes estimated and plugged-in parameters, which leaves no prior '
                f'over the remaining basis ratios.')
    return frozenset(frozen)


def get_prior_precision(regularization_coefficients_df: pd.DataFrame,
                        feature_names: list[str],
                        frozen_basis_ratios: frozenset[TestedRatio] = frozenset(),
                        device: str = 'cpu') -> torch.Tensor:
    """
    The inverse prior covariance of the estimated basis ratios, per design column.

    Reads the covariance `estimate_prior_variance.R` fitted jointly over the three basis ratios
    and inverts it, giving the matrix the penalty is a quadratic form in. A design column whose
    covariance is unusable -- incomplete, because some ratio was not estimable there, or not
    positive definite -- falls back to the diagonal built from that column's per-ratio `lambda`,
    which penalises whatever directions were usable and leaves the rest effectively free.

    `frozen_basis_ratios` are the ones a plug-in holds fixed. Their rows and columns are dropped
    before the inverse and come back as zeros, so the penalty is the **marginal** prior over the
    ratios still being estimated and `get_prior_penalty` needs no special case: the frozen value
    still enters its quadratic form, against zeros.

    The marginal rather than the prior conditioned on the frozen value, which inverting the whole
    covariance would give instead. Conditioning is right for a value that is known, and a
    plugged-in one is an estimate: against a plugged-in value carrying sampling variance `V1`, the
    cross term the conditional prior applies is attenuated by `Psi / (Psi + V1)`, where `Psi` is
    that ratio's prior variance. `V1` is routinely of the prior's own size or larger, the coverage
    shape of one gene's introns being a far thinner slice of the data than the whole dataset the
    prior was fitted over, so the unattenuated cross term moves the shrinkage target by multiples
    of what the data support, and it moves it most for the genes whose plugged-in value is least
    certain. Dropping the term is the other end of the same formula, at `V1` large, and it keeps
    the target at zero, which is what `estimate_prior_variance.R` fits a prior covariance of
    `Psi + mu mu'` for.

    Dropping the rows before the finiteness check is also what lets a design column where the
    frozen ratio was not estimable keep a real prior over the rest, rather than falling back to
    the diagonal for a column it no longer reads.
    """
    coefficients = regularization_coefficients_df.set_index(
        ['rate_ratio', 'feature_name']).sort_index()
    precision = np.zeros((len(feature_names), len(LFC_PRIOR_BASIS), len(LFC_PRIOR_BASIS)))
    estimated_indices = [index for index, ratio in enumerate(LFC_PRIOR_BASIS)
                         if ratio not in frozen_basis_ratios]
    block = np.ix_(estimated_indices, estimated_indices)
    covariance_columns = [f'prior_cov_{LFC_PRIOR_BASIS[index].removeprefix("lfc_")}'
                          for index in estimated_indices]

    for feature_index, feature_name in enumerate(feature_names):
        rows = [coefficients.loc[(LFC_PRIOR_BASIS[index], feature_name)]
                for index in estimated_indices]
        covariance = torch.from_numpy(np.array(
            [[row[column] for column in covariance_columns] for row in rows], dtype=np.float64))
        # The same two tests the Fisher information goes through, and definite rather than
        # semidefinite because this matrix is inverted whole, where a flat direction leaves it
        # singular. The condition is reachable: `estimate_prior_variance.R` flags it as a
        # degenerate prior but fills in the covariance columns before doing so, so the numbers
        # arrive finite. It is tested here rather than read off that status because under a
        # plug-in the matrix inverted is a submatrix, which can be definite where the whole is
        # not. A narrow but positive direction passes and is inverted as it stands, giving a
        # precision that constrains the refit hard along it, which is the prior that was fitted.
        inverse = (try_invert(covariance)
                   if torch.isfinite(covariance).all() and is_positive_definite(covariance)
                   else None)
        if inverse is not None:
            precision[feature_index][block] = inverse.numpy()
        else:
            # A ratio whose fitted prior had no width at all arrives with an infinite `lambda`,
            # which would make the penalty and so the refit's loss non-finite. R reports that as
            # a degenerate prior rather than as a finding, so it is left unpenalised here, the
            # same answer as for a ratio there was nothing to learn a prior from.
            lambdas = np.array([row['lambda'] for row in rows], dtype=np.float64)
            precision[feature_index][estimated_indices, estimated_indices] = np.where(
                np.isfinite(lambdas), 2 * lambdas, 0.0)
    return torch.from_numpy(precision).to(dtype=MODEL_DTYPE, device=device)


def get_prior_penalty(model: RNAKineticsModel, prior_precision: torch.Tensor) -> torch.Tensor:
    """
    The L2 penalty, as the negative log density of the prior `get_prior_precision` describes.

    A Gaussian prior of covariance `Sigma` contributes `0.5 * t(theta) %*% inv(Sigma) %*% theta`,
    which for a one-dimensional prior reduces to `lambda * lfc^2` with
    `lambda = 1 / (2 * sigma^2)`. Here `theta` is one design column's three basis ratios at one
    intron, so the penalty sums over design columns and introns.

    The basis ratios are read out of the model's parameters through `RATIO_AS_LFC_COMBINATION`
    rather than named here, so the penalty is the same quadratic form in the same three linear
    functionals however the model happens to be parametrized.

    The initiation LFC has no intron axis, so it enters once per intron of the gene. That matches
    how the prior was fitted, where a (gene, intron) pair is the unit and the gene's initiation
    estimate is broadcast onto each of its introns.

    `IntronCoverageModel` needs none of this: its ratio space is one-dimensional, so its prior has
    no covariance to invert and the penalty stays the plain `lambda * lfc^2` term in
    `_make_intron_coverage_closures`.
    """
    # The width of the intron-axis LFCs' second axis: the introns when those LFCs are
    # intron-specific, and a broadcasting singleton when they are shared across the gene. It has to
    # come from the parameter rather than from `len(model.intron_names)`, which would be wrong in
    # the shared case and would try to expand initiation against a width-one axis.
    num_lfc_introns = model.lfc_splice_over_deg.shape[1]

    def get_basis_ratio_lfc(basis_ratio: TestedRatio) -> torch.Tensor:
        """One basis ratio's LFC, as (num_features, num_lfc_introns)."""
        terms = []
        for attribute, coefficient in RATIO_AS_LFC_COMBINATION[basis_ratio]:
            lfc = getattr(model, attribute)
            if lfc.dim() == 1:  # initiation, which the prior was fitted per intron as well
                lfc = lfc.unsqueeze(1).expand(-1, num_lfc_introns)
            terms.append(coefficient * lfc)
        return torch.stack(terms).sum(dim=0)

    # (num_features, num_lfc_introns, 3), the three basis ratios per design column and intron
    lfc_per_basis_ratio = torch.stack(
        [get_basis_ratio_lfc(basis_ratio) for basis_ratio in LFC_PRIOR_BASIS], dim=2)
    # One batched matmul per design column, then the elementwise product summed, which is
    # sum over design columns and introns of t(lfc) %*% prior_precision %*% lfc.
    precision_times_lfc = lfc_per_basis_ratio @ prior_precision
    return 0.5 * torch.sum(precision_times_lfc * lfc_per_basis_ratio)


def _make_rna_kinetics_closures(
        model: RNAKineticsModel,
        optimizer: optim.Optimizer,
        gene_data: GeneData,
        dataset_metadata: DatasetMetadata,
        reduced_matrix: Optional[torch.Tensor] = None,
        prior_precision: Optional[torch.Tensor] = None,
) -> tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]]:
    rna_kinetics_loss = RNAKineticsLoss(
        num_position_coverage=gene_data.coverage.shape[2]
    ).to(next(model.parameters()).device)

    def _compute_loss() -> torch.Tensor:
        predicted_reads_exon, predicted_reads_intron, predicted_pi = model(
            design_matrix=dataset_metadata.design_matrix,
            log_library_sizes=dataset_metadata.log_library_sizes,
            isoform_length_offset=gene_data.isoform_length_offset,
            reduced_design_matrix=reduced_matrix,
        )
        loss = rna_kinetics_loss(
            reads_exon=gene_data.exon_reads,
            reads_intron=gene_data.intron_reads,
            intron_coverage=gene_data.coverage,
            predicted_reads_exon=predicted_reads_exon,
            predicted_reads_intron=predicted_reads_intron,
            predicted_pi=predicted_pi,
            exon_count_dispersion=gene_data.exon_count_dispersion,
            intron_count_dispersion=gene_data.intron_count_dispersion,
        )
        if prior_precision is not None:
            loss += get_prior_penalty(model, prior_precision)
        return loss

    return make_closures(optimizer, _compute_loss)


def _make_intron_coverage_closures(
        model: IntronCoverageModel,
        optimizer: optim.Optimizer,
        coverage: torch.Tensor,
        design_matrix: torch.Tensor,
        regularization_coefficients_elong_over_splice: Optional[torch.Tensor] = None,
) -> tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]]:
    coverage_loss_fn = CoverageLoss(num_position_coverage=coverage.shape[2]).to(coverage.device)

    def _compute_loss() -> torch.Tensor:
        loss = coverage_loss_fn(model(design_matrix), coverage)
        if regularization_coefficients_elong_over_splice is not None:
            loss += torch.sum(regularization_coefficients_elong_over_splice * torch.square(model.lfc_elong_over_splice))
        return loss

    return make_closures(optimizer, _compute_loss)


# A reduced model landing below the full model means the full fit stopped somewhere worse, since
# the full model nests the reduced one. Restarting the full fit from that solution fixes it, and
# the reduced models are then refitted from the improved full fit, which can find better optima of
# their own and so trigger another round. Each round lowers the full loss by at least the
# tolerance, so this terminates; the cap is a backstop.
MAX_FULL_MODEL_RESTARTS = 10

# Only restart when a reduced model beats the full one by more than this. Guards the degenerate
# case where the reduced design spans the same space as the full one, which makes the comparison
# pure floating-point noise; on real contrasts the gap is orders of magnitude larger than this.
RESTART_LOSS_TOLERANCE = 0.1


@dataclass
class FullAndReducedFits:
    """
    A full model fitted together with the reduced models of its likelihood ratio tests.

    `results_reduced` is aligned with the reduced models the caller enumerated, and each entry is
    the best attempt for that test rather than the last, so its training flags describe the
    attempt whose loss is reported.
    """
    model_full: nn.Module
    results_full: TrainingResults
    results_reduced: list[TrainingResults]
    num_full_model_restarts: int


def fit_full_and_reduced_models(
        fit_full_from_scratch: Callable[[], tuple[nn.Module, TrainingResults]],
        fit_reduced: Callable[[nn.Module, int], tuple[nn.Module, TrainingResults]],
        refit_full_from_reduced: Callable[[nn.Module, int], tuple[nn.Module, TrainingResults]],
        num_reduced_models: int,
        max_restarts: int = MAX_FULL_MODEL_RESTARTS,
        loss_tolerance: float = RESTART_LOSS_TOLERANCE,
) -> FullAndReducedFits:
    """
    Fit the full model and every reduced model, restarting the full fit whenever a reduced one
    beats it.

    The caller supplies three closures and the number of reduced models. `lrt_index` addresses
    the caller's own list of enumerated tests, which must stay in the same order throughout:
    `fit_reduced(model_full, lrt_index)` hot starts that test's reduced model from the current
    full fit, and `refit_full_from_reduced(model_reduced, lrt_index)` points a fresh full model
    at that test's reduced solution and refits it.

    Everything the caller derives from the full fit -- parameter values, standard errors,
    chi-squared statistics, the reported effect sizes and the state dict cached for the
    regularized refit -- has to be computed from the result of this function rather than from an
    earlier fit, because the full model can change here.
    """
    model_full, results_full = fit_full_from_scratch()
    best_results_reduced: list[Optional[TrainingResults]] = [None] * num_reduced_models
    best_models_reduced: list[Optional[nn.Module]] = [None] * num_reduced_models
    num_full_model_restarts = 0

    while True:
        for lrt_index in range(num_reduced_models):
            model_reduced, results_reduced = fit_reduced(model_full, lrt_index)
            previous_best = best_results_reduced[lrt_index]
            if previous_best is None or results_reduced.final_loss < previous_best.final_loss:
                best_results_reduced[lrt_index] = results_reduced
                best_models_reduced[lrt_index] = model_reduced

        outperforming_indices = [
            lrt_index for lrt_index in range(num_reduced_models)
            if best_results_reduced[lrt_index].final_loss < results_full.final_loss - loss_tolerance]
        if not outperforming_indices or num_full_model_restarts >= max_restarts:
            break

        best_index = min(outperforming_indices,
                         key=lambda lrt_index: best_results_reduced[lrt_index].final_loss)
        candidate_model, candidate_results = refit_full_from_reduced(
            best_models_reduced[best_index], best_index)
        num_full_model_restarts += 1
        if candidate_results.final_loss >= results_full.final_loss:
            # Unreachable while the reverse hot start is exact: the refit starts at the reduced
            # model's loss, which is below the full model's by more than the tolerance, and
            # train_model never returns worse than its starting point. Kept because without it a
            # non-improving restart would repeat identical rounds until the cap.
            break
        model_full, results_full = candidate_model, candidate_results

    return FullAndReducedFits(model_full=model_full,
                              results_full=results_full,
                              results_reduced=best_results_reduced,
                              num_full_model_restarts=num_full_model_restarts)


def _iter_reported_ratios(intron_names: list[str],
                          intron_specific_lfc: bool,
                          tested_ratios: Sequence[TestedRatio] = tuple(TestedRatio)
                          ) -> Iterator[tuple[TestedRatio, Optional[str], int]]:
    """
    Every (ratio, intron) combination to report, with the intron's index.

    Every ratio except init/deg involves elongation or splicing, so it is reported one intron at a
    time whenever those LFCs are intron-specific. Shared by the LRT enumeration and the
    `ratio_lfcs` table, which pass the same `tested_ratios` wherever they cover the same set.

    They do not under the plug-in: `ratio_lfcs` reports an effect size for all six ratios,
    elong/splice included, since the plugged-in value is a perfectly good estimate of it, while
    the LRT enumeration passes PLUG_IN_TESTABLE_RATIOS because that one ratio has no reduced model
    here. Its test comes from the coverage fit instead.
    """
    for tested_ratio in tested_ratios:
        if intron_specific_lfc and tested_ratio != TestedRatio.INIT_OVER_DEG:
            yield from ((tested_ratio, intron_name, intron_index)
                        for intron_index, intron_name in enumerate(intron_names))
        else:
            yield tested_ratio, None, 0


def _get_tested_intron_index(gene_data: GeneData, lrt_specification: LRTSpecification) -> int:
    """Position of the tested intron, or 0 when the tested ratio is shared across the gene."""
    if lrt_specification.tested_intron is None:
        return 0
    return gene_data.intron_names.index(lrt_specification.tested_intron)


def get_rna_kinetics_model_results(gene_data: GeneData,
                                   dataset_metadata: DatasetMetadata,
                                   intron_specific_lfc: bool,
                                   plug_in_coverage_fits: Optional[list[PlugInCoverageFit]] = None,
                                   run_lrt: bool = True,
                                   device='cpu'
                                   ) -> ModelFitResults:
    """
    Fit the per-gene model and run every likelihood ratio test it supports.

    Given `plug_in_coverage_fits`, lfc_elong_over_splice and the pi intercepts are taken from
    those coverage fits rather than estimated here, and the model reduces to the count arm alone.
    One fit per intron is needed where the LFCs are intron-specific and a single fit covering the
    gene otherwise; `PlugInCoverage.from_fits` is where that is checked. Five of the six ratios
    are then tested as usual and elong/splice is not, its test being the coverage fit's own.

    The standard errors account for the uncertainty in what was plugged in, through
    `get_two_stage_parameter_covariance`, so they are comparable with those of a joint fit. The
    chi-squared statistics do not: each reduced fit treats the plugged-in values as known, so the
    likelihood ratio tests are anti-conservative where the Wald columns beside them are not.

    With `run_lrt` False no reduced model is fitted. The test results keep one row per test, with
    the Wald columns filled and the likelihood ratio test columns null.
    """
    gene_data = gene_data.to(device)
    dataset_metadata = dataset_metadata.to(device)

    plug_in_coverage: Optional[PlugInCoverage] = None
    if plug_in_coverage_fits is not None:
        plug_in_coverage = PlugInCoverage.from_fits(plug_in_coverage_fits,
                                                    gene_data.intron_names,
                                                    intron_specific_lfc).to(device)
    tested_ratios = (PLUG_IN_TESTABLE_RATIOS if plug_in_coverage is not None
                     else tuple(TestedRatio))

    # Enumerated up front so the reduced models can be indexed by position by the restart loop.
    lrt_inputs: list[LRTInputs] = []
    for _, lrt_metadata_row in dataset_metadata.lrt_metadata.iterrows():
        reduced_design_matrix = dataset_metadata.reduced_matrices[lrt_metadata_row['test_id']]
        for tested_ratio, intron_name, _ in _iter_reported_ratios(gene_data.intron_names,
                                                                  intron_specific_lfc,
                                                                  tested_ratios):
            lrt_inputs.append(LRTInputs(
                lrt_metadata_row=lrt_metadata_row,
                reduced_design_matrix=reduced_design_matrix,
                lrt_specification=LRTSpecification(
                    num_features_reduced_matrix=reduced_design_matrix.shape[1],
                    tested_ratio=tested_ratio,
                    tested_intron=intron_name)))

    def build_full_model() -> RNAKineticsModel:
        return RNAKineticsModel(feature_names=dataset_metadata.feature_names,
                                intron_names=gene_data.intron_names,
                                intron_specific_lfc=intron_specific_lfc,
                                plug_in_coverage=plug_in_coverage).to(device)

    def fit(model: RNAKineticsModel,
            reduced_design_matrix: Optional[torch.Tensor]) -> tuple[RNAKineticsModel,
                                                                    TrainingResults]:
        optimizer = make_lbfgs_optimizer(model)
        closure, evaluate_loss = _make_rna_kinetics_closures(
            model, optimizer, gene_data, dataset_metadata, reduced_matrix=reduced_design_matrix)
        return train_model(model, optimizer, closure, evaluate_loss)

    def fit_full_from_scratch() -> tuple[RNAKineticsModel, TrainingResults]:
        model = build_full_model()
        model.initialize_parameters(gene_data, dataset_metadata.library_sizes,
                                   dataset_metadata.design_matrix)
        return fit(model, None)

    def fit_reduced(model_full: RNAKineticsModel,
                    lrt_index: int) -> tuple[RNAKineticsModel, TrainingResults]:
        lrt_input = lrt_inputs[lrt_index]
        model_reduced = RNAKineticsModel(
            feature_names=dataset_metadata.feature_names,
            intron_names=gene_data.intron_names,
            intron_specific_lfc=intron_specific_lfc,
            lrt_specification=lrt_input.lrt_specification,
            plug_in_coverage=plug_in_coverage).to(device)

        set_reduced_rna_kinetics_model_from_full(
            model_reduced, model_full, dataset_metadata.design_matrix,
            lrt_input.reduced_design_matrix,
            _get_tested_intron_index(gene_data, lrt_input.lrt_specification))
        return fit(model_reduced, lrt_input.reduced_design_matrix)

    def refit_full_from_reduced(model_reduced: RNAKineticsModel,
                                lrt_index: int) -> tuple[RNAKineticsModel, TrainingResults]:
        model = build_full_model()
        set_full_rna_kinetics_model_from_reduced(
            model, model_reduced, dataset_metadata.design_matrix,
            lrt_inputs[lrt_index].reduced_design_matrix)
        return fit(model, None)

    fits = fit_full_and_reduced_models(fit_full_from_scratch=fit_full_from_scratch,
                                       fit_reduced=fit_reduced,
                                       refit_full_from_reduced=refit_full_from_reduced,
                                       num_reduced_models=len(lrt_inputs) if run_lrt else 0)
    model_full, results_full = fits.model_full, fits.results_full

    # Everything below is derived from the full fit, so it has to come after the restart loop:
    # a restart replaces the full model, and the parameter values, standard errors, chi-squared
    # statistics, effect sizes and cached state dict all describe whichever fit was kept.
    model_param_df = model_full.get_param_df()
    model_param_df['gene_name'] = gene_data.gene_name
    model_param_df['loss_full_model'] = results_full.final_loss
    model_param_df['training_diverged_full_model'] = results_full.training_diverged
    model_param_df['training_converged_full_model'] = results_full.converged
    model_param_df['num_epochs_full_model'] = results_full.num_epochs

    # Wald test
    rna_kinetics_loss = RNAKineticsLoss(num_position_coverage=gene_data.coverage.shape[2]).to(device)

    def rna_kinetics_loss_by_params(params):
        predicted_reads_exon, predicted_reads_intron, predicted_pi = functional_call(
            model_full, params,
            (dataset_metadata.design_matrix, dataset_metadata.log_library_sizes, gene_data.isoform_length_offset),
        )
        return rna_kinetics_loss(reads_exon=gene_data.exon_reads,
                                reads_intron=gene_data.intron_reads,
                                intron_coverage=gene_data.coverage,
                                predicted_reads_exon=predicted_reads_exon,
                                predicted_reads_intron=predicted_reads_intron,
                                predicted_pi=predicted_pi,
                                exon_count_dispersion=gene_data.exon_count_dispersion,
                                intron_count_dispersion=gene_data.intron_count_dispersion)

    fisher_information_matrix = get_fisher_information_matrix(model_full, rna_kinetics_loss_by_params)
    if plug_in_coverage is None:
        parameter_covariance = get_parameter_covariance(fisher_information_matrix)
        free_information_matrix = fisher_information_matrix
    else:
        # The Hessian spans the plugged-in values too, and only its free block is an information
        # matrix, so that is the block the reported diagnostics describe. A stage one that handed
        # over no information leaves nothing to correct with, so there are no Wald results at all,
        # and then `loss_hessian_is_positive_definite` is False although the block itself was
        # fine; the coverage fit's own `CoverageFitStatus` is what says which of the two happened.
        num_free_parameters = sum(parameter.numel() for parameter in model_full.parameters())
        free_information_matrix = fisher_information_matrix[
            :num_free_parameters, :num_free_parameters]
        parameter_covariance = None
        if plug_in_coverage.fisher_information is not None:
            parameter_covariance = get_two_stage_parameter_covariance(
                fisher_information_matrix, num_free_parameters,
                plug_in_coverage.fisher_information)
    model_param_df = add_wald_test_results(model_param_df, parameter_covariance,
                                           free_information_matrix)

    ratio_lfc_df = get_ratio_lfc_df(
        model_full, parameter_covariance,
        [RatioToReport(ratio=tested_ratio,
                       lfc_combination=RATIO_AS_LFC_COMBINATION[tested_ratio],
                       intron_name=intron_name,
                       intron_index=intron_index)
         for tested_ratio, intron_name, intron_index
         in _iter_reported_ratios(gene_data.intron_names, intron_specific_lfc)])
    ratio_lfc_df['gene_name'] = gene_data.gene_name
    ratio_lfc_df['training_diverged_full_model'] = results_full.training_diverged

    state_dict_model_full = model_full.state_dict()
    feature_names = dataset_metadata.feature_names
    test_results_list: list[dict] = []
    for lrt_index, lrt_input in enumerate(lrt_inputs):
        lrt_specification = lrt_input.lrt_specification
        results_reduced = fits.results_reduced[lrt_index] if run_lrt else None
        tested_intron_index = _get_tested_intron_index(gene_data, lrt_specification)
        # Read off the tested ratio rather than the parameter table: three of the six ratios have
        # no row of their own there.
        lfc_of_tested_ratio = get_lfc_of_tested_ratio(
            state_dict_model_full, lrt_specification.tested_ratio, tested_intron_index)

        test_result = lrt_input.lrt_metadata_row.to_dict()
        test_result['rate_ratio'] = lrt_specification.tested_ratio
        test_result['gene_name'] = gene_data.gene_name
        test_result['intron_name'] = lrt_specification.tested_intron
        test_result['lfc'] = (
                get_lfc_of_column(lfc_of_tested_ratio, feature_names,
                                  test_result['lfc_column_positive'])
                - get_lfc_of_column(lfc_of_tested_ratio, feature_names,
                                    test_result['lfc_column_negative']))
        test_result.update(get_contrast_wald_results(
            parameter_covariance,
            get_contrast_of_ratio(model_full,
                                  RATIO_AS_LFC_COMBINATION[lrt_specification.tested_ratio],
                                  test_result['lfc_column_positive'],
                                  test_result['lfc_column_negative'],
                                  tested_intron_index),
            test_result['lfc']))
        test_result['loss_full_model'] = results_full.final_loss
        test_result.update(get_lrt_results(results_full, results_reduced, test_result['lrt_df']))
        test_result['num_full_model_restarts'] = fits.num_full_model_restarts
        test_results_list.append(test_result)

    return ModelFitResults(model_parameters=model_param_df,
                           ratio_lfcs=ratio_lfc_df,
                           test_results=pd.DataFrame(test_results_list),
                           state_dict_full_model=state_dict_model_full)


def get_regularized_rna_kinetics_model_results(gene_data: GeneData,
                                               dataset_metadata: DatasetMetadata,
                                               intron_specific_lfc: bool,
                                               hot_start_state_dict: StateDict,
                                               regularization_coefficients_df: pd.DataFrame,
                                               plug_in_coverage_fits: Optional[
                                                   list[PlugInCoverageFit]] = None,
                                               device='cpu'
                                               ) -> RegularizedFitResults:
    """
    Refit the per-gene model under the prior the first fit was used to estimate.

    `plug_in_coverage_fits` are the coverage model's **regularized** fits, where the unregularized
    fit was handed its MLEs. Freezing the MLE here would leave `lfc_elong_over_deg` and
    `lfc_init_over_elong` reported in the regularized column while one of the parts they are built
    from had never been shrunk, and that part is the one the coverage shape determines least well:
    where it is flat the MLE is at the edge of pi's range, so the frozen value is arbitrarily
    large and the ratios containing it inherit that. Both values come from the refit as a pair,
    since the intercepts re-optimize against the penalized LFC and the count arm reads the two of
    them only through the nascent fraction they imply together.

    The count arm's own `lfc_splice_over_deg` responds, because the intron counts pin down the
    total while the plugged-in values set how it splits between the nascent and unspliced classes.
    Its sensitivity is `d lfc_splice_over_deg / d lfc_elong_over_splice = -pi`, so a fraction pi
    of whatever the shrinkage moved is absorbed there, and `lfc_elong_over_deg` carries the rest.
    That is bounded by the nascent fraction rather than by how far the frozen value moved, which
    is what keeps it small where the shrinkage is largest.

    Passing none where the first fit had them would silently turn the plugged-in values back into
    free parameters: a state dict names them identically either way, so `load_state_dict` accepts
    it and the refit then estimates against the count arm what stage one read off the coverage
    shape. For the same reason the hot start, taken from the fit that froze the MLEs, overwrites
    these two buffers and `set_plug_in_coverage` has to put them back afterwards.

    The prior a plug-in is refitted under is the marginal one over the ratios still being
    estimated, which `get_prior_precision` builds from `get_frozen_basis_ratios` here; the reason
    is given there. `get_prior_penalty` itself needs no special case either way.
    """
    gene_data = gene_data.to(device)
    dataset_metadata = dataset_metadata.to(device)

    plug_in_coverage: Optional[PlugInCoverage] = None
    if plug_in_coverage_fits is not None:
        plug_in_coverage = PlugInCoverage.from_fits(plug_in_coverage_fits,
                                                    gene_data.intron_names,
                                                    intron_specific_lfc).to(device)
    model_regularized = RNAKineticsModel(feature_names=dataset_metadata.feature_names,
                                         intron_names=gene_data.intron_names,
                                         intron_specific_lfc=intron_specific_lfc,
                                         plug_in_coverage=plug_in_coverage).to(device)
    model_regularized.load_state_dict(hot_start_state_dict)
    if plug_in_coverage is not None:
        model_regularized.set_plug_in_coverage(plug_in_coverage)

    prior_precision = get_prior_precision(regularization_coefficients_df,
                                          model_regularized.feature_names,
                                          get_frozen_basis_ratios(model_regularized),
                                          device)

    optimizer_regularized = make_lbfgs_optimizer(model_regularized)
    closure_regularized, evaluate_loss_regularized = _make_rna_kinetics_closures(
        model_regularized, optimizer_regularized, gene_data, dataset_metadata,
        prior_precision=prior_precision,
    )
    model_regularized, training_results_regularized = train_model(
        model_regularized, optimizer_regularized, closure_regularized, evaluate_loss_regularized,
    )

    model_param_df = model_regularized.get_param_df()
    model_param_df['gene_name'] = gene_data.gene_name
    model_param_df['loss_regularized_model'] = training_results_regularized.final_loss
    model_param_df['training_diverged_regularized_model'] = training_results_regularized.training_diverged
    model_param_df['training_converged_regularized_model'] = training_results_regularized.converged

    ratio_lfc_df = get_ratio_lfc_df(
        model_regularized, parameter_covariance=None, with_standard_errors=False,
        ratios_to_report=[RatioToReport(ratio=tested_ratio,
                                        lfc_combination=RATIO_AS_LFC_COMBINATION[tested_ratio],
                                        intron_name=intron_name,
                                        intron_index=intron_index)
                          for tested_ratio, intron_name, intron_index
                          in _iter_reported_ratios(gene_data.intron_names, intron_specific_lfc)])
    ratio_lfc_df['gene_name'] = gene_data.gene_name
    return RegularizedFitResults(model_parameters=model_param_df, ratio_lfcs=ratio_lfc_df)


def get_intron_coverage_model_results(
        coverage: torch.Tensor,
        dataset_metadata: DatasetMetadata,
        intron_names: list[str],
        is_global_model: bool = False,
        intron_specific_lfc: bool = False,
        device: str = 'cpu',
        verbose: bool = False,
) -> ModelFitResults:
    """
    Fit IntronCoverageModel to the given introns and run its likelihood ratio tests.

    `is_global_model` says the single LFC is shared across the whole dataset's introns rather than
    one gene's, which changes two things. Training goes through `train_global_model`, because
    there are then tens of thousands of per-intron intercepts against that one LFC and optimising
    the LFC alone first is far better conditioned. And no Wald test is computed, because the
    Fisher information would be one row and column per intercept, making it very large.

    `intron_specific_lfc` says this call is one intron's fit, which is what decides whether the
    LFC is reported under that intron's name. It is declared rather than read off the length of
    `intron_names`, so that a gene with one intron is still a gene-wide fit.
    """
    coverage = coverage.to(device)
    dataset_metadata = dataset_metadata.to(device)

    # The intron the LFC is reported under; None wherever it is shared, which is every fit but a
    # per-intron one, the global model included.
    lfc_intron_name = intron_names[0] if intron_specific_lfc else None

    # No lrt_specification: this model's single LFC is the only thing it can restrict.
    lrt_inputs = [LRTInputs(lrt_metadata_row=row,
                            reduced_design_matrix=dataset_metadata.reduced_matrices[row['test_id']])
                  for _, row in dataset_metadata.lrt_metadata.iterrows()]

    def fit(model: IntronCoverageModel,
            design_matrix: torch.Tensor,
            label: str) -> tuple[IntronCoverageModel, TrainingResults]:
        if is_global_model:
            return train_global_model(
                model,
                global_param_names={LFCParameter.ELONG_OVER_SPLICE},
                make_closures=lambda opt: _make_intron_coverage_closures(
                    model, opt, coverage, design_matrix),
                label=label,
                verbose=verbose,
            )
        optimizer = make_lbfgs_optimizer(model)
        closure, evaluate_loss = _make_intron_coverage_closures(
            model, optimizer, coverage, design_matrix)
        return train_model(model, optimizer, closure, evaluate_loss, verbose=verbose)

    def build_full_model() -> IntronCoverageModel:
        return IntronCoverageModel(feature_names=dataset_metadata.feature_names,
                                   intron_names=intron_names,
                                   intron_specific_lfc=intron_specific_lfc).to(device)

    def fit_full_from_scratch() -> tuple[IntronCoverageModel, TrainingResults]:
        model = build_full_model()
        model.initialize_parameters(coverage)
        return fit(model, dataset_metadata.design_matrix, 'Intron coverage | full model')

    def fit_reduced(model_full: IntronCoverageModel,
                    lrt_index: int) -> tuple[IntronCoverageModel, TrainingResults]:
        reduced_matrix = lrt_inputs[lrt_index].reduced_design_matrix
        model_reduced = IntronCoverageModel(
            feature_names=[f'reduced_feature_{i}' for i in range(reduced_matrix.shape[1])],
            intron_names=intron_names,
            intron_specific_lfc=intron_specific_lfc,
        ).to(device)
        set_reduced_coverage_model_from_full(model_reduced, reduced_matrix,
                                            model_full, dataset_metadata.design_matrix)
        return fit(model_reduced, reduced_matrix,
                   f'Intron coverage | LRT {lrt_inputs[lrt_index].lrt_metadata_row["test_id"]}')

    def refit_full_from_reduced(model_reduced: IntronCoverageModel,
                                lrt_index: int) -> tuple[IntronCoverageModel, TrainingResults]:
        model = build_full_model()
        set_full_coverage_model_from_reduced(model, dataset_metadata.design_matrix,
                                             model_reduced,
                                             lrt_inputs[lrt_index].reduced_design_matrix)
        return fit(model, dataset_metadata.design_matrix,
                   'Intron coverage | full model restart')

    fits = fit_full_and_reduced_models(fit_full_from_scratch=fit_full_from_scratch,
                                       fit_reduced=fit_reduced,
                                       refit_full_from_reduced=refit_full_from_reduced,
                                       num_reduced_models=len(lrt_inputs))
    model_full, results_full = fits.model_full, fits.results_full

    # Derived from the full fit, so after the restart loop: a restart replaces the full model.
    model_param_df = model_full.get_param_df()
    model_param_df['loss_full_model'] = results_full.final_loss
    model_param_df['training_diverged_full_model'] = results_full.training_diverged
    model_param_df['training_converged_full_model'] = results_full.converged
    model_param_df['num_epochs_full_model'] = results_full.num_epochs

    parameter_covariance: Optional[ParameterCovariance] = None
    fisher_information_matrix: Optional[torch.Tensor] = None
    if not is_global_model:
        coverage_loss_fn = CoverageLoss(num_position_coverage=coverage.shape[2]).to(device)

        def intron_coverage_loss_by_params(params):
            pi = functional_call(model_full, params, (dataset_metadata.design_matrix,))
            return coverage_loss_fn(pi, coverage)

        fisher_information_matrix = get_fisher_information_matrix(model_full,
                                                                  intron_coverage_loss_by_params)
        parameter_covariance = get_parameter_covariance(fisher_information_matrix)
        model_param_df = add_wald_test_results(model_param_df, parameter_covariance,
                                               fisher_information_matrix)

    # A one-dimensional ratio space: the model's single LFC is itself a rate ratio, so this table
    # has one row per design column and no standard errors for the global model.
    ratio_lfc_df = get_ratio_lfc_df(
        model_full, parameter_covariance,
        [RatioToReport(ratio=LFCParameter.ELONG_OVER_SPLICE,
                       lfc_combination=((LFCParameter.ELONG_OVER_SPLICE, 1.0),),
                       intron_name=lfc_intron_name)])
    ratio_lfc_df['training_diverged_full_model'] = results_full.training_diverged

    feature_names = dataset_metadata.feature_names
    lfc_elong_over_splice = model_full.lfc_elong_over_splice.detach()[:, 0]

    test_results_list: list[dict] = []
    for lrt_index, lrt_input in enumerate(lrt_inputs):
        lrt_row = lrt_input.lrt_metadata_row
        results_reduced = fits.results_reduced[lrt_index]

        test_result = lrt_row.to_dict()
        test_result['rate_ratio'] = LFCParameter.ELONG_OVER_SPLICE
        test_result['intron_name'] = lfc_intron_name
        test_result['lfc'] = (
                get_lfc_of_column(lfc_elong_over_splice, feature_names,
                                  lrt_row['lfc_column_positive'])
                - get_lfc_of_column(lfc_elong_over_splice, feature_names,
                                    lrt_row['lfc_column_negative']))
        if not is_global_model:
            # The global model has no Fisher information, so it reports no Wald columns at all
            # rather than a column of nulls.
            test_result.update(get_contrast_wald_results(
                parameter_covariance,
                get_contrast_of_ratio(model_full, ((LFCParameter.ELONG_OVER_SPLICE, 1.0),),
                                      lrt_row['lfc_column_positive'],
                                      lrt_row['lfc_column_negative']),
                test_result['lfc']))
        test_result['loss_full_model'] = results_full.final_loss
        test_result.update(get_lrt_results(results_full, results_reduced, lrt_row['lrt_df']))
        test_result['num_full_model_restarts'] = fits.num_full_model_restarts
        test_results_list.append(test_result)

    return ModelFitResults(
        model_parameters=model_param_df,
        ratio_lfcs=ratio_lfc_df,
        test_results=pd.DataFrame(test_results_list),
        state_dict_full_model=model_full.state_dict(),
        plug_in_coverage_fit=get_plug_in_coverage_fit(
            model_full, fisher_information_matrix,
            training_diverged=results_full.training_diverged,
            computes_fisher_information=not is_global_model))


def _make_global_rna_kinetics_closures(
        model: GlobalRNAKineticsModel,
        optimizer: optim.Optimizer,
        global_gene_data: GlobalGeneData,
        dataset_metadata: DatasetMetadata,
        reduced_matrix: Optional[torch.Tensor] = None,
) -> tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]]:
    rna_kinetics_loss = RNAKineticsLoss(
        num_position_coverage=global_gene_data.coverage.shape[2]
    ).to(next(model.parameters()).device)

    def _compute_loss() -> torch.Tensor:
        predicted_reads_exon, predicted_reads_intron, predicted_pi = model(
            design_matrix=dataset_metadata.design_matrix,
            log_library_sizes=dataset_metadata.log_library_sizes,
            isoform_length_offset=global_gene_data.isoform_length_offset,
            reduced_design_matrix=reduced_matrix,
        )
        return rna_kinetics_loss(
            reads_exon=global_gene_data.exon_reads,
            reads_intron=global_gene_data.intron_reads,
            intron_coverage=global_gene_data.coverage,
            predicted_reads_exon=predicted_reads_exon,
            predicted_reads_intron=predicted_reads_intron,
            predicted_pi=predicted_pi,
            exon_count_dispersion=global_gene_data.exon_count_dispersion,
            intron_count_dispersion=global_gene_data.intron_count_dispersion,
        )

    return make_closures(optimizer, _compute_loss)


def get_global_rna_kinetics_model_results(
        global_gene_data: GlobalGeneData,
        dataset_metadata: DatasetMetadata,
        plug_in_coverage_fits: Optional[list[PlugInCoverageFit]] = None,
        run_lrt: bool = True,
        device: str = 'cpu',
        verbose: bool = False,
) -> ModelFitResults:
    """
    Fit the genome-wide model and run every likelihood ratio test it supports.

    With `run_lrt` False no reduced model is fitted, and the test results keep one row per test
    with the likelihood ratio test columns null.

    Given `plug_in_coverage_fits`, lfc_elong_over_splice and the pi intercepts come from the
    coverage fits covering this dataset's introns rather than from here. The LFC is shared across
    the whole dataset, so a single fit over all of them is what this takes, and of the three
    ratios the model can test only two are left: elong/splice is the coverage fit's own test.

    `ratio_lfcs` contains all three ratios, including a plugged-in one, and no standard errors,
    since this model computes no Fisher information. Jackknife replicates report it, and the
    contrasts are computed from it afterwards.
    """
    global_gene_data = global_gene_data.to(device)
    dataset_metadata = dataset_metadata.to(device)
    feature_names = dataset_metadata.feature_names

    plug_in_coverage: Optional[PlugInCoverage] = None
    if plug_in_coverage_fits is not None:
        plug_in_coverage = PlugInCoverage.from_fits(plug_in_coverage_fits,
                                                    global_gene_data.intron_names,
                                                    intron_specific_lfc=False).to(device)
    tested_ratios = (GLOBAL_PLUG_IN_TESTABLE_RATIOS if plug_in_coverage is not None
                     else GLOBAL_MODEL_TESTED_RATIOS)

    lrt_inputs: list[LRTInputs] = []
    for _, lrt_row in dataset_metadata.lrt_metadata.iterrows():
        reduced_matrix = dataset_metadata.reduced_matrices[lrt_row['test_id']]
        for tested_ratio in tested_ratios:
            lrt_inputs.append(LRTInputs(
                lrt_metadata_row=lrt_row,
                reduced_design_matrix=reduced_matrix,
                lrt_specification=LRTSpecification(
                    num_features_reduced_matrix=reduced_matrix.shape[1],
                    tested_ratio=tested_ratio)))

    def build_model(lrt_specification: Optional[LRTSpecification]) -> GlobalRNAKineticsModel:
        return GlobalRNAKineticsModel(
            feature_names=feature_names,
            gene_names=global_gene_data.gene_names,
            intron_names=global_gene_data.intron_names,
            gene_idx=global_gene_data.gene_idx,
            lrt_specification=lrt_specification,
            plug_in_coverage=plug_in_coverage,
        ).to(device)

    def fit(model: GlobalRNAKineticsModel,
            reduced_matrix: Optional[torch.Tensor],
            label: str) -> tuple[GlobalRNAKineticsModel, TrainingResults]:
        # Named rather than selected, so a plugged-in lfc_elong_over_splice simply is not among
        # the model's parameters and the first stage optimizes what is left. The staging is then
        # redundant, the plug-in objective being convex, but it costs one extra warm start.
        global_param_names = {LFCParameter.ELONG_OVER_SPLICE, LFCParameter.SPLICE_OVER_DEG}
        if reduced_matrix is not None:
            global_param_names = global_param_names | {'reduced_lfc'}
        return train_global_model(
            model,
            global_param_names=global_param_names,
            make_closures=lambda opt: _make_global_rna_kinetics_closures(
                model, opt, global_gene_data, dataset_metadata, reduced_matrix),
            label=label,
            verbose=verbose,
        )

    def fit_full_from_scratch() -> tuple[GlobalRNAKineticsModel, TrainingResults]:
        model = build_model(None)
        model.initialize_parameters(global_gene_data, dataset_metadata.library_sizes,
                                   dataset_metadata.design_matrix)
        return fit(model, None, 'Global RNA kinetics | full model')

    def fit_reduced(model_full: GlobalRNAKineticsModel,
                    lrt_index: int) -> tuple[GlobalRNAKineticsModel, TrainingResults]:
        lrt_input = lrt_inputs[lrt_index]
        model_reduced = build_model(lrt_input.lrt_specification)

        set_reduced_rna_kinetics_model_from_full(
            model_reduced, model_full, dataset_metadata.design_matrix,
            lrt_input.reduced_design_matrix)
        return fit(model_reduced, lrt_input.reduced_design_matrix,
                   f'Global RNA kinetics | LRT {lrt_input.lrt_metadata_row["test_id"]} '
                   f'| {lrt_input.lrt_specification.tested_ratio}')

    def refit_full_from_reduced(model_reduced: GlobalRNAKineticsModel,
                                lrt_index: int) -> tuple[GlobalRNAKineticsModel, TrainingResults]:
        model = build_model(None)
        set_full_global_rna_kinetics_model_from_reduced(
            model, model_reduced, dataset_metadata.design_matrix,
            lrt_inputs[lrt_index].reduced_design_matrix)
        return fit(model, None, 'Global RNA kinetics | full model restart')

    fits = fit_full_and_reduced_models(fit_full_from_scratch=fit_full_from_scratch,
                                       fit_reduced=fit_reduced,
                                       refit_full_from_reduced=refit_full_from_reduced,
                                       num_reduced_models=len(lrt_inputs) if run_lrt else 0)
    model, training_results = fits.model_full, fits.results_full

    # Derived from the full fit, so after the restart loop: a restart replaces the full model.
    model_param_df = model.get_param_df()
    model_param_df['loss_full_model'] = training_results.final_loss
    model_param_df['training_diverged_full_model'] = training_results.training_diverged
    model_param_df['training_converged_full_model'] = training_results.converged
    model_param_df['num_epochs_full_model'] = training_results.num_epochs

    full_state = model.state_dict()
    test_results_list: list[dict] = []
    for lrt_index, lrt_input in enumerate(lrt_inputs):
        lrt_row, lrt_specification = lrt_input.lrt_metadata_row, lrt_input.lrt_specification
        results_reduced = fits.results_reduced[lrt_index] if run_lrt else None
        lfc_of_tested_ratio = get_lfc_of_tested_ratio(full_state, lrt_specification.tested_ratio)

        test_result = lrt_row.to_dict()
        test_result['rate_ratio'] = lrt_specification.tested_ratio
        test_result['lfc'] = (
                get_lfc_of_column(lfc_of_tested_ratio, feature_names,
                                  lrt_row['lfc_column_positive'])
                - get_lfc_of_column(lfc_of_tested_ratio, feature_names,
                                    lrt_row['lfc_column_negative']))
        test_result['loss_full_model'] = training_results.final_loss
        test_result.update(get_lrt_results(training_results, results_reduced, lrt_row['lrt_df']))
        test_result['num_full_model_restarts'] = fits.num_full_model_restarts
        test_results_list.append(test_result)

    ratio_lfc_df = get_ratio_lfc_df(
        model, parameter_covariance=None, with_standard_errors=False,
        ratios_to_report=[RatioToReport(ratio=tested_ratio,
                                        lfc_combination=RATIO_AS_LFC_COMBINATION[tested_ratio])
                          for tested_ratio in GLOBAL_MODEL_TESTED_RATIOS])
    ratio_lfc_df['training_diverged_full_model'] = training_results.training_diverged

    return ModelFitResults(model_parameters=model_param_df,
                           ratio_lfcs=ratio_lfc_df,
                           test_results=pd.DataFrame(test_results_list),
                           state_dict_full_model=full_state)


def get_regularized_intron_coverage_model_results(
        input_data: GeneData | IntronData,
        dataset_metadata: DatasetMetadata,
        hot_start_state_dict: StateDict,
        regularization_coefficients_df: pd.DataFrame,
        intron_specific_lfc: bool = False,
        device: str = 'cpu',
) -> RegularizedFitResults:
    coverage = input_data.coverage.to(device)
    dataset_metadata = dataset_metadata.to(device)
    regularization_coefficients_df = regularization_coefficients_df.set_index(['rate_ratio', 'feature_name'])

    model_regularized = IntronCoverageModel(
        feature_names=dataset_metadata.feature_names,
        intron_names=input_data.intron_names,
        intron_specific_lfc=intron_specific_lfc,
    ).to(device)
    model_regularized.load_state_dict(hot_start_state_dict)

    regularization_coefficients_elong_over_splice = torch.zeros((len(dataset_metadata.feature_names), 1)).to(device)
    for feature_index, feature_name in enumerate(dataset_metadata.feature_names):
        regularization_coefficients_elong_over_splice[feature_index] = regularization_coefficients_df.loc[
            (LFCParameter.ELONG_OVER_SPLICE, feature_name)]['lambda']

    optimizer_regularized = make_lbfgs_optimizer(model_regularized)
    closure_regularized, evaluate_loss_regularized = _make_intron_coverage_closures(
        model_regularized, optimizer_regularized, coverage, dataset_metadata.design_matrix,
        regularization_coefficients_elong_over_splice=regularization_coefficients_elong_over_splice,
    )
    model_regularized, training_results_regularized = train_model(
        model_regularized, optimizer_regularized, closure_regularized, evaluate_loss_regularized,
    )

    model_param_df = model_regularized.get_param_df()
    model_param_df['gene_name'] = input_data.gene_name
    model_param_df['loss_regularized_model'] = training_results_regularized.final_loss
    model_param_df['training_diverged_regularized_model'] = training_results_regularized.training_diverged
    model_param_df['training_converged_regularized_model'] = training_results_regularized.converged

    # The same rule `IntronCoverageModel.get_param_df` follows, and it has to be: this table and
    # `model_parameters.tsv` are joined on `intron_name` by `merge_regularization.R`.
    lfc_intron_name = input_data.intron_names[0] if intron_specific_lfc else None
    ratio_lfc_df = get_ratio_lfc_df(
        model_regularized, parameter_covariance=None, with_standard_errors=False,
        ratios_to_report=[RatioToReport(ratio=LFCParameter.ELONG_OVER_SPLICE,
                                        lfc_combination=((LFCParameter.ELONG_OVER_SPLICE, 1.0),),
                                        intron_name=lfc_intron_name)])
    ratio_lfc_df['gene_name'] = input_data.gene_name
    return RegularizedFitResults(
        model_parameters=model_param_df,
        ratio_lfcs=ratio_lfc_df,
        plug_in_coverage_fit=get_plug_in_coverage_fit(
            model_regularized, fisher_information_matrix=None,
            training_diverged=training_results_regularized.training_diverged,
            computes_fisher_information=False))
