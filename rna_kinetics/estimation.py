from dataclasses import dataclass
from typing import Callable, Iterator, NamedTuple, Optional, OrderedDict, Sequence

import numpy as np
import pandas as pd
import scipy
import torch
from scipy import stats
from torch import nn, optim
from torch.func import functional_call, hessian

from rna_kinetics.data import GeneData, IntronData, DatasetMetadata, GlobalGeneData
from rna_kinetics.models import RNAKineticsLoss, RNAKineticsModel, TestedRatio, LRTSpecification, \
    IntronCoverageModel, CoverageLoss, GlobalRNAKineticsModel, RATIO_AS_LFC_COMBINATION, \
    GLOBAL_MODEL_TESTED_RATIOS, GLOBAL_PLUG_IN_TESTABLE_RATIOS, PLUG_IN_TESTABLE_RATIOS, \
    LFCCombination, LFCParameter, CoverageFitStatus, PlugInCoverage, PlugInCoverageFit, \
    PlugInCoverageFits

LOSS_CLAMP_VALUE = 1e30

StateDict = OrderedDict[str, torch.Tensor]


@dataclass
class TrainingResults:
    final_loss: float
    converged_within_max_epochs: bool
    num_epochs: int
    losses: list[float]
    training_diverged: bool


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


def _get_lfc_without_intron_axis(lfc: torch.Tensor, intron_index: int) -> torch.Tensor:
    """One LFC parameter as a (num_features,) vector, dropping the intron axis where it has one."""
    return lfc if lfc.dim() == 1 else lfc[:, intron_index]


def get_lfc_of_lfc_combination(state_dict: StateDict,
                               lfc_combination: LFCCombination,
                               intron_index: int = 0) -> torch.Tensor:
    """
    The fitted LFC of one rate ratio, as a (num_features,) vector.

    The ratio arrives already decomposed into (parameter, coefficient) pairs, which is what lets
    the intron coverage model use this as well: its single LFC is a ratio in its own right and has
    no entry in `RATIO_AS_LFC_COMBINATION`.

    The elongation and splicing LFCs may carry an intron axis, in which case the given intron's
    column is taken; initiation never does.
    """
    terms = [coefficient * _get_lfc_without_intron_axis(state_dict[attribute], intron_index)
             for attribute, coefficient in lfc_combination]
    return torch.stack(terms).sum(dim=0)


def get_lfc_of_tested_ratio(state_dict: StateDict,
                            tested_ratio: TestedRatio,
                            intron_index: int = 0) -> torch.Tensor:
    """
    The fitted LFC of the rate ratio under test, as a (num_features,) vector.

    Three of the six ratios are single parameters; the other three combine two or three of them,
    which is well defined because the parametrized three span all six.

    Used both to hot start the reduced model and to report the tested ratio's effect size, neither
    of which can read a single parameter for the three unparametrized ratios.
    """
    return get_lfc_of_lfc_combination(state_dict, RATIO_AS_LFC_COMBINATION[tested_ratio],
                                      intron_index)


def get_lfc_of_column(lfc_of_tested_ratio: torch.Tensor,
                      feature_names: list[str],
                      column_name) -> float:
    """
    One design column's share of the tested ratio's LFC, for assembling a reported effect size.

    A contrast is a difference between two design columns, named by `lfc_column_positive` and
    `lfc_column_negative` in the LRT metadata. Either can be absent, in which case
    `create_design_matrices.R` writes an empty string, which arrives here as NaN once the TSV has
    been through pandas, and the column contributes zero. That happens in two situations:

    - a categorical contrast against the factor's reference level, which has no column of its own
      because the intercept absorbs it
    - a continuous covariate, which has no second group at all

    So `lfc = get_lfc_of_column(positive) - get_lfc_of_column(negative)` gives the LFC relative to
    the reference in the first case, and the covariate's own LFC in the second.
    """
    if pd.isna(column_name):
        return 0.0
    return lfc_of_tested_ratio[feature_names.index(column_name)].item()


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


def make_lbfgs_optimizer(model: nn.Module) -> optim.LBFGS:
    return optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,
        tolerance_change=1e-9,
        tolerance_grad=1e-7,
        history_size=100,
        line_search_fn='strong_wolfe',
    )


def train_model(
        model: nn.Module,
        optimizer: optim.Optimizer,
        closure: Callable[[], torch.Tensor],
        evaluate_loss: Callable[[], torch.Tensor],
        max_epochs=200,
        max_patience=3,
        loss_change_tolerance=1e-6,
        verbose: bool = False,
) -> tuple[nn.Module, TrainingResults]:
    previous_loss = None
    patience_counter = 0

    # Seeded with the starting point, so that training can never return a model worse than the one
    # it was given. Without this the first epoch's loss is accepted unconditionally, even when the
    # optimizer stepped uphill, which would break the warm starts the LRT restarts rely on.
    initial_loss = evaluate_loss().item()
    best_loss = initial_loss if np.isfinite(initial_loss) else np.inf
    best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

    converged_within_max_epochs = False
    training_diverged = False

    losses: list[float] = []
    for epoch in range(max_epochs):
        optimizer.step(closure)
        loss_tensor = evaluate_loss()
        loss = loss_tensor.item()
        losses.append(loss)
        if verbose:
            print(f'Epoch {epoch + 1}/{max_epochs}: loss={loss:.6f}', flush=True)
        if not torch.isfinite(loss_tensor):
            training_diverged = True
            break

        if loss < best_loss:
            best_loss = loss
            best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if previous_loss is not None:
            relative_loss_change = abs(loss - previous_loss) / (abs(previous_loss) + 1e-10)
            if relative_loss_change < loss_change_tolerance:
                patience_counter += 1
                if patience_counter >= max_patience:
                    converged_within_max_epochs = True
                    break
            else:
                patience_counter = 0

        previous_loss = loss

    model.load_state_dict(best_state_dict)
    # best_loss, not the last epoch's loss: the model is restored to the best state, and the two
    # differ whenever the run ended on a worse epoch -- most obviously when it diverged, where the
    # last loss is non-finite while the returned parameters are the finite best.
    training_results = TrainingResults(
        final_loss=best_loss,
        converged_within_max_epochs=converged_within_max_epochs,
        num_epochs=epoch + 1,
        losses=losses,
        training_diverged=training_diverged,
    )

    return model, training_results


def _make_closures(
        optimizer: optim.Optimizer,
        compute_loss: Callable[[], torch.Tensor],
) -> tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]]:
    """
    Wrap a loss function as the (closure, evaluate_loss) pair train_model expects.

    A non-finite loss is reported to the optimizer as LOSS_CLAMP_VALUE rather than propagated,
    so that LBFGS' line search backs off instead of stepping on a NaN gradient.
    """

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = compute_loss()
        if not torch.isfinite(loss):
            return torch.as_tensor(LOSS_CLAMP_VALUE, dtype=loss.dtype, device=loss.device)
        loss.backward()
        return loss

    def evaluate_loss() -> torch.Tensor:
        with torch.no_grad():
            return compute_loss()

    return closure, evaluate_loss


# The ratios the prior is defined over, in the order `estimate_prior_variance.R` writes its
# covariance columns. The other three ratios are linear combinations of these, so penalising these
# three jointly penalises all six; doing it with one covariance rather than six separate weights is
# what keeps the implied prior a distribution at all.
#
# These are the three the models parametrize, and the basis has to be handed its covariances
# rather than reconstruct them. From variances alone a pair's covariance can only be recovered as
# `cov(a, b) = (var(a) + var(b) - var(a - b)) / 2`, which needs `a - b` to be one of the six
# ratios as well, and two ratios combine into a third only when they share a rate. init/deg and
# elong/splice share none, so `get_ratio_lfc_df` publishes `cov_with_*` instead; see
# `get_contrast_covariance`.
#
# `get_prior_penalty` still writes each basis ratio in terms of the parameters through
# `RATIO_AS_LFC_COMBINATION`, which is now the identity here. It is kept general so that the basis
# remains a wire format, fixed by the column names of `regularization_coefficients.tsv`, that can
# move without the parametrization following it.
LFC_PRIOR_BASIS = (TestedRatio.INIT_OVER_DEG,
                   TestedRatio.ELONG_OVER_SPLICE,
                   TestedRatio.SPLICE_OVER_DEG)

# The model attributes the basis ratios are built from. `IntronCoverageModel` holds only
# `lfc_elong_over_splice`, so it has no joint covariance to publish and
# `estimate_prior_variance.R` fits its prior one ratio at a time. Which of the two applies is read
# off the model rather than its class, matching how the R side dispatches on what its input holds.
LFC_PRIOR_BASIS_PARAMETERS = frozenset(
    attribute for basis_ratio in LFC_PRIOR_BASIS
    for attribute, _ in RATIO_AS_LFC_COMBINATION[basis_ratio])


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
    return torch.from_numpy(precision).float().to(device)


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
        )
        if prior_precision is not None:
            loss += get_prior_penalty(model, prior_precision)
        return loss

    return _make_closures(optimizer, _compute_loss)


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

    return _make_closures(optimizer, _compute_loss)


def get_flat_parameter_offsets(model_parameters: dict[str, torch.Tensor]) -> dict[str, int]:
    """
    Where each parameter's elements start once all parameters are concatenated into one vector.

    That vector indexes the rows of the flattened Fisher information and of the parameter
    dataframe alike, which is what lets a contrast name the rows it combines.
    """
    parameter_offsets: dict[str, int] = {}
    offset = 0
    for name, parameter in model_parameters.items():
        parameter_offsets[name] = offset
        offset += parameter.numel()
    return parameter_offsets


def build_hessian_matrix_from_dict(hessian_dict: dict[str, dict[str, torch.Tensor]],
                                   hessian_variables: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Assemble the per-parameter-pair Hessian blocks into a single matrix.

    `torch.func.hessian` returns a nested dict mirroring the parameter tree, each block
    carrying the natural shape of both parameters it relates. Every block is reshaped down
    to two axes and placed at its offset, which is what the rank-revealing QR and the
    inverse downstream need.

    The matrix carries the variables' own device and dtype, so everything downstream of it reads
    its precision off the model rather than off the default dtype.

    Row/column order is load-bearing: `add_wald_test_results` maps rows of this matrix onto rows
    of the parameter dataframe by position, so the variables are concatenated in
    `get_hessian_variables` order and each block is flattened in C order, matching
    `build_param_df` in models.py.
    """
    param_names = list(hessian_variables.keys())
    param_sizes = {name: hessian_variables[name].numel() for name in param_names}
    param_offsets = get_flat_parameter_offsets(hessian_variables)
    total_param_size = sum(param_sizes.values())
    # Device and dtype both follow the variables, so a model in double precision gets a Hessian
    # in double precision. Fixing the dtype here instead would silently round every block down to
    # the default one on assignment.
    reference = next(iter(hessian_variables.values()))
    hessian_matrix = torch.zeros((total_param_size, total_param_size),
                                 device=reference.device, dtype=reference.dtype)

    for param_1 in param_names:
        for param_2 in param_names:
            block = hessian_dict[param_1][param_2].detach()
            block = block.reshape(param_sizes[param_1], param_sizes[param_2])
            hessian_matrix[
                param_offsets[param_1]:param_offsets[param_1] + param_sizes[param_1],
                param_offsets[param_2]:param_offsets[param_2] + param_sizes[param_2]
            ] = block

    return hessian_matrix


# The two values a plug-in model holds as buffers. The count arm still depends on both, so the
# Hessian is taken over them as well as over the free parameters, which is what gives the mixed
# block the two-stage variance correction needs.
PLUG_IN_VALUE_NAMES = (str(LFCParameter.ELONG_OVER_SPLICE), 'intercept_pi_logit')


def get_hessian_variables(model: nn.Module) -> dict[str, torch.Tensor]:
    """
    The variables the Hessian is taken with respect to, in the order its rows are indexed by.

    These are tensors the model already holds. `torch.func.hessian` differentiates with respect
    to whatever keys this dict carries, and the loss closure substitutes those same keys back
    into the model through `functional_call`, which accepts a buffer exactly where it accepts a
    parameter. A value frozen as a buffer is therefore still differentiable, and this is the one
    place that says so.

    The free parameters come first, in `named_parameters()` order, so the row order the parameter
    table and the Wald contrasts rely on is the same whether or not the model is a plug-in one. A
    plug-in model appends the two values it took from the coverage fit, which the count arm still
    depends on and which the two-stage correction needs a mixed block over.
    """
    values = dict(model.named_parameters())
    # IntronCoverageModel carries no such flag: it is the model a coverage fit comes from, so
    # whether coverage is plugged in is not a question about it.
    if not getattr(model, 'coverage_is_plugged_in', False):
        return values
    buffers = dict(model.named_buffers())
    return values | {name: buffers[name] for name in PLUG_IN_VALUE_NAMES}


def get_fisher_information_matrix(model: nn.Module,
                                  loss_fn: Callable[[StateDict], torch.Tensor]) -> torch.Tensor:
    """
    The observed Fisher information at the model's current parameters.

    This is the Hessian of `loss_fn`, which is the same thing only because the loss is a negative
    log-likelihood. Pass an unregularized loss: with an L2 penalty in it the Hessian is that of
    the penalized objective and no longer an information matrix, so a Wald test built on it would
    understate the uncertainty. "Observed" because it is evaluated on this dataset at the fitted
    parameters, rather than being an expectation over datasets.

    Everything below this function calls the result a Hessian, everything above it the Fisher
    information.

    For a plug-in model it spans the plugged-in values too (`get_hessian_variables`), and then
    only the free block is an information matrix: the plugged-in values' own information came from
    stage one and is carried in the handoff instead. The other two blocks are the count arm's
    curvature in the plugged-in directions, once against the free parameters and once against
    themselves. `get_two_stage_parameter_covariance` reads the three apart and uses the first
    two, the mixed one being how it couples the stages; the plugged-in block itself goes unread,
    stage one's information taking its place.
    """
    hessian_variables = get_hessian_variables(model)
    hessian_dict = hessian(loss_fn)(hessian_variables)
    hessian_matrix = build_hessian_matrix_from_dict(hessian_dict, hessian_variables)
    return hessian_matrix


class ParameterCovariance(NamedTuple):
    """
    The inverse Fisher information, restricted to the parameters that are identifiable.

    `matrix` inverts the identifiable submatrix, and `rows` says which row of the full Fisher
    information each of its rows came from, which is also a row of the parameter dataframe since
    the two share an order. `num_parameters` is the length of that dataframe. Under a plug-in fit
    `rows` runs past it, the covariance covering the plugged-in values as well, and those have no
    row in the dataframe; see `get_two_stage_parameter_covariance`.

    Unidentifiable parameters are left out rather than given an infinite variance: the information
    is singular in their directions, so they have no standard error, and neither has any contrast
    involving them. Inverting the submatrix is also what makes a contrast on a single parameter
    come out with exactly that parameter's own standard error.
    """
    matrix: torch.Tensor
    rows: np.ndarray
    num_parameters: int


def get_eigenvalue_tolerance(eigenvalues: torch.Tensor,
                             data_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Below which magnitude an eigenvalue counts as zero.

    The tolerance `torch.linalg.matrix_rank` applies, so rank, negative curvature and a flat
    direction are decided against one notion of numerically zero. It scales with the largest
    eigenvalue, which makes it a statement about the matrix as a whole rather than about any one
    direction in it.

    `data_dtype` is the precision the entries were computed in, for a matrix now held in a wider
    one. Promoting a matrix to double for the arithmetic leaves its entries no more accurate than
    they were, so a tolerance read off the container would be orders too tight and an exact zero
    would come back looking like a small nonzero eigenvalue.
    """
    eps = torch.finfo(data_dtype if data_dtype is not None else eigenvalues.dtype).eps
    return eigenvalues.shape[0] * eps * eigenvalues.abs().max()


def is_positive_semidefinite(matrix: torch.Tensor) -> bool:
    """
    Whether a symmetric matrix carries no negative curvature, to the tolerance the rank test uses.

    An eigenvalue small enough to be rounding noise is therefore not read as negative curvature.
    Semidefinite rather than definite, so that a matrix with a flat direction passes: a caller
    that also needs definiteness establishes full rank separately, the two together implying it.
    An all-zero matrix passes as well, which matters where its rows are still being handed on.
    """
    if matrix.shape[0] == 0:
        # Rank zero leaves an empty subspace, which holds no direction that could carry negative
        # curvature. Reporting it as semidefinite keeps it on the path it already took, where
        # every parameter comes out unidentifiable and nothing has a standard error.
        return True
    eigenvalues = torch.linalg.eigvalsh(matrix)
    return bool(eigenvalues[0] >= -get_eigenvalue_tolerance(eigenvalues))


def is_positive_definite(matrix: torch.Tensor) -> bool:
    """
    Whether a symmetric matrix carries curvature in every direction, to the rank test's tolerance.

    For a caller that inverts the matrix whole, where a flat direction leaves it singular.
    `is_positive_semidefinite` passes such a direction on purpose, which is what a caller wanting
    definiteness needs when it first restricts to a subspace of full rank.

    An empty matrix fails, there being no inverse of it to go on with.
    """
    if matrix.shape[0] == 0:
        return False
    eigenvalues = torch.linalg.eigvalsh(matrix)
    return bool(eigenvalues[0] > get_eigenvalue_tolerance(eigenvalues))


def try_invert(matrix: torch.Tensor) -> Optional[torch.Tensor]:
    """
    A matrix's inverse, or None where it has none. Callers establish definiteness themselves.

    A true inverse rather than a pseudoinverse. A pseudoinverse zeroes whatever singular value
    falls below its tolerance, turning a direction the matrix barely determines into one it pins
    exactly, and each caller here would read that the wrong way round. Inverting an information
    matrix, it reports a *smaller* variance for a direction the data say least about. Inverting a
    prior covariance, it reports *no* penalty along the direction the prior constrains hardest. A
    true inverse gives a large value instead, and refuses outright once the matrix is singular, so
    the failure is visible in the output rather than hidden in it.

    None rather than the exception, so a matrix that is numerically singular costs only what was
    being read off it: one gene's Wald results, or one design column's joint prior.
    """
    try:
        return torch.linalg.inv(matrix)
    except torch.linalg.LinAlgError:
        return None


def get_parameter_covariance(
        fisher_information_matrix: torch.Tensor) -> Optional[ParameterCovariance]:
    """
    Invert the Fisher information on the subspace where it has full rank.

    None when the information has non-finite entries, when the subspace it is inverted on is not
    positive definite, and when that subspace turns out singular anyway: the rank-revealing QR
    gives it the right rank, not a smallest singular value clear of the inverse's own tolerance.
    Any of the three leaves no Wald test at all, neither for the individual parameters nor for a
    contrast of them.

    Positive definiteness is a separate question from rank, and has to be asked separately: rank is
    decided on the magnitude of the eigenvalues, so an indefinite matrix is full rank whenever none
    of its eigenvalues is near zero. The information is the Hessian at the point the optimizer
    stopped, and the optimizer stops on the gradient and the loss rather than on a second-order
    condition, so that point can carry a direction of negative curvature. It is then not a maximum
    of the likelihood, the parameters are not a maximum likelihood estimate, and the inverse is not
    a covariance matrix -- a variance can come out negative, and a contrast's variance can fall
    below what the variances of its terms allow. None of its entries is a standard error, including
    the ones that look reasonable, which is why this rejects the whole fit rather than the
    offending direction.
    """
    if not torch.isfinite(fisher_information_matrix).all().item():
        return None

    rank = torch.linalg.matrix_rank(fisher_information_matrix).item()

    # Rank-revealing QR is currently not available in Pytorch (see https://github.com/pytorch/pytorch/issues/10454),
    # so we are going to use SciPy implementation.
    _, _, qr_pivots = scipy.linalg.qr(fisher_information_matrix.cpu().numpy(), pivoting=True)
    identifiable_indices = np.sort(qr_pivots[:rank])
    identifiable_subset = fisher_information_matrix[identifiable_indices][:, identifiable_indices]

    # identifiable_subset has full rank by the way it is constructed, and full rank together with
    # no negative curvature is positive definiteness, which is what makes the inverse a covariance
    # matrix. Full rank on its own does not: an indefinite matrix is full rank whenever none of
    # its eigenvalues is near zero.
    if not is_positive_semidefinite(identifiable_subset):
        return None

    matrix = try_invert(identifiable_subset)
    if matrix is None:
        return None

    return ParameterCovariance(matrix=matrix.cpu(),
                               rows=identifiable_indices,
                               num_parameters=fisher_information_matrix.shape[0])


# How much of the flat subspace a parameter may lie in and still be reported. Dimensionless, the
# quantity being a projection of a coordinate direction, so one fixed number covers every matrix.
#
# The model's own structure puts the two quantities it has to separate far apart, and does it
# without reference to the data. An intron's predicted reads are
#
#     mu = exp(intercept_unspliced + offset) * (exp(intercept_pi_logit + nascent_lfc)
#                                               + exp(unspliced_lfc)),
#
# so d log(mu) / d intercept_unspliced = 1 while d log(mu) / d intercept_pi_logit = pi, in every
# sample. In the plane those two parameters span the count arm's score is therefore always along
# (1, pi), making its information there rank one however many samples there are: the number of
# observations and the design matrix scale that one direction and leave its orthogonal complement
# flat. So a coverage fit that could not pin the logit down leaves a flat direction along
# (-pi, 1) / sqrt(1 + pi^2), which projects at least 1 / sqrt(2) onto the logit itself and
# pi / sqrt(1 + pi^2) onto the free intercept beside it. Its reach into any LFC is orders smaller
# again, the LFCs being what the design separates while this plane holds no design at all.
#
# Any value between those two reaches therefore drops the logit the coverage fit could not
# determine and keeps every LFC, for every pi, every intron count, every design and either
# arithmetic precision. Inside that range what it decides is the free intercept beside the logit,
# at the pi solving pi / sqrt(1 + pi^2) = the threshold, so the value below is to within half a
# percent the nascent fraction under which that intercept is still reported.
#
# What the bound does rest on is that no design column is constant across samples, since such a
# column is confounded with the model's own intercepts and the flat direction is then no longer
# confined to the plane above. `create_design_matrices.R` drops `(Intercept)` from every design
# matrix it writes, which is what rules that out.
#
# The rule is one-sided because the ends of pi's range are not mirror images. At pi -> 0 there
# are no nascent reads, so the logit cannot move the counts and its reach into everything else
# vanishes: irrelevant. At pi -> 1 every intron read is nascent, so the logit and the unspliced
# intercept scale the total by the same factor: redundant rather than irrelevant, and both have
# to go. The coverage arm is the symmetric one, its information about the logit going as
# pi * (1 - pi).
MIN_NULL_SPACE_COMPONENT = 0.1


def get_rows_outside_null_space(matrix: torch.Tensor,
                                data_dtype: Optional[torch.dtype] = None) -> np.ndarray:
    """
    The rows of a symmetric positive semidefinite matrix that no flat direction reaches.

    Dropping *rows* rather than directions is what makes the result invertible while still being
    addressable by parameter, which is what the parameter table and the Wald contrasts need. A
    row that lies along a flat direction has no variance to report: the information says nothing
    about that direction, so nothing is known about any parameter with a component in it. The
    rows that survive are then reported conditional on the dropped ones, inverting a submatrix
    being what that amounts to, so a standard error beside a dropped parameter can come out
    slightly below what the same fit reported with that parameter constrained. This is the
    convention `get_parameter_covariance` already follows, there being no finite marginal
    variance to report instead.

    `get_parameter_covariance` asks a narrower version of the same question through a
    rank-revealing QR, which drops one row per flat direction and keeps the rest. That is the
    right answer when the flat directions are themselves coordinate directions, which is the case
    for a single-stage fit, where a parameter is unidentifiable when the data say nothing about
    it. A two-stage information matrix is different: a flat direction of stage one reappears in
    the joint information tilted into stage two's free parameters, so it can run across several
    coordinates at once and QR would keep all but one of them, each with the standard error it
    would have had if the flat direction were known.

    The surviving submatrix is positive definite whenever the whole matrix is semidefinite. A
    vector in its kernel, extended by zeros, has zero curvature in the full matrix and so lies in
    its null space, while being zero on every row this dropped -- and a null space of dimension
    `d` over `n` rows always has some row projecting at least `sqrt(d / n)` into it, so there is
    always such a row.
    """
    # Scaled to a unit diagonal before the flat directions are looked for, because the tolerance
    # is relative to the largest eigenvalue and so compares every direction against the stiffest
    # one. Here that is the count arm's, set by read depth, against a coverage arm carrying one
    # intron's bins, so on the matrix itself a parameter would be called flat for being less
    # curved than something it has nothing to do with, and more of them the deeper the sequencing.
    # The scaling leaves the off-diagonal entries bounded by one as well, the matrix being
    # positive semidefinite, so a small eigenvalue means a linear dependence between coordinates
    # and nothing else.
    #
    # A coordinate carrying no curvature at all is left unscaled instead, so its zero row stays
    # one and reads as the flat direction it is. Semidefiniteness leaves nothing else on that
    # branch: a negative diagonal entry can only be rounding against a zero.
    diagonal = torch.diag(matrix)
    scale = torch.where(diagonal > 0, diagonal.rsqrt(), torch.ones_like(diagonal))
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix * scale[:, None] * scale[None, :])
    tolerance = get_eigenvalue_tolerance(eigenvalues, data_dtype)
    null_space = eigenvectors[:, eigenvalues.abs() <= tolerance]
    if null_space.shape[1] == 0:
        return np.arange(matrix.shape[0])
    # Back into the matrix's own coordinates, which is where the threshold above is derived: `x`
    # is a null vector of the matrix exactly when `x / scale` is one of the scaled matrix. The
    # reach is a projection, so the mapped basis is orthonormalized again before it is measured.
    null_space, _ = torch.linalg.qr(null_space * scale[:, None])
    # How far each coordinate direction projects into the flat subspace. The row norm rather than
    # the largest entry, because `eigh` returns an arbitrary orthonormal basis of a subspace of
    # more than one dimension and only the norm is the same whichever basis it picks.
    reach = null_space.norm(dim=1)
    return np.flatnonzero((reach < MIN_NULL_SPACE_COMPONENT).numpy())


def get_two_stage_parameter_covariance(
        hessian_matrix: torch.Tensor,
        num_free_parameters: int,
        plug_in_information: torch.Tensor) -> Optional[ParameterCovariance]:
    """
    The covariance of a plug-in fit, corrected for the uncertainty in what was plugged in.

    A stage-two fit that treats the plugged-in values as known understates its own uncertainty,
    because those values are themselves estimates. Murphy and Topel (1985) give the correction;
    with `V2` the naive stage-two covariance, `I_21` the mixed block of the stage-two loss
    Hessian, `V1` the plugged-in values' covariance and `M = V2 I_21`,

        Var(free)             = V2 + M V1 M'
        Cov(free, plugged-in) = -M V1
        Var(plugged-in)       = V1

    The general form carries a third term in the covariance between the two stages' scores. Here
    that term is exactly zero, and not merely asymptotically: the stage-two score is a function
    of the read counts alone, while the stage-one score is that of the coverage shape, which
    given those counts is multinomial and so has conditional mean zero. That is the same
    Poisson-multinomial factorization the joint likelihood rests on, and it is what leaves the
    correction a sum.

    Those three blocks are one matrix inverse, which is the form this computes. Writing `I_1` for
    stage one's own information, so `V1 = I_1^-1`, and `I_22` for the free block of the
    stage-two Hessian, so `V2 = I_22^-1`, the correction above is the inverse of

        J = [ I_22                      I_21 ]
            [ I_12      I_12 V2 I_21 +  I_1  ]

    as follows from `I_22 M = I_21`. Taking that route is what lets stage one hand over its
    information rather than its inverse, and that is the difference between a usable correction
    and none at all: an intron whose pi the coverage shape cannot pin down leaves a row of zeros
    in `I_1`, which `J` carries without complaint, where `V1` would not exist. The flat direction
    is then a flat direction of `J` -- exactly, since `J (-M v; v) = 0` whenever `I_1 v = 0` --
    and `get_rows_outside_null_space` drops the parameters that lie along it, leaving the rest of
    the gene with standard errors. Where `I_1` has no flat direction nothing is dropped and this
    is the plain correction.

    `J` therefore departs from the information of the joint model, which has `I_11 + I_1` in
    that corner with `I_11` the count arm's own curvature in the plugged-in directions. Standing
    `I_12 V2 I_21` there instead is the whole difference between plugging stage one's values in
    and fitting both arms together, and it is why the count arm's curvature in those directions
    is never read.

    The arithmetic runs in double precision and the result is cast back. `J`'s plugged-in corner
    is a sum whose two terms can be many orders apart: a pi against the edge of its range makes
    `I_1` arbitrarily small, while `I_12 V2 I_21` beside it stays the size of the count arm. In
    single precision that product's own rounding error exceeds `I_1` outright, and what is left
    of stage one is noise -- full rank, generically, so the flat direction would not even show up
    as one and the standard errors would come back off the noise.

    The result spans the plugged-in values as well as the free parameters, so a contrast mixing
    the two needs no special case and every rate ratio gets a standard error again, including the
    one that is plugged in -- whose standard error comes back as exactly stage one's own. Its rows
    past `num_parameters` are the plugged-in values, which have no row in the parameter table.

    Positive definiteness needs no separate check here, unlike in `get_parameter_covariance`:
    `J` is `blkdiag(I_22, I_1)` plus `[0; I_12] V2 [0; I_21]`, so it is positive semidefinite
    whenever `I_22` and `I_1` are, and dropping the flat rows leaves it definite. An `I_1` large
    enough to make `V1` zero reproduces the naive covariance, and reports the plugged-in values
    as known.

    None when the free block cannot be inverted, when the mixed block carries non-finite entries,
    and when `J` restricted to the reported rows is singular anyway, which dropping only the rows
    a flat direction reaches does not rule out. A stage one that handed over no `I_1` at all does
    not reach here, there being nothing to correct with; the caller reports no Wald results for
    that fit, and `CoverageFitStatus` on it says why stage one had none.
    """
    # The free block first, on its own, both because `V2` appears in `J` and because this is
    # where an unusable stage-two Hessian is caught. `J` is then built on whatever subspace of
    # the free parameters it could invert.
    covariance_free = get_parameter_covariance(
        hessian_matrix[:num_free_parameters, :num_free_parameters])
    if covariance_free is None:
        return None

    # How accurately `J`'s entries will be known: the precision its two inputs were computed in,
    # not the double it is assembled in. `J` is exactly singular wherever `I_1` is, so reading the
    # tolerance off the container instead would leave that zero looking merely small and an
    # effectively singular matrix would be inverted.
    data_dtype = max((hessian_matrix.dtype, plug_in_information.dtype),
                     key=lambda dtype: torch.finfo(dtype).eps)
    hessian_matrix = hessian_matrix.cpu().double()
    information_free = hessian_matrix[covariance_free.rows][:, covariance_free.rows]
    mixed = hessian_matrix[covariance_free.rows][:, num_free_parameters:]
    plug_in_block = (mixed.T @ covariance_free.matrix.double() @ mixed
                     + plug_in_information.cpu().double())
    joint_information = torch.cat([
        torch.cat([information_free, mixed], dim=1),
        torch.cat([mixed.T, plug_in_block], dim=1)], dim=0)

    # The mixed block is outside what the free block's own check covered, so it is checked here.
    if not torch.isfinite(joint_information).all().item():
        return None

    rows = np.concatenate([covariance_free.rows,
                           np.arange(num_free_parameters, hessian_matrix.shape[0])])
    identifiable = get_rows_outside_null_space(joint_information, data_dtype)
    matrix = try_invert(joint_information[identifiable][:, identifiable])
    if matrix is None:
        return None

    return ParameterCovariance(matrix=matrix.to(covariance_free.matrix.dtype),
                               rows=rows[identifiable],
                               num_parameters=num_free_parameters)


def get_plug_in_coverage_fit(model: IntronCoverageModel,
                             fisher_information_matrix: Optional[torch.Tensor],
                             training_diverged: bool,
                             computes_fisher_information: bool = True) -> PlugInCoverageFit:
    """
    Read one `IntronCoverageModel` fit into the plug-in record, whatever its granularity.

    The Fisher information is handed on as it stands rather than inverted. Inverting it here
    would make an intron whose pi the coverage shape cannot pin down cost the whole fit its
    correction, where a row of zeros costs only that intron;
    `get_two_stage_parameter_covariance` is where the inverse is taken, jointly with stage two's
    own information, and where the flat directions are dropped.

    The rows are this fit's parameters, addressed through `get_flat_parameter_offsets` the same
    way the Wald contrasts are, so this inherits the positional row order rather than assuming
    one. For a fit of all its own parameters that selection is the whole matrix; it is written
    as a selection anyway, so that the row order the consumer relies on is stated here.

    None when the Hessian has non-finite entries, and when it carries negative curvature: the
    optimizer stops on the gradient and the loss rather than on a second-order condition, so the
    point it stopped at need not be a maximum of the likelihood, and the Hessian there is not an
    information matrix. A flat direction is not negative curvature and passes, a matrix that is
    flat in every direction included: its rows are still handed on, and the correction then
    reports whichever free parameters the count arm does not couple to the plugged-in values.

    `computes_fisher_information` is False for the penalized refit and the genome-wide variant,
    neither of which builds a Fisher information at all, so an absent one there is not a failed
    fit.
    """
    with torch.no_grad():
        lfc_elong_over_splice = model.lfc_elong_over_splice.detach()[:, 0].cpu()
        intercept_pi_logit = model.intercept_pi_logit.detach().cpu()

    fisher_information: Optional[torch.Tensor] = None
    if fisher_information_matrix is not None:
        offsets = get_flat_parameter_offsets(dict(model.named_parameters()))
        parameter_rows = (
                [offsets[LFCParameter.ELONG_OVER_SPLICE] + feature_index
                 for feature_index in range(len(model.feature_names))]
                + [offsets['intercept_pi_logit'] + intron_index
                   for intron_index in range(len(model.intron_names))])
        selected = fisher_information_matrix[parameter_rows][:, parameter_rows].cpu()
        if torch.isfinite(selected).all().item() and is_positive_semidefinite(selected):
            fisher_information = selected

    if training_diverged:
        status = CoverageFitStatus.TRAINING_DIVERGED
    elif fisher_information is None and computes_fisher_information:
        status = CoverageFitStatus.NOT_ESTIMABLE
    else:
        status = CoverageFitStatus.OK

    return PlugInCoverageFit(intron_names=list(model.intron_names),
                             lfc_elong_over_splice=lfc_elong_over_splice,
                             intercept_pi_logit=intercept_pi_logit,
                             fisher_information=fisher_information,
                             status=status)


def concat_plug_in_coverage_fits(fits: list[PlugInCoverageFit],
                                 feature_names: list[str],
                                 is_regularized: bool = False) -> PlugInCoverageFits:
    """
    Collect one chunk's fits into the artifact it writes.

    `feature_names` is passed in, so that a chunk holding no fits still writes a well-formed
    artifact, and so that a Fisher information of the wrong width is caught here rather than much
    later as a correction built from the wrong parameters.
    """
    for fit in fits:
        expected_side = len(feature_names) + len(fit.intron_names)
        expected_shape = (expected_side, expected_side)
        if (fit.fisher_information is not None
                and tuple(fit.fisher_information.shape) != expected_shape):
            raise ValueError(
                f'Coverage fit of {fit.intron_names} has a '
                f'{tuple(fit.fisher_information.shape)} Fisher information, expected '
                f'{expected_shape} for {len(feature_names)} features.')
    return PlugInCoverageFits(fits=list(fits), feature_names=list(feature_names),
                              is_regularized=is_regularized)


def merge_plug_in_coverage_fits(artifacts: list[PlugInCoverageFits]) -> PlugInCoverageFits:
    """
    One chunk's coverage fits per artifact, into the single artifact a plug-in stage two reads.

    Every chunk has to describe the same design and the same stage, a consumer reading one
    `feature_names` and one `is_regularized` for all of them.

    No intron may appear twice. Chunks cover disjoint introns by construction, so a repeat is a
    wiring fault rather than something the data can produce, and it would be a quiet one:
    `PlugInCoverage.from_fits` keeps whichever fit it indexed last, and the Fisher information it
    lays out is block diagonal by fit only because the fits are over disjoint introns.
    """
    if not artifacts:
        raise ValueError('No coverage fit artifacts to merge.')
    feature_names = artifacts[0].feature_names
    is_regularized = artifacts[0].is_regularized
    for artifact in artifacts[1:]:
        if artifact.feature_names != feature_names:
            raise ValueError(f'Coverage fit chunks describe different designs: {feature_names} '
                             f'and {artifact.feature_names}.')
        if artifact.is_regularized != is_regularized:
            raise ValueError('Coverage fit chunks mix the regularized refit with the fit it '
                             'started from.')

    fits = [fit for artifact in artifacts for fit in artifact.fits]
    covered: set[str] = set()
    repeated: set[str] = set()
    for fit in fits:
        for intron_name in fit.intron_names:
            if intron_name in covered:
                repeated.add(intron_name)
            covered.add(intron_name)
    if repeated:
        raise ValueError(f'Coverage fit chunks cover introns {sorted(repeated)} more than once.')
    return concat_plug_in_coverage_fits(fits, feature_names, is_regularized=is_regularized)


def add_wald_test_results(df_param: pd.DataFrame,
                          parameter_covariance: Optional[ParameterCovariance],
                          fisher_information_matrix: torch.Tensor) -> pd.DataFrame:
    """
    Adds results of the Wald test to the dataframe with model parameters.

    The information matrix is taken as well as the covariance built from it, so that the two
    reasons a fit can have no Wald test at all are reported apart: a non-finite Hessian, and a
    Hessian that is finite but not positive definite.
    """
    df_param = df_param.copy()
    df_param['SE'] = np.nan
    df_param['z_score'] = np.nan
    df_param['p_value_wald'] = np.nan
    df_param['identifiable'] = False
    df_param['loss_hessian_is_finite'] = torch.isfinite(fisher_information_matrix).all().item()
    df_param['loss_hessian_is_positive_definite'] = parameter_covariance is not None
    if parameter_covariance is None:
        return df_param

    if not df_param.index.equals(pd.RangeIndex(parameter_covariance.num_parameters)):
        raise ValueError(f'Unexpected value of df_param.index: {df_param=}')

    # A plug-in fit's covariance also covers the values taken from the coverage fit, which are
    # not parameters of this model and have no row here. They are reported by the fit that
    # estimated them.
    has_parameter_row = parameter_covariance.rows < parameter_covariance.num_parameters
    identifiable_indices = parameter_covariance.rows[has_parameter_row]
    # No clamp on the variance. A negative one is below the rounding error of the inverse it was
    # read from, so the variance it stands for is somewhere between zero and that bound rather
    # than at either end, and NaN is the only one of the two a reader cannot mistake for a
    # measurement. A standard error of exactly zero would say the parameter is known exactly, and
    # `estimate_prior_variance.R` would take it as a sampling variance of zero, which makes the
    # covariance matrix it builds per gene singular.
    standard_errors = torch.sqrt(
        torch.diag(parameter_covariance.matrix)).numpy()[has_parameter_row]

    df_param['identifiable'] = df_param.index.isin(identifiable_indices)
    df_param.loc[identifiable_indices, 'SE'] = standard_errors
    df_param.loc[identifiable_indices, 'z_score'] = df_param.loc[identifiable_indices, 'value'] / standard_errors
    df_param.loc[identifiable_indices, 'p_value_wald'] = 2 * (
            1 - stats.norm.cdf(np.abs(df_param.loc[identifiable_indices, 'z_score'])))

    return df_param


def get_contrast_of_ratio(model: nn.Module,
                          lfc_combination: LFCCombination,
                          lfc_column_positive,
                          lfc_column_negative,
                          intron_index: int = 0) -> dict[int, float]:
    """
    One reported LFC of a rate ratio, written as a combination of model parameters.

    Keys are rows of the flattened Fisher information and values their coefficients, so the dot
    product with the fitted parameters is the LFC itself. It is assembled from the same two pieces
    as that LFC: the ratio as a linear combination of LFC parameters, and the comparison as a
    difference of two design columns. A column that is absent contributes nothing, exactly as in
    `get_lfc_of_column`, which is how a single column is asked for -- pass it as the positive one
    and leave the negative NaN.

    `model` has to be the full model. A reduced one holds the tested ratio in `reduced_lfc`, which
    has neither a row in the parameter dataframe nor a standard error to report.

    A plugged-in LFC does have a row, `get_hessian_variables` placing it after the free
    parameters, so a ratio that mixes plugged-in and fitted values is addressed here like any
    other and the two-stage covariance has the entries to evaluate it.
    """
    hessian_variables = get_hessian_variables(model)
    parameter_offsets = get_flat_parameter_offsets(hessian_variables)

    def get_row(lfc_attribute: str, column_name) -> Optional[int]:
        if pd.isna(column_name):
            return None
        lfc = hessian_variables[lfc_attribute]
        feature_index = model.feature_names.index(column_name)
        # Mirrors `_get_lfc_without_intron_axis`: the elongation and splicing LFCs carry a second
        # axis, over the introns or a singleton, while initiation never does.
        index = (feature_index,) if lfc.dim() == 1 else (feature_index, intron_index)
        return parameter_offsets[lfc_attribute] + int(np.ravel_multi_index(index, lfc.shape))

    contrast: dict[int, float] = {}
    for lfc_attribute, ratio_coefficient in lfc_combination:
        for column_name, column_sign in ((lfc_column_positive, 1.0),
                                         (lfc_column_negative, -1.0)):
            row = get_row(lfc_attribute, column_name)
            if row is not None:
                contrast[row] = contrast.get(row, 0.0) + ratio_coefficient * column_sign
    return contrast


def get_contrast_covariance(parameter_covariance: Optional[ParameterCovariance],
                            contrast_a: dict[int, float],
                            contrast_b: dict[int, float]) -> float:
    """
    `c_a' Sigma c_b` over the rows the two contrasts touch, or NaN where it is unavailable.

    The covariance between two reported effect sizes, as the same quadratic form with one contrast
    twice is the variance of a single one -- which is how `get_contrast_wald_results` gets it, so
    the diagonal of what this returns and the reported standard errors cannot disagree.

    Each contrast is one linear functional of the parameters, as `get_contrast_of_ratio` builds
    it: keys are rows of the flattened parameter vector and values their coefficients, so a
    contrast already pins down parameter, design column and intron and the result is a scalar.
    A plug-in fit's plugged-in values sit past the last row of the parameter table, which this
    needs no special case for.

    It is published because the empirical Bayes step fits a prior over the three basis ratios
    jointly and so needs each unit's whole sampling covariance. Reading that off a variance column
    alone is possible only for a basis whose ratios pairwise share a rate, which the
    parametrization does not; see `LFC_PRIOR_BASIS`.

    NaN whenever either contrast is empty or touches a row the covariance does not report, which
    is the rule `get_contrast_wald_results` already applies to a single contrast. A covariance
    involving an unidentifiable parameter is unavailable, and the ratio it belongs to carries no
    standard error for that unit either, so it is dropped there regardless.
    """
    if parameter_covariance is None or not contrast_a or not contrast_b:
        return np.nan

    position_in_covariance = {row: position
                              for position, row in enumerate(parameter_covariance.rows)}
    rows = sorted(set(contrast_a) | set(contrast_b))
    if not all(row in position_in_covariance for row in rows):
        return np.nan

    positions = [position_in_covariance[row] for row in rows]
    dtype = parameter_covariance.matrix.dtype
    weights_a = torch.tensor([contrast_a.get(row, 0.0) for row in rows], dtype=dtype)
    weights_b = torch.tensor([contrast_b.get(row, 0.0) for row in rows], dtype=dtype)
    submatrix = parameter_covariance.matrix[positions][:, positions]
    return (weights_a @ submatrix @ weights_b).item()


def get_contrast_wald_results(parameter_covariance: Optional[ParameterCovariance],
                              contrast: dict[int, float],
                              lfc: float) -> dict[str, float | bool]:
    """
    Wald test of one contrast, as the columns reported next to its effect size.

    The standard error is sqrt(c' Sigma c) over the contrast's own entries of the inverse Fisher
    information, so a contrast on a single parameter reproduces the standard error reported for
    that parameter in the parameter table. A contrast touching an unidentifiable parameter has no
    standard error at all and is reported as not identifiable, which is the conservative reading
    when the ratio combines two parameters and only one of them is identifiable.

    An empty contrast is reported the same way: there is nothing for the covariance to be
    evaluated on, and a standard error of zero would instead say the effect size was known
    exactly.
    """
    not_identifiable = {'lfc_se': np.nan,
                        'z_score': np.nan,
                        'p_value_wald': np.nan,
                        'lfc_identifiable': False}
    variance = get_contrast_covariance(parameter_covariance, contrast, contrast)
    # NaN is how `get_contrast_covariance` reports a contrast it cannot evaluate at all, which is
    # a different thing from one whose variance came out non-positive.
    if np.isnan(variance):
        return not_identifiable

    # Unclamped, and so NaN once the variance comes out negative, for the reason given in
    # add_wald_test_results. Spelled out rather than left to `sqrt`, which would warn.
    standard_error = float(np.sqrt(variance)) if variance >= 0.0 else np.nan

    z_score = lfc / standard_error if standard_error > 0 else np.nan
    return {'lfc_se': standard_error,
            'z_score': z_score,
            'p_value_wald': 2 * (1 - stats.norm.cdf(np.abs(z_score))),
            'lfc_identifiable': True}


class RatioToReport(NamedTuple):
    """
    One rate ratio as it should appear in `ratio_lfcs`, for one intron or shared across them.

    `lfc_combination` is the ratio decomposed into (parameter, coefficient) pairs, as
    `RATIO_AS_LFC_COMBINATION` gives it for the per-gene model. `intron_name` is what the row is
    labelled with and `intron_index` which column of an intron-specific LFC to read; both are
    unused for a ratio the model shares across introns.
    """
    ratio: str
    lfc_combination: LFCCombination
    intron_name: Optional[str] = None
    intron_index: int = 0


def get_ratio_lfc_df(model: nn.Module,
                     parameter_covariance: Optional[ParameterCovariance],
                     ratios_to_report: list[RatioToReport],
                     with_standard_errors: bool = True) -> pd.DataFrame:
    """
    Every rate ratio's LFC per design column, with its Wald standard error.

    One row per ratio, intron and design column. Unlike the test results, which report a contrast
    between two design columns, a row here is a single column's own coefficient. That is the
    quantity the L2 penalty acts on, so it is also the one a prior has to be fitted for, and it is
    why the three ratios that are not parameters appear here on the same footing as the three that
    are -- their standard errors need the whole inverse Fisher information rather than one
    diagonal entry of it.

    `cov_with_*` carries each row's covariance with each of `LFC_PRIOR_BASIS`, at the same intron,
    which is the unit the prior is fitted over. Those are what let `estimate_prior_variance.R`
    assemble a unit's sampling covariance directly: the basis cannot be recovered from variances
    alone, for the reason given at `LFC_PRIOR_BASIS`. They are absent for a model whose ratio
    space is one-dimensional, which has no joint covariance to report, and the R side dispatches
    on that the same way.

    z scores and p values are deliberately left out: a per-column Wald test is not a result the
    pipeline reports, and the tested comparisons already carry theirs in the test results.

    `with_standard_errors` is a property of the caller rather than of `parameter_covariance`, so
    that one gene whose Hessian came out non-finite cannot give its chunk a different set of
    columns from the rest. The regularized refit passes False: the Hessian of a penalized
    objective is not an information matrix, so it has no standard error to report.
    """
    state_dict = model.state_dict()
    # Whether this model's ratio space is the full three dimensions, and so whether there is a
    # joint covariance for the prior to be fitted against.
    has_prior_basis = all(hasattr(model, attribute)
                          for attribute in LFC_PRIOR_BASIS_PARAMETERS)
    rows: list[dict] = []
    for ratio_to_report in ratios_to_report:
        lfc_of_ratio = get_lfc_of_lfc_combination(state_dict, ratio_to_report.lfc_combination,
                                                  ratio_to_report.intron_index)
        for feature_index, feature_name in enumerate(model.feature_names):
            lfc = lfc_of_ratio[feature_index].item()
            row = {'rate_ratio': ratio_to_report.ratio,
                   'intron_name': ratio_to_report.intron_name,
                   'feature_name': feature_name,
                   'lfc': lfc}
            if with_standard_errors:
                def get_contrast(lfc_combination: LFCCombination) -> dict[int, float]:
                    return get_contrast_of_ratio(model, lfc_combination,
                                                 lfc_column_positive=feature_name,
                                                 lfc_column_negative=np.nan,
                                                 intron_index=ratio_to_report.intron_index)

                contrast = get_contrast(ratio_to_report.lfc_combination)
                wald_results = get_contrast_wald_results(parameter_covariance, contrast, lfc)
                row['lfc_se'] = wald_results['lfc_se']
                row['lfc_identifiable'] = wald_results['lfc_identifiable']
                if has_prior_basis:
                    # This ratio's covariance with each basis ratio, at the same intron, that
                    # being the unit the prior is fitted over. For a basis ratio the entry
                    # against itself is its own variance, so the three rows of a unit's
                    # sampling covariance are read straight off these columns.
                    for basis_ratio in LFC_PRIOR_BASIS:
                        row[f'cov_with_{basis_ratio}'] = get_contrast_covariance(
                            parameter_covariance, contrast,
                            get_contrast(RATIO_AS_LFC_COMBINATION[basis_ratio]))
            rows.append(row)
    return pd.DataFrame(rows)


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
                                       num_reduced_models=len(lrt_inputs))
    model_full, results_full = fits.model_full, fits.results_full

    # Everything below is derived from the full fit, so it has to come after the restart loop:
    # a restart replaces the full model, and the parameter values, standard errors, chi-squared
    # statistics, effect sizes and cached state dict all describe whichever fit was kept.
    model_param_df = model_full.get_param_df()
    model_param_df['gene_name'] = gene_data.gene_name
    model_param_df['loss_full_model'] = results_full.final_loss
    model_param_df['training_diverged_full_model'] = results_full.training_diverged
    model_param_df[
        'training_converged_within_max_epochs_full_model'] = results_full.converged_within_max_epochs
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
                                predicted_pi=predicted_pi)

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
        results_reduced = fits.results_reduced[lrt_index]
        tested_intron_index = _get_tested_intron_index(gene_data, lrt_specification)
        # Read off the tested ratio rather than the parameter table: three of the six ratios have
        # no row of their own there.
        lfc_of_tested_ratio = get_lfc_of_tested_ratio(
            state_dict_model_full, lrt_specification.tested_ratio, tested_intron_index)

        test_result = lrt_input.lrt_metadata_row.to_dict()
        test_result['rate_ratio'] = lrt_specification.tested_ratio
        test_result['gene_name'] = gene_data.gene_name
        test_result['intron_name'] = lrt_specification.tested_intron
        test_result['training_diverged_reduced_model'] = results_reduced.training_diverged
        test_result[
            'training_converged_within_max_epochs_reduced_model'] = results_reduced.converged_within_max_epochs
        test_result['num_epochs_reduced_model'] = results_reduced.num_epochs
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
        test_result['loss_reduced_model'] = results_reduced.final_loss
        test_result['chi2_test_statistics'] = 2 * (results_reduced.final_loss
                                                   - results_full.final_loss)
        test_result['p_value'] = 1 - stats.chi2.cdf(test_result['chi2_test_statistics'],
                                                    df=test_result['lrt_df'])
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

    `plug_in_coverage_fits` has to be the same fits the unregularized model was given, so that the
    refit is the same model and the hot start lands in it. Passing none where the first fit had
    them would silently turn the plugged-in values back into free parameters: a state dict names
    them identically either way, so `load_state_dict` accepts it and the refit then estimates
    against the count arm what stage one read off the coverage shape.

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
    model_param_df[
        'training_converged_within_max_epochs_regularized_model'] = training_results_regularized.converged_within_max_epochs

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
    one gene's, which changes two things. Training goes through `_train_two_stage`, because there
    are then tens of thousands of per-intron intercepts against that one LFC and optimising the
    LFC alone first is far better conditioned. And no Wald test is computed, because the Fisher
    information would be one row and column per intercept, making it very large.

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
            return _train_two_stage(
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
    model_param_df['training_converged_within_max_epochs_full_model'] = results_full.converged_within_max_epochs
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

        chi2_stat = 2 * (results_reduced.final_loss - results_full.final_loss)
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
        test_result['loss_reduced_model'] = results_reduced.final_loss
        test_result['chi2_test_statistics'] = chi2_stat
        test_result['p_value'] = 1 - stats.chi2.cdf(chi2_stat, df=lrt_row['lrt_df'])
        test_result['training_diverged_reduced_model'] = results_reduced.training_diverged
        test_result['training_converged_within_max_epochs_reduced_model'] = results_reduced.converged_within_max_epochs
        test_result['num_epochs_reduced_model'] = results_reduced.num_epochs
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
        )

    return _make_closures(optimizer, _compute_loss)


# The global models train in two stages and can need many more epochs than a per-gene fit, which
# `train_model` caps at its own, lower default.
MAX_EPOCHS_TWO_STAGE = 1000


def _train_two_stage(
        model: nn.Module,
        global_param_names: set[str],
        make_closures: Callable[[optim.Optimizer], tuple[Callable, Callable]],
        label: str = '',
        verbose: bool = False,
) -> tuple[nn.Module, TrainingResults]:
    prefix = f'[{label}] ' if label else ''

    # Stage 1: freeze per-gene/per-intron params; optimize only shared params
    if verbose:
        print(f'{prefix}Stage 1: optimizing shared params ({", ".join(sorted(global_param_names))})', flush=True)
    for name, param in model.named_parameters():
        param.requires_grad_(name in global_param_names)
    stage_1_params = [p for p in model.parameters() if p.requires_grad and p.numel() > 0]
    if stage_1_params:
        optimizer_stage_1 = optim.LBFGS(
            stage_1_params,
            lr=1.0, max_iter=20, tolerance_change=1e-9, tolerance_grad=1e-7,
            history_size=100, line_search_fn='strong_wolfe',
        )
        closure_stage_1, eval_stage_1 = make_closures(optimizer_stage_1)
        model, _ = train_model(model, optimizer_stage_1, closure_stage_1, eval_stage_1,
                               max_epochs=MAX_EPOCHS_TWO_STAGE, verbose=verbose)

    # Stage 2: unfreeze all, refine jointly from warm start
    if verbose:
        print(f'{prefix}Stage 2: joint optimization (all params)', flush=True)
    for param in model.parameters():
        param.requires_grad_(True)
    optimizer_stage_2 = make_lbfgs_optimizer(model)
    closure_stage_2, eval_stage_2 = make_closures(optimizer_stage_2)
    model, training_results = train_model(model, optimizer_stage_2, closure_stage_2, eval_stage_2,
                                          max_epochs=MAX_EPOCHS_TWO_STAGE,
                                          verbose=verbose)
    return model, training_results


def get_global_rna_kinetics_model_results(
        global_gene_data: GlobalGeneData,
        dataset_metadata: DatasetMetadata,
        plug_in_coverage_fits: Optional[list[PlugInCoverageFit]] = None,
        device: str = 'cpu',
        verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit the genome-wide model and run every likelihood ratio test it supports.

    Given `plug_in_coverage_fits`, lfc_elong_over_splice and the pi intercepts come from the
    coverage fits covering this dataset's introns rather than from here. The LFC is shared across
    the whole dataset, so a single fit over all of them is what this takes, and of the three
    ratios the model can test only two are left: elong/splice is the coverage fit's own test.
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
        return _train_two_stage(
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
                                       num_reduced_models=len(lrt_inputs))
    model, training_results = fits.model_full, fits.results_full

    # Derived from the full fit, so after the restart loop: a restart replaces the full model.
    model_param_df = model.get_param_df()
    model_param_df['loss_full_model'] = training_results.final_loss
    model_param_df['training_diverged_full_model'] = training_results.training_diverged
    model_param_df['training_converged_within_max_epochs_full_model'] = training_results.converged_within_max_epochs
    model_param_df['num_epochs_full_model'] = training_results.num_epochs

    full_state = model.state_dict()
    test_results_list: list[dict] = []
    for lrt_index, lrt_input in enumerate(lrt_inputs):
        lrt_row, lrt_specification = lrt_input.lrt_metadata_row, lrt_input.lrt_specification
        results_reduced = fits.results_reduced[lrt_index]
        lfc_of_tested_ratio = get_lfc_of_tested_ratio(full_state, lrt_specification.tested_ratio)

        chi2_stat = 2 * (results_reduced.final_loss - training_results.final_loss)
        test_result = lrt_row.to_dict()
        test_result['rate_ratio'] = lrt_specification.tested_ratio
        test_result['lfc'] = (
                get_lfc_of_column(lfc_of_tested_ratio, feature_names,
                                  lrt_row['lfc_column_positive'])
                - get_lfc_of_column(lfc_of_tested_ratio, feature_names,
                                    lrt_row['lfc_column_negative']))
        test_result['loss_full_model'] = training_results.final_loss
        test_result['loss_reduced_model'] = results_reduced.final_loss
        test_result['chi2_test_statistics'] = chi2_stat
        test_result['p_value'] = 1 - stats.chi2.cdf(chi2_stat, df=lrt_row['lrt_df'])
        test_result['training_diverged_reduced_model'] = results_reduced.training_diverged
        test_result[
            'training_converged_within_max_epochs_reduced_model'] = results_reduced.converged_within_max_epochs
        test_result['num_epochs_reduced_model'] = results_reduced.num_epochs
        test_result['num_full_model_restarts'] = fits.num_full_model_restarts
        test_results_list.append(test_result)

    return model_param_df, pd.DataFrame(test_results_list)


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
    model_param_df[
        'training_converged_within_max_epochs_regularized_model'] = training_results_regularized.converged_within_max_epochs

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
