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
    intron_specific_lfc: Optional[bool] = None  # RNAKineticsModel only: LFC parameter shape


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

    `plug_in_coverage_fit` is set by the coverage models only, and carries no covariance for the
    same reason `ratio_lfcs` carries no standard errors.
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
# These are ratios rather than parameters, and need not be the three the models parametrize: the
# basis is the wire format shared with `estimate_prior_variance.R`, so it is fixed by the column
# names of `regularization_coefficients.tsv`. `get_prior_penalty` writes each basis ratio in terms
# of the parameters through `RATIO_AS_LFC_COMBINATION`, which is also why the penalty itself is
# unchanged by a change of basis: the quadratic form is the same one either way.
LFC_PRIOR_BASIS = (TestedRatio.INIT_OVER_DEG,
                   TestedRatio.ELONG_OVER_DEG,
                   TestedRatio.SPLICE_OVER_DEG)


def get_prior_precision(regularization_coefficients_df: pd.DataFrame,
                        feature_names: list[str],
                        device: str = 'cpu') -> torch.Tensor:
    """
    The inverse prior covariance of the three basis ratios, per design column.

    Reads the covariance `estimate_prior_variance.R` fitted jointly over those three and inverts
    it, giving the matrix the penalty is a quadratic form in. A design column whose covariance is
    incomplete -- some ratio was not estimable there -- falls back to the diagonal built from that
    column's per-ratio `lambda`, which penalises whatever directions were estimable and leaves the
    rest effectively free.
    """
    coefficients = regularization_coefficients_df.set_index(['rate_ratio', 'feature_name'])
    precision = torch.zeros((len(feature_names), len(LFC_PRIOR_BASIS), len(LFC_PRIOR_BASIS)))
    covariance_columns = [f'prior_cov_{ratio.removeprefix("lfc_")}' for ratio in LFC_PRIOR_BASIS]

    for feature_index, feature_name in enumerate(feature_names):
        rows = [coefficients.loc[(ratio, feature_name)] for ratio in LFC_PRIOR_BASIS]
        covariance = np.array([[row[column] for column in covariance_columns] for row in rows],
                              dtype=np.float64)
        if np.isfinite(covariance).all():
            precision[feature_index] = torch.from_numpy(np.linalg.inv(covariance)).float()
        else:
            lambdas = [row['lambda'] for row in rows]
            precision[feature_index] = torch.diag(2 * torch.tensor(lambdas, dtype=torch.float32))
    return precision.to(device)


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
                                   model_parameters: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Assemble the per-parameter-pair Hessian blocks into a single matrix.

    `torch.func.hessian` returns a nested dict mirroring the parameter tree, each block
    carrying the natural shape of both parameters it relates. Every block is reshaped down
    to two axes and placed at its offset, which is what the rank-revealing QR and the
    inverse downstream need.

    Row/column order is load-bearing: `add_wald_test_results` maps rows of this matrix onto rows
    of the parameter dataframe by position, so parameters are concatenated in `named_parameters()`
    order and each block is flattened in C order, matching `build_param_df` in models.py.
    """
    param_names = list(model_parameters.keys())
    param_sizes = {name: model_parameters[name].numel() for name in param_names}
    param_offsets = get_flat_parameter_offsets(model_parameters)
    total_param_size = sum(param_sizes.values())
    device = next(iter(model_parameters.values())).device
    hessian_matrix = torch.zeros((total_param_size, total_param_size), device=device)

    for param_1 in param_names:
        for param_2 in param_names:
            block = hessian_dict[param_1][param_2].detach()
            block = block.reshape(param_sizes[param_1], param_sizes[param_2])
            hessian_matrix[
                param_offsets[param_1]:param_offsets[param_1] + param_sizes[param_1],
                param_offsets[param_2]:param_offsets[param_2] + param_sizes[param_2]
            ] = block

    return hessian_matrix


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
    """
    model_parameters = dict(model.named_parameters())
    hessian_dict = hessian(loss_fn)(model_parameters)
    hessian_matrix = build_hessian_matrix_from_dict(hessian_dict, model_parameters)
    return hessian_matrix


class ParameterCovariance(NamedTuple):
    """
    The inverse Fisher information, restricted to the parameters that are identifiable.

    `matrix` inverts the identifiable submatrix, and `rows` says which row of the full Fisher
    information each of its rows came from, which is also a row of the parameter dataframe since
    the two share an order.

    Unidentifiable parameters are left out rather than given an infinite variance: the information
    is singular in their directions, so they have no standard error, and neither has any contrast
    involving them. Inverting the submatrix is also what makes a contrast on a single parameter
    come out with exactly that parameter's own standard error.
    """
    matrix: torch.Tensor
    rows: np.ndarray
    num_parameters: int


def is_positive_definite(matrix: torch.Tensor) -> bool:
    """
    Whether a symmetric matrix is positive definite, to the tolerance the rank test uses.

    The tolerance is the one `torch.linalg.matrix_rank` applies, so a single notion of
    "numerically zero" covers both checks, and an eigenvalue small enough to be rounding noise is
    not read as negative curvature.
    """
    if matrix.shape[0] == 0:
        # Rank zero leaves an empty subspace, which holds no direction that could carry negative
        # curvature. Reporting it as positive definite keeps it on the path it already took, where
        # every parameter comes out unidentifiable and nothing has a standard error.
        return True
    eigenvalues = torch.linalg.eigvalsh(matrix)
    tolerance = (matrix.shape[0] * torch.finfo(matrix.dtype).eps
                 * eigenvalues.abs().max())
    return bool(eigenvalues[0] > -tolerance)


def get_parameter_covariance(
        fisher_information_matrix: torch.Tensor) -> Optional[ParameterCovariance]:
    """
    Invert the Fisher information on the subspace where it has full rank.

    None when the information has non-finite entries, and also when the subspace it is inverted on
    is not positive definite. Either way there is no Wald test at all, neither for the individual
    parameters nor for a contrast of them.

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

    # identifiable_subset has full rank by the way it is constructed, but full rank does not make
    # it positive definite, and only the latter makes its inverse a covariance matrix.
    if not is_positive_definite(identifiable_subset):
        return None

    return ParameterCovariance(matrix=torch.linalg.pinv(identifiable_subset).cpu(),
                               rows=identifiable_indices,
                               num_parameters=fisher_information_matrix.shape[0])


def get_plug_in_coverage_fit(model: IntronCoverageModel,
                             parameter_covariance: Optional[ParameterCovariance],
                             training_diverged: bool,
                             computes_fisher_information: bool = True) -> PlugInCoverageFit:
    """
    Read one `IntronCoverageModel` fit into the plug-in record, whatever its granularity.

    The covariance is the submatrix of the inverse Fisher information on this fit's parameters,
    addressed through `get_flat_parameter_offsets` the same way the Wald contrasts are, so this
    inherits the positional row order rather than assuming one. For a fit of all its own
    parameters that submatrix is the whole matrix; it is written as a selection anyway, because the
    identifiable subspace can be smaller.

    A single unidentifiable row leaves the whole covariance as None, since stage two freezes the
    parameters as a vector and a correction built from part of that vector would be wrong rather
    than partial.

    `computes_fisher_information` is False for the penalized refit and the genome-wide variant,
    neither of which builds a Fisher information at all, so an absent covariance there is not a
    failed fit.
    """
    with torch.no_grad():
        lfc_elong_over_splice = model.lfc_elong_over_splice.detach()[:, 0].cpu()
        intercept_pi_logit = model.intercept_pi_logit.detach().cpu()

    covariance: Optional[torch.Tensor] = None
    if parameter_covariance is not None:
        offsets = get_flat_parameter_offsets(dict(model.named_parameters()))
        parameter_rows = (
                [offsets[LFCParameter.ELONG_OVER_SPLICE] + feature_index
                 for feature_index in range(len(model.feature_names))]
                + [offsets['intercept_pi_logit'] + intron_index
                   for intron_index in range(len(model.intron_names))])
        position_of_row = {row: position
                           for position, row in enumerate(parameter_covariance.rows)}
        if all(row in position_of_row for row in parameter_rows):
            positions = [position_of_row[row] for row in parameter_rows]
            covariance = parameter_covariance.matrix[positions][:, positions]

    if training_diverged:
        status = CoverageFitStatus.TRAINING_DIVERGED
    elif covariance is None and computes_fisher_information:
        status = CoverageFitStatus.NOT_ESTIMABLE
    else:
        status = CoverageFitStatus.OK

    return PlugInCoverageFit(intron_names=list(model.intron_names),
                             lfc_elong_over_splice=lfc_elong_over_splice,
                             intercept_pi_logit=intercept_pi_logit,
                             covariance=covariance,
                             status=status)


def concat_plug_in_coverage_fits(fits: list[PlugInCoverageFit],
                                 feature_names: list[str],
                                 is_regularized: bool = False) -> PlugInCoverageFits:
    """
    Collect one chunk's fits into the artifact it writes.

    `feature_names` is passed in, so that a chunk holding no fits still writes a well-formed
    artifact, and so that a covariance of the wrong width is caught here rather than much later as
    a correction built from the wrong parameters.
    """
    for fit in fits:
        expected_side = len(feature_names) + len(fit.intron_names)
        expected_shape = (expected_side, expected_side)
        if fit.covariance is not None and tuple(fit.covariance.shape) != expected_shape:
            raise ValueError(
                f'Coverage fit of {fit.intron_names} has a {tuple(fit.covariance.shape)} '
                f'covariance, expected {expected_shape} for '
                f'{len(feature_names)} features.')
    return PlugInCoverageFits(fits=list(fits), feature_names=list(feature_names),
                              is_regularized=is_regularized)


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

    identifiable_indices = parameter_covariance.rows
    # No clamp on the variance: get_parameter_covariance only returns a positive definite matrix,
    # so a negative here would be a bug, and a NaN standard error is how it should show up. Clamping
    # it to zero would instead report an infinite z score and a p value of exactly zero.
    standard_errors = torch.sqrt(torch.diag(parameter_covariance.matrix)).numpy()

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

    A plugged-in LFC has no row either, being a buffer rather than a parameter, so it drops out
    of the contrast the same way an absent design column does. The standard error that comes back
    is then the one conditional on the plugged-in value, which understates the uncertainty until
    the stage-one variance is added back. A contrast left with no rows at all -- the plugged-in
    ratio itself -- is reported as not identifiable rather than as a standard error of zero.
    """
    model_parameters = dict(model.named_parameters())
    parameter_offsets = get_flat_parameter_offsets(model_parameters)

    def get_row(lfc_attribute: str, column_name) -> Optional[int]:
        if pd.isna(column_name) or lfc_attribute not in model_parameters:
            return None
        lfc = model_parameters[lfc_attribute]
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

    An empty contrast is reported the same way. It is what a ratio built entirely from plugged-in
    values comes to, and this Fisher information genuinely has nothing to say about it; a
    standard error of zero would instead say it was known exactly.
    """
    not_identifiable = {'lfc_se': np.nan,
                        'z_score': np.nan,
                        'p_value_wald': np.nan,
                        'lfc_identifiable': False}
    if parameter_covariance is None or not contrast:
        return not_identifiable

    position_in_covariance = {row: position
                              for position, row in enumerate(parameter_covariance.rows)}
    rows = sorted(contrast)
    if not all(row in position_in_covariance for row in rows):
        return not_identifiable

    positions = [position_in_covariance[row] for row in rows]
    weights = torch.tensor([contrast[row] for row in rows],
                           dtype=parameter_covariance.matrix.dtype)
    variance = weights @ parameter_covariance.matrix[positions][:, positions] @ weights
    # Unclamped for the reason given in add_wald_test_results.
    standard_error = torch.sqrt(variance).item()

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

    z scores and p values are deliberately left out: a per-column Wald test is not a result the
    pipeline reports, and the tested comparisons already carry theirs in the test results.

    `with_standard_errors` is a property of the caller rather than of `parameter_covariance`, so
    that one gene whose Hessian came out non-finite cannot give its chunk a different set of
    columns from the rest. The regularized refit passes False: the Hessian of a penalized
    objective is not an information matrix, so it has no standard error to report.
    """
    state_dict = model.state_dict()
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
                contrast = get_contrast_of_ratio(model, ratio_to_report.lfc_combination,
                                                 lfc_column_positive=feature_name,
                                                 lfc_column_negative=np.nan,
                                                 intron_index=ratio_to_report.intron_index)
                wald_results = get_contrast_wald_results(parameter_covariance, contrast, lfc)
                row['lfc_se'] = wald_results['lfc_se']
                row['lfc_identifiable'] = wald_results['lfc_identifiable']
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

    The standard errors reported under the plug-in are conditional on the plugged-in values, so
    they understate the uncertainty until the variance correction is applied downstream.
    """
    gene_data = gene_data.to(device)
    dataset_metadata = dataset_metadata.to(device)

    plug_in_coverage: Optional[PlugInCoverage] = None
    if plug_in_coverage_fits is not None:
        plug_in_coverage = PlugInCoverage.from_fits(plug_in_coverage_fits,
                                                    gene_data.intron_names,
                                                    intron_specific_lfc)
        plug_in_coverage = PlugInCoverage(
            *(value.to(device) for value in plug_in_coverage))
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
    parameter_covariance = get_parameter_covariance(fisher_information_matrix)
    model_param_df = add_wald_test_results(model_param_df, parameter_covariance,
                                           fisher_information_matrix)

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
                                               device='cpu'
                                               ) -> RegularizedFitResults:
    gene_data = gene_data.to(device)
    dataset_metadata = dataset_metadata.to(device)
    model_regularized = RNAKineticsModel(feature_names=dataset_metadata.feature_names,
                                         intron_names=gene_data.intron_names,
                                         intron_specific_lfc=intron_specific_lfc).to(device)
    model_regularized.load_state_dict(hot_start_state_dict)

    prior_precision = get_prior_precision(regularization_coefficients_df,
                                          model_regularized.feature_names, device)

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
    """
    coverage = coverage.to(device)
    dataset_metadata = dataset_metadata.to(device)

    # The intron the LFC is reported under; None when it is shared across several introns, which
    # is always the case for the global model.
    lfc_intron_name = intron_names[0] if len(intron_names) == 1 else None

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
                                   intron_names=intron_names).to(device)

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
            model_full, parameter_covariance,
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
                                                    intron_specific_lfc=False)
        plug_in_coverage = PlugInCoverage(
            *(value.to(device) for value in plug_in_coverage))
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
        device: str = 'cpu',
) -> RegularizedFitResults:
    coverage = input_data.coverage.to(device)
    dataset_metadata = dataset_metadata.to(device)
    regularization_coefficients_df = regularization_coefficients_df.set_index(['rate_ratio', 'feature_name'])

    model_regularized = IntronCoverageModel(
        feature_names=dataset_metadata.feature_names,
        intron_names=input_data.intron_names,
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

    # One intron means the shared LFC is that intron's, so it is reported under its name -- the
    # same rule `IntronCoverageModel.get_param_df` follows.
    lfc_intron_name = (input_data.intron_names[0] if len(input_data.intron_names) == 1 else None)
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
            model_regularized, parameter_covariance=None,
            training_diverged=training_results_regularized.training_diverged,
            computes_fisher_information=False))
