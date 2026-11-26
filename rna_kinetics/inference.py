from enum import StrEnum
from typing import Callable, NamedTuple, Optional

import numpy as np
import pandas as pd
import scipy
import torch
from scipy import stats
from torch import nn
from torch.func import hessian

from rna_kinetics.models import IntronCoverageModel, RATIO_AS_LFC_COMBINATION, LFCCombination, \
    LFCParameter, CoverageFitStatus, PlugInCoverageFit, PlugInCoverageFits, StateDict, \
    get_lfc_of_lfc_combination, LFC_PRIOR_BASIS, LFC_PRIOR_BASIS_PARAMETERS
from rna_kinetics.fitting import TrainingResults


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

    # `V2` is inverted again here in double rather than taken from `covariance_free`, so that
    # this block does not depend on the precision the fit ran in. `J`'s inverse reproduces the
    # three blocks above only because `I_22 M = I_21` holds exactly, and `J` is built to be
    # exactly singular wherever stage one is, so that cancellation is the whole of what its
    # conditioning rests on. A `V2` carrying error from a narrower dtype leaves the corner
    # inconsistent with the block beside it by about that dtype's `eps`, and the near-singular
    # inverse amplifies it by orders of magnitude: the corrected variances come back wrong,
    # negative ones among them, while the matrix stays finite and definite-looking. Promoting
    # `covariance_free.matrix` cannot recover it, a promoted matrix being no more accurate than
    # the one it came from. With `MODEL_DTYPE` at double the two agree and this inverse is
    # redundant; it is kept because the invariant is about `J`, not about the fit's dtype.
    covariance_free_double = try_invert(information_free)
    if covariance_free_double is None:
        return None
    plug_in_block = (mixed.T @ covariance_free_double @ mixed
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
    df_param.loc[identifiable_indices, 'p_value_wald'] = 2 * stats.norm.sf(
        np.abs(df_param.loc[identifiable_indices, 'z_score']))

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
            'p_value_wald': 2 * stats.norm.sf(np.abs(z_score)),
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


def get_lrt_results(results_full: TrainingResults,
                    results_reduced: Optional[TrainingResults],
                    lrt_df: int) -> dict:
    """
    The likelihood ratio test columns of one row of test results.

    All null where `results_reduced` is None, that is where the reduced model was not fitted, so
    that a table without likelihood ratio tests keeps the columns of one with them.
    """
    if results_reduced is None:
        return {'loss_reduced_model': np.nan,
                'chi2_test_statistics': np.nan,
                'p_value_lrt': np.nan,
                'training_diverged_reduced_model': None,
                'training_converged_reduced_model': None,
                'num_epochs_reduced_model': None}
    chi2_statistic = 2 * (results_reduced.final_loss - results_full.final_loss)
    return {'loss_reduced_model': results_reduced.final_loss,
            'chi2_test_statistics': chi2_statistic,
            'p_value_lrt': stats.chi2.sf(chi2_statistic, df=lrt_df),
            'training_diverged_reduced_model': results_reduced.training_diverged,
            'training_converged_reduced_model': results_reduced.converged,
            'num_epochs_reduced_model': results_reduced.num_epochs}


class JackknifeStatus(StrEnum):
    """Availability of jackknife results for a test of a global model."""
    OK = 'ok'
    # Removing some sample reduces the rank of the design (with intercept). This occurs for a
    # group with a single sample, and whenever the number of samples equals the rank, leaving no
    # residual degrees of freedom.
    RANK_DEFICIENT = 'rank_deficient'
    # At least one jackknife replicate diverged.
    FIT_DIVERGED = 'fit_diverged'


JACKKNIFE_NUMERIC_COLUMNS = ('lfc_se_jackknife', 't_statistic_jackknife', 'df_jackknife',
                             'p_value_jackknife')


def _get_rank_with_intercept(design_matrix: np.ndarray) -> int:
    """Rank of the design matrix augmented with a constant column, which the models' intercepts
    span."""
    constant = np.ones((design_matrix.shape[0], 1))
    return int(np.linalg.matrix_rank(np.hstack([constant, design_matrix])))


def get_jackknife_degrees_of_freedom(design_matrix: np.ndarray) -> int:
    """Residual degrees of freedom of a linear model of the design fitted to one value per sample:
    `n - rank([1, X])`."""
    return design_matrix.shape[0] - _get_rank_with_intercept(design_matrix)


def get_jackknife_design_status(design_matrix: np.ndarray) -> JackknifeStatus:
    """
    Whether the design keeps its rank when any single sample is removed.

    Depends on the design only, so it is evaluated before any fit. The check applies to the whole
    design rather than to individual contrasts, because each jackknife replicate estimates all
    design columns jointly, and a replicate with an inestimable column has no unique optimum.
    """
    rank = _get_rank_with_intercept(design_matrix)
    for sample_index in range(design_matrix.shape[0]):
        if _get_rank_with_intercept(np.delete(design_matrix, sample_index, axis=0)) < rank:
            return JackknifeStatus.RANK_DEFICIENT
    return JackknifeStatus.OK


def get_jackknife_ratio_lfc_df(ratio_lfcs: pd.DataFrame,
                               model_parameters: pd.DataFrame,
                               dropped_sample: str) -> pd.DataFrame:
    """
    Rate ratio LFCs of one jackknife replicate, labelled with the removed sample.

    The training diagnostics identify replicates that reached the epoch limit without meeting the
    stopping rule. They do not identify replicates that met the stopping rule before reaching the
    optimum: the global fits stop on the loss change per epoch, which can fall below the tolerance
    while the loss is still decreasing.
    """
    training_columns = ['loss_full_model', 'num_epochs_full_model',
                        'training_converged_full_model']
    return ratio_lfcs.assign(
        dropped_sample=dropped_sample,
        **{column: model_parameters[column].iloc[0] for column in training_columns})


def add_jackknife_results(test_results: pd.DataFrame,
                          jackknife_ratio_lfcs: pd.DataFrame,
                          design_matrix: np.ndarray,
                          sample_names: list[str]) -> pd.DataFrame:
    """
    Add delete-one-sample jackknife standard errors and t tests to a global model's test results.

    A global LFC is a single effect shared by all genes or introns, so its sampling variance is
    determined by the variation between samples. The likelihood treats genes as independent
    replicates: variation shared by all genes within a sample, whether technical (for example,
    the fraction of nascent RNA captured in a library) or biological (for example, a replicate's
    overall elongation or splicing speed), is averaged over genes as if it were independent
    between them, although it is a single draw per sample. The LRT is therefore anti-conservative for a
    global effect. Refitting the model with each sample removed in turn estimates the variance
    between samples directly.

    `jackknife_ratio_lfcs` contains one fit per removed sample, identified by `dropped_sample`,
    in the format of `ratio_lfcs`: one row per rate ratio and design column. `sample_names` lists
    all samples of the design, in the row order of `design_matrix`; each must appear exactly once
    as `dropped_sample`. Each test's contrast is computed from every replicate as from the full
    fit, and the standard error is
    `sqrt((n - 1) / n * sum_i (lfc_i - lfc_mean)^2)`, where `n` is the number of samples, `lfc_i`
    the contrast estimated with sample `i` removed and `lfc_mean` the average of the `lfc_i`. The
    t statistic is the full fit's `lfc` divided by this standard error, with `n - rank([1, X])`
    degrees of freedom, `X` being the design matrix.

    The jackknife variance is conservative in small groups. For the difference between the means
    of two groups `g = 1, 2`, with `n_g` samples and sample variance `s_g^2` in group `g`, the
    unbiased variance estimate is `sum_g s_g^2 / n_g`; the jackknife variance is
    `(n - 1) / n * sum_g s_g^2 / (n_g - 1)`, which exceeds it by the factor
    `(n - 1) / n * n_g / (n_g - 1)` in each group's term.

    Tests without jackknife results retain the jackknife columns, set to NaN, and
    `jackknife_status` records the reason.
    """
    test_results = test_results.copy()
    for column in JACKKNIFE_NUMERIC_COLUMNS:
        test_results[column] = np.nan
    status = get_jackknife_design_status(design_matrix)
    test_results['jackknife_status'] = str(status)
    if status != JackknifeStatus.OK:
        return test_results

    # Both checks detect wiring errors: the workflow runs one replicate per sample whenever the
    # design passes the check above.
    if sorted(set(jackknife_ratio_lfcs['dropped_sample'])) != sorted(sample_names):
        raise ValueError(f'The jackknife fits leave out '
                         f'{sorted(set(jackknife_ratio_lfcs["dropped_sample"]))}, '
                         f'not every sample of the design: {sorted(sample_names)}.')
    lfc_by_key = jackknife_ratio_lfcs.set_index(
        ['dropped_sample', 'rate_ratio', 'feature_name'])['lfc']
    if not lfc_by_key.index.is_unique:
        raise ValueError('The jackknife fits report a rate ratio and design column more than once '
                         'for the same left-out sample.')

    if jackknife_ratio_lfcs['training_diverged_full_model'].astype(bool).any():
        test_results['jackknife_status'] = str(JackknifeStatus.FIT_DIVERGED)
        return test_results

    num_samples = len(sample_names)
    degrees_of_freedom = get_jackknife_degrees_of_freedom(design_matrix)
    for row_index, row in test_results.iterrows():
        def get_lfc_of_column_without(sample_name: str, column_name) -> float:
            # An absent column contributes zero, as in `get_lfc_of_column`.
            if pd.isna(column_name):
                return 0.0
            return float(lfc_by_key.loc[(sample_name, row['rate_ratio'], column_name)])

        lfc_without_sample = np.array([
            get_lfc_of_column_without(sample_name, row['lfc_column_positive'])
            - get_lfc_of_column_without(sample_name, row['lfc_column_negative'])
            for sample_name in sample_names])
        lfc_se = np.sqrt((num_samples - 1) / num_samples
                         * np.sum((lfc_without_sample - lfc_without_sample.mean()) ** 2))
        t_statistic = row['lfc'] / lfc_se
        test_results.loc[row_index, 'lfc_se_jackknife'] = lfc_se
        test_results.loc[row_index, 't_statistic_jackknife'] = t_statistic
        test_results.loc[row_index, 'df_jackknife'] = degrees_of_freedom
        test_results.loc[row_index, 'p_value_jackknife'] = 2 * stats.t.sf(abs(t_statistic),
                                                                           degrees_of_freedom)
    return test_results
