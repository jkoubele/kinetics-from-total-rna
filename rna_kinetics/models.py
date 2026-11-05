import itertools
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional, NamedTuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.nn.functional as functional

from rna_kinetics.data import GeneData, GlobalGeneData


def safe_exp(x: torch.Tensor, output_threshold: float = 1e20) -> torch.Tensor:
    output_threshold_tensor = torch.tensor(output_threshold, dtype=x.dtype, device=x.device)
    input_threshold = torch.log(output_threshold_tensor)
    return torch.where(
        x <= input_threshold,
        torch.exp(x),
        output_threshold_tensor + output_threshold_tensor * (x - input_threshold)
    )


class LFCParameter(StrEnum):
    """
    The models' log fold change parameters, by attribute name.

    Separate from TestedRatio, which enumerates hypotheses rather than parameters, though every
    parameter here is also a tested ratio under the same name. The three tested ratios that are
    not parameters, elong/deg, init/elong and init/splice, are linear combinations of these.

    Which three of the six ratios carry the parametrization is a free choice, and these three are
    the ones the models build their predictions from directly. elong/splice is the whole of the
    nascent fraction's logit shift, which is the ratio the coverage shape is informative about, so
    `IntronCoverageModel` parametrizes that same ratio under this same name.
    """
    INIT_OVER_DEG = 'lfc_init_over_deg'
    ELONG_OVER_SPLICE = 'lfc_elong_over_splice'
    SPLICE_OVER_DEG = 'lfc_splice_over_deg'


class TestedRatio(StrEnum):
    """
    Which rate ratio an LRT restricts.

    All six ratios of the four rates are testable. Three of them are model parameters; the other
    three are linear combinations of those, which is well-defined because four rates admit only
    three independent ratios and the parametrized three span them.

    The values are the LFC attribute names, and are written verbatim to the `rate_ratio`
    column of the test results and on into the volcano plot paths. The three ratios that are not
    parameters have no attribute of their own, but are named the same way.
    """
    INIT_OVER_DEG = 'lfc_init_over_deg'
    ELONG_OVER_DEG = 'lfc_elong_over_deg'
    SPLICE_OVER_DEG = 'lfc_splice_over_deg'
    INIT_OVER_ELONG = 'lfc_init_over_elong'
    INIT_OVER_SPLICE = 'lfc_init_over_splice'
    ELONG_OVER_SPLICE = 'lfc_elong_over_splice'


# One rate ratio written in the model's parameters: (parameter, coefficient) pairs whose weighted
# sum is that ratio's LFC. A single pair of coefficient 1.0 is a ratio that is a parameter itself.
LFCCombination = tuple[tuple[str, float], ...]

# Each tested ratio as such a combination. Used to build the reduced model's hot start, to report
# the effect size and to assemble the Wald contrast, none of which can read a single parameter for
# the three ratios that are not parametrized.
#
# Which ratios take more than one term follows from the parametrization and not from the biology.
# With elong/splice and splice/deg parametrized, elong/deg is their sum, and init/elong is
# init/deg less both.
RATIO_AS_LFC_COMBINATION: dict[TestedRatio, LFCCombination] = {
    TestedRatio.INIT_OVER_DEG: ((LFCParameter.INIT_OVER_DEG, 1.0),),
    TestedRatio.ELONG_OVER_DEG: ((LFCParameter.ELONG_OVER_SPLICE, 1.0),
                                 (LFCParameter.SPLICE_OVER_DEG, 1.0)),
    TestedRatio.SPLICE_OVER_DEG: ((LFCParameter.SPLICE_OVER_DEG, 1.0),),
    TestedRatio.INIT_OVER_ELONG: ((LFCParameter.INIT_OVER_DEG, 1.0),
                                  (LFCParameter.ELONG_OVER_SPLICE, -1.0),
                                  (LFCParameter.SPLICE_OVER_DEG, -1.0)),
    TestedRatio.INIT_OVER_SPLICE: ((LFCParameter.INIT_OVER_DEG, 1.0),
                                   (LFCParameter.SPLICE_OVER_DEG, -1.0)),
    TestedRatio.ELONG_OVER_SPLICE: ((LFCParameter.ELONG_OVER_SPLICE, 1.0),),
}

# The global model's lfc_init_over_deg is per gene rather than shared, so no ratio involving
# initiation describes a single genome-wide effect and only these three can be tested there.
GLOBAL_MODEL_TESTED_RATIOS = (TestedRatio.ELONG_OVER_DEG,
                              TestedRatio.SPLICE_OVER_DEG,
                              TestedRatio.ELONG_OVER_SPLICE)


# A plug-in RNAKineticsModel takes lfc_elong_over_splice from a coverage fit instead of fitting
# it, so restricting that one ratio is no longer a restriction on anything the model is free to
# move. Its test is the coverage model's own LRT, which that fit already reports.
PLUG_IN_TESTABLE_RATIOS = tuple(tested_ratio for tested_ratio in TestedRatio
                                if tested_ratio != TestedRatio.ELONG_OVER_SPLICE)
GLOBAL_PLUG_IN_TESTABLE_RATIOS = tuple(tested_ratio for tested_ratio in GLOBAL_MODEL_TESTED_RATIOS
                                       if tested_ratio in PLUG_IN_TESTABLE_RATIOS)


class CoverageFitStatus(StrEnum):
    """
    Whether one intron's coverage fit can be used as a plug-in value.

    Single-valued, with `training_diverged` taking precedence over `not_estimable`, because the
    only question a consumer asks is whether the fit is usable. The individual diagnostics stay in
    `model_parameters.tsv`, which reports the divergence flag and the Wald columns separately.
    """
    OK = 'ok'
    TRAINING_DIVERGED = 'training_diverged'
    NOT_ESTIMABLE = 'not_estimable'


@dataclass
class PlugInCoverageFit:
    """
    One `IntronCoverageModel` fit, in the form a plug-in estimator consumes.

    A plug-in stage two freezes a whole coverage fit at once, so the variance correction needs the
    joint covariance of everything that fit estimated: one `lfc_elong_over_splice` together with
    the intercepts of every intron it was fitted over. The three granularities of the coverage
    model differ only in the length of `intron_names`, which is one intron, a gene's introns, or
    the whole dataset's.

    `lfc_elong_over_splice` is (num_features,), shared across `intron_names`, and
    `intercept_pi_logit` is (num_introns,). Both also appear in `model_parameters.tsv`; they are
    carried here so that the artifact stands on its own and a consumer needs no join against a
    published table.

    `covariance` is the inverse Fisher information over those same values, in that order, so it is
    square of side `num_features + num_introns`. It is the one quantity the TSVs have no column
    for, which is the reason this artifact exists: `model_parameters.tsv` reports only its
    diagonal, as `SE`. None when there is no covariance to report, which `status` then explains.
    """
    intron_names: list[str]
    lfc_elong_over_splice: torch.Tensor  # (num_features,)
    intercept_pi_logit: torch.Tensor  # (num_introns,)
    covariance: Optional[torch.Tensor]  # (num_features + num_introns) square, or None
    status: CoverageFitStatus


@dataclass
class PlugInCoverageFits:
    """
    The coverage fits of one chunk, pickled via `torch.save` for a later process.

    A wire format, so the field names cross a process boundary. Chunks merge by concatenating
    `fits`. `feature_names` describes the design, which every fit in the chunk shares and which is
    what makes the covariance axes readable on the far side; `intron_names` describes a single fit
    and so sits on it.

    `is_regularized` says these came from the penalized refit, whose fits carry no covariance: the
    Hessian of a penalized objective is not an information matrix.
    """
    fits: list[PlugInCoverageFit]
    feature_names: list[str]
    is_regularized: bool = False


class PlugInCoverage(NamedTuple):
    """
    The coverage fit's values laid out the way the model that plugs them in holds them.

    `lfc_elong_over_splice` is (num_features, num_introns) where the model's own LFC would be
    intron-specific and (num_features, 1) where it would be shared, and `intercept_pi_logit` is
    (num_introns,) in the model's intron order. Built by `from_fits`, which is where the fits'
    intron names are matched against the model's.
    """
    lfc_elong_over_splice: torch.Tensor
    intercept_pi_logit: torch.Tensor

    def check_shapes(self, lfc_shape: tuple[int, ...], num_introns: int) -> None:
        """Refuse a layout built for a different model than the one plugging it in."""
        if tuple(self.lfc_elong_over_splice.shape) != lfc_shape:
            raise ValueError(
                f'Plugged-in lfc_elong_over_splice has shape '
                f'{tuple(self.lfc_elong_over_splice.shape)}, expected {lfc_shape}.')
        if tuple(self.intercept_pi_logit.shape) != (num_introns,):
            raise ValueError(
                f'Plugged-in intercept_pi_logit has shape '
                f'{tuple(self.intercept_pi_logit.shape)}, expected {(num_introns,)}.')

    @staticmethod
    def from_fits(fits: list[PlugInCoverageFit],
                  intron_names: list[str],
                  intron_specific_lfc: bool) -> 'PlugInCoverage':
        """
        Lay out the coverage fits covering `intron_names`, in that order.

        An intron-specific LFC needs its own value per intron and so one fit per intron, while a
        shared one needs a single fit that covers them all. Which of the two is being built is
        read off `intron_specific_lfc` rather than off the number of fits, so a gene whose introns
        happen to number one cannot be served the wrong layout.

        Every intron of the model must appear in the fits, and a fit covering introns the model
        does not hold is accepted: a genome-wide coverage fit serves every gene.
        """
        fit_of_intron: dict[str, PlugInCoverageFit] = {}
        for fit in fits:
            for intron_name in fit.intron_names:
                fit_of_intron[intron_name] = fit

        missing = [intron_name for intron_name in intron_names if intron_name not in fit_of_intron]
        if missing:
            raise ValueError(f'No coverage fit covers introns {missing}.')

        intercept_pi_logit = torch.stack([
            fit_of_intron[intron_name].intercept_pi_logit[
                fit_of_intron[intron_name].intron_names.index(intron_name)]
            for intron_name in intron_names])

        if intron_specific_lfc:
            lfc_columns = [fit_of_intron[intron_name].lfc_elong_over_splice
                           for intron_name in intron_names]
            lfc_elong_over_splice = torch.stack(lfc_columns, dim=1)
        else:
            # Shared across the model's introns, so every one of them has to come from the same
            # fit; otherwise the values being plugged in are not a single shared LFC at all.
            distinct_fits = {id(fit_of_intron[intron_name]) for intron_name in intron_names}
            if len(distinct_fits) > 1:
                raise ValueError(
                    f'lfc_elong_over_splice is shared across {len(intron_names)} introns, so they '
                    f'must come from one coverage fit, but they come from {len(distinct_fits)}.')
            lfc_elong_over_splice = fit_of_intron[
                intron_names[0]].lfc_elong_over_splice.unsqueeze(1)

        return PlugInCoverage(lfc_elong_over_splice=lfc_elong_over_splice,
                              intercept_pi_logit=intercept_pi_logit)


class LRTSpecification(NamedTuple):
    num_features_reduced_matrix: int
    tested_ratio: TestedRatio
    tested_intron: Optional[str] = None


def _fit_poisson_glm(design_matrix: np.ndarray, y: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """
    Initialise lfc_init_over_deg via Poisson GLM (log link, statsmodels IRLS).
    design_matrix: (num_samples, num_features), y: (num_samples,) or (num_samples, num_genes),
    offset: same shape as y.
    Returns LFCs of shape (num_features,) or (num_genes, num_features).
    Falls back to zeros for any gene where the GLM fails (e.g. all-zero counts).
    """
    num_features = design_matrix.shape[1]
    if y.ndim == 1:
        try:
            return sm.GLM(y, design_matrix, family=sm.families.Poisson(), offset=offset).fit(disp=False).params
        except Exception:
            return np.zeros(num_features)
    num_genes = y.shape[1]
    lfc_init_over_deg = np.zeros((num_genes, num_features))
    for gene_index in range(num_genes):
        try:
            lfc_init_over_deg[gene_index] = sm.GLM(y[:, gene_index], design_matrix, family=sm.families.Poisson(),
                                                   offset=offset[:, gene_index]).fit(disp=False).params
        except Exception:
            pass
    return lfc_init_over_deg


# Parameters that carry no downstream meaning and are excluded from get_param_df().
# reduced_lfc only exists in LRT mode, where the reduced model's parameters are never reported.
UNREPORTED_PARAMETERS = frozenset({'reduced_lfc'})


def build_param_df(model: nn.Module, parameter_axes: dict[str, tuple[Optional[str], ...]]) -> pd.DataFrame:
    """
    Serialise model parameters into the long-format dataframe consumed downstream.

    `parameter_axes` maps each parameter attribute name to the axes it is indexed by, in order.
    An axis is 'gene', 'feature' or 'intron', or None for a singleton axis, which is indexed
    with 0 and reported with a null name. A `<axis>_name` column is emitted for every axis the
    model has names for, so per-gene models get no `gene_name` column.

    Row order is load-bearing: `add_wald_test_results` maps rows of the flattened Fisher
    information matrix onto rows of this dataframe by position. Rows are therefore emitted in
    `named_parameters()` order, each parameter flattened in C order, matching
    `build_hessian_matrix_from_dict`.
    """
    axis_to_names = {
        'gene': getattr(model, 'gene_names', None),
        'feature': getattr(model, 'feature_names', None),
        'intron': getattr(model, 'intron_names', None),
    }
    emitted_axes = [axis for axis, names in axis_to_names.items() if names is not None]

    parameter_names = [name for name, _ in model.named_parameters() if name not in UNREPORTED_PARAMETERS]
    undeclared = set(parameter_names) - set(parameter_axes)
    if undeclared:
        raise RuntimeError(f"Parameters missing from the axis declaration: {sorted(undeclared)}")

    parameter_data: list[dict] = []
    for param_name in parameter_names:
        param_value = getattr(model, param_name)
        axes = parameter_axes[param_name]
        index_ranges = [range(1) if axis is None else range(len(axis_to_names[axis])) for axis in axes]
        num_rows_before = len(parameter_data)
        for index_combination in itertools.product(*index_ranges):
            # parameter_type is the attribute name itself, which is also what reaches the
            # output TSVs and the volcano plot paths, so renaming a parameter is a schema change.
            row = {'parameter_type': param_name}
            row.update({f'{axis}_name': None for axis in emitted_axes})
            for axis, axis_index in zip(axes, index_combination):
                if axis is not None:
                    row[f'{axis}_name'] = axis_to_names[axis][axis_index]
            row['value'] = param_value[index_combination].item()
            parameter_data.append(row)
        num_rows_emitted = len(parameter_data) - num_rows_before
        if num_rows_emitted != param_value.numel():
            raise RuntimeError(
                f"Axis declaration for '{param_name}' produced {num_rows_emitted} rows "
                f"but the parameter has {param_value.numel()} elements.")

    return pd.DataFrame(data=parameter_data)


class CoverageLoss(nn.Module):

    def __init__(self, num_position_coverage: int):
        super().__init__()
        locations = torch.linspace(start=1 / (2 * num_position_coverage),
                                   end=1 - 1 / (2 * num_position_coverage),
                                   steps=num_position_coverage)
        location_term = 1 - 2 * locations
        self.register_buffer("location_term", location_term)

    def forward(self, pi, coverage):
        loss_per_location = -torch.log(1 + pi.unsqueeze(2) * self.location_term)
        return torch.sum(loss_per_location * coverage)

    def loss_for_pi_grid(self, candidate_pi: torch.Tensor, aggregated_coverage: torch.Tensor) -> torch.Tensor:
        """
        Compute coverage loss for a grid of candidate pi values.
        Args:
            candidate_pi: Tensor of shape (k,), candidate values of pi.
            aggregated_coverage: Tensor of shape (i, b), coverage summed over samples.
        Returns:
            Tensor of shape (k, i) containing the total loss for each candidate pi
            and each intron.
        """
        loss_terms = -torch.log(
            1 + candidate_pi[:, None] * self.location_term[None, :]
        )  # shape (k, b)
        return torch.einsum("kb,ib->ki", loss_terms, aggregated_coverage)


def estimate_initial_pi(coverage: torch.Tensor,
                        pi_eps: float = 0.01,
                        num_pi_grid_points: int = 20) -> torch.Tensor:
    """
    Grid-search the nascent fraction pi that best explains the observed intron coverage shape.
    coverage: Tensor of shape (num_samples, num_introns, num_coverage_bins).
    Returns a tensor of shape (num_introns,), with values confined to [pi_eps, 1 - pi_eps].
    """
    with torch.no_grad():
        aggregated_coverage = coverage.sum(dim=0)  # (num_introns, num_coverage_bins)
        pi_grid = torch.linspace(
            pi_eps,
            1 - pi_eps,
            num_pi_grid_points,
            device=aggregated_coverage.device,
            dtype=aggregated_coverage.dtype,
        )
        coverage_loss = CoverageLoss(
            num_position_coverage=aggregated_coverage.shape[1]
        ).to(aggregated_coverage.device)
        coverage_loss_grid = coverage_loss.loss_for_pi_grid(pi_grid, aggregated_coverage)
        return pi_grid[coverage_loss_grid.argmin(dim=0)]


def _get_predicted_pi_from_read_classes(intercept_pi_logit: torch.Tensor,
                                        lfc_term_nascent: torch.Tensor,
                                        lfc_term_unspliced: torch.Tensor,
                                        coverage_is_plugged_in: bool) -> Optional[torch.Tensor]:
    """
    The nascent fraction the two intron read classes imply, or None under the plug-in.

    For the two RNA kinetics models only, which is what taking the read-class terms as arguments
    says: pi is not an independent quantity there, and deriving it from the same two terms that
    build the counts is what keeps the coverage loss and the Poisson count loss fitting one pi
    under every null. `IntronCoverageModel` has no read classes and builds the same pi straight
    from its own two parameters. That the two routes give the same quantity is what a plug-in
    estimator rests on, so neither should be changed without the other.

    With the coverage plugged in the two terms differ by a frozen lfc_elong_over_splice, so pi is
    whatever the coverage fit said and carries no gradient. Nothing is returned then, which keeps
    the coverage loss out of the graph rather than leaving it to add a constant.
    """
    if coverage_is_plugged_in:
        return None
    return torch.sigmoid(intercept_pi_logit + lfc_term_nascent - lfc_term_unspliced)


def _link_unspliced_to_nascent(unspliced_term: torch.Tensor,
                               nascent_term: torch.Tensor,
                               elong_over_splice_term: torch.Tensor,
                               coverage_is_plugged_in: bool) -> torch.Tensor:
    """
    The unspliced read class's term, consistent with the nascent one under the plug-in.

    For the two RNA kinetics models only, read classes being theirs alone.

    A frozen lfc_elong_over_splice pins the two intron read classes to each other, so a reduced
    model may restrict only one of them and the other follows. The restrictions that rebuild the
    unspliced class already derive the nascent one from it and this returns their input unchanged;
    the ones that rebuild the nascent class leave the unspliced class at its unrestricted value,
    and this is where the restriction propagates back into it.

    Without the plug-in all three read classes are free to move independently, and rebuilding one
    of them says nothing about the others, so the term is returned as the branches built it.
    """
    if not coverage_is_plugged_in:
        return unspliced_term
    return nascent_term + elong_over_splice_term


class RNAKineticsModel(nn.Module):

    def __init__(self,
                 feature_names: list[str],
                 intron_names: list[str],
                 intron_specific_lfc: bool,
                 lrt_specification: Optional[LRTSpecification] = None,
                 plug_in_coverage: Optional[PlugInCoverage] = None
                 ):
        super().__init__()
        self.feature_names = feature_names
        self.intron_names = intron_names

        num_features = len(feature_names)
        num_introns = len(intron_names)
        lfc_intron_axis = num_introns if intron_specific_lfc else 1

        # Given a coverage fit, lfc_elong_over_splice and intercept_pi_logit are that fit's values
        # rather than anything this model estimates, so they become buffers. That alone is what
        # makes this a plug-in model: leaving named_parameters() takes them out of the gradient,
        # the Fisher information, the parameter table and the prior penalty at once, none of which
        # has to know the difference. Registration order is unchanged either way, because the
        # remaining parameters keep the row order the Fisher information indexes them by.
        self.coverage_is_plugged_in = plug_in_coverage is not None
        if plug_in_coverage is not None:
            plug_in_coverage.check_shapes((num_features, lfc_intron_axis), num_introns)

        self.lfc_init_over_deg = nn.Parameter(torch.zeros(num_features))
        self.intercept_exon = nn.Parameter(torch.zeros(1))

        if plug_in_coverage is None:
            self.lfc_elong_over_splice = nn.Parameter(torch.zeros(num_features, lfc_intron_axis))
        else:
            self.register_buffer('lfc_elong_over_splice',
                                 plug_in_coverage.lfc_elong_over_splice.clone())
        self.lfc_splice_over_deg = nn.Parameter(torch.zeros(num_features, lfc_intron_axis))
        self.intron_specific_lfc = intron_specific_lfc

        self.intercept_unspliced = nn.Parameter(torch.zeros(num_introns))
        if plug_in_coverage is None:
            self.intercept_pi_logit = nn.Parameter(torch.zeros(num_introns))
        else:
            self.register_buffer('intercept_pi_logit',
                                 plug_in_coverage.intercept_pi_logit.clone())

        self.lrt_specification = lrt_specification
        self.tested_intron_index: Optional[int] = None
        if lrt_specification is not None:
            if (self.coverage_is_plugged_in
                    and lrt_specification.tested_ratio not in PLUG_IN_TESTABLE_RATIOS):
                raise ValueError(
                    f'A plug-in RNAKineticsModel LRT supports only {PLUG_IN_TESTABLE_RATIOS}, '
                    f'got {lrt_specification.tested_ratio!r}: that ratio comes from the coverage '
                    f'fit rather than being estimated here, so restricting it restricts nothing '
                    f'the model is free to move, and its test is the coverage model\'s own.')
            self.reduced_lfc = nn.Parameter(torch.zeros(lrt_specification.num_features_reduced_matrix))
            # Every ratio except init/deg involves elongation or splicing, whose LFCs can be
            # intron-specific, so those are tested one intron at a time. init/deg involves only
            # lfc_init_over_deg, which is shared across the gene, so it has no tested intron.
            tests_single_intron = (intron_specific_lfc
                                   and lrt_specification.tested_ratio != TestedRatio.INIT_OVER_DEG)
            if tests_single_intron:
                self.tested_intron_index = self.intron_names.index(lrt_specification.tested_intron)

    def initialize_parameters(self,
                              gene_data: GeneData,
                              library_sizes: torch.Tensor,
                              design_matrix: torch.Tensor,
                              pi_eps: float = 0.01,
                              num_pi_grid_points: int = 20) -> None:
        with torch.no_grad():
            intercept_exon_scalar = torch.log(gene_data.exon_reads.mean() / library_sizes.mean())
            self.intercept_exon.data[:] = intercept_exon_scalar  # preserve shape [1]

            offset = (intercept_exon_scalar + torch.log(library_sizes) + gene_data.isoform_length_offset).cpu().numpy()
            glm_lfc = _fit_poisson_glm(design_matrix.cpu().numpy(), gene_data.exon_reads.cpu().numpy(), offset)
            self.lfc_init_over_deg.data.copy_(torch.from_numpy(glm_lfc).to(self.lfc_init_over_deg.dtype))

            # The unspliced intercept has to sit under the pi the model will actually use,
            # since the two intron read classes sum to the observed counts only at that pi. A
            # plugged-in intercept_pi_logit is therefore read rather than grid searched, and
            # log(1 - pi) is taken as -softplus(logit) so that a saturated one stays finite: at
            # logit 1055 softplus returns 1055, while log1p(-sigmoid(1055)) is -inf.
            mean_intron_reads = gene_data.intron_reads.mean(dim=0) / library_sizes.mean()
            if self.coverage_is_plugged_in:
                self.intercept_unspliced.data.copy_(
                    torch.log(mean_intron_reads)
                    - functional.softplus(self.intercept_pi_logit))
            else:
                best_pi = estimate_initial_pi(gene_data.coverage, pi_eps, num_pi_grid_points)
                self.intercept_pi_logit.data.copy_(torch.logit(best_pi, eps=pi_eps))
                self.intercept_unspliced.data.copy_(torch.log(mean_intron_reads * (1 - best_pi)))

    def get_lfc_terms_per_read_class(self,
                                     design_matrix: torch.Tensor,
                                     reduced_design_matrix: Optional[torch.Tensor]
                                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        LFC term of each read class the model predicts. Unrestricted, the three are

            exon      = X @ lfc_init_over_deg
            unspliced = X @ lfc_init_over_deg - X @ lfc_splice_over_deg
            nascent   = unspliced - X @ lfc_elong_over_splice

        Writing the nascent class as a shift of the unspliced one is what this parametrization
        buys: that shift is the entirety of the nascent fraction's logit (see `forward`), so it is
        the same quantity `IntronCoverageModel` reads off the coverage shape, under the same name.
        It also matches how the intercepts are built, where the unspliced class is the base and the
        nascent one is `intercept_pi_logit` above it.

        Every LRT is a single substitution here: the term of the rate ratio under test is built
        from the reduced design matrix, while the other read classes keep the full one. Which ratio
        is tested therefore decides which read class gets substituted.

        The exon term is (num_samples,); the intron terms are (num_samples, num_introns), or
        (num_samples, 1) broadcasting over introns when the LFCs are shared across them.
        """
        init_over_deg_term = design_matrix @ self.lfc_init_over_deg
        elong_over_splice_term = design_matrix @ self.lfc_elong_over_splice
        splice_over_deg_term = design_matrix @ self.lfc_splice_over_deg
        init_over_deg_per_intron = init_over_deg_term.unsqueeze(1)

        if self.lrt_specification is None:
            unspliced_term = init_over_deg_per_intron - splice_over_deg_term
            return (init_over_deg_term,
                    unspliced_term - elong_over_splice_term,
                    unspliced_term)

        if reduced_design_matrix is None:
            raise ValueError(
                "reduced_design_matrix must be provided in the LRT mode "
                "(i.e. when lrt_specification is not None).")
        # Stands in for whichever LFC is under test, so below it is used in exactly the position
        # that LFC occupies in the unrestricted terms.
        reduced_lfc_term = reduced_design_matrix @ self.reduced_lfc  # (num_samples,)
        reduced_per_intron = reduced_lfc_term.unsqueeze(1)
        tested_ratio = self.lrt_specification.tested_ratio

        if tested_ratio == TestedRatio.INIT_OVER_DEG:
            # init/deg feeds all three read classes and is never intron-specific, so all three
            # are rebuilt. It enters each of them positively.
            unspliced_term = reduced_per_intron - splice_over_deg_term
            return (reduced_lfc_term,
                    unspliced_term - elong_over_splice_term,
                    unspliced_term)

        unspliced_term = init_over_deg_per_intron - splice_over_deg_term

        # Each remaining ratio is restricted by rebuilding one intron read class from the reduced
        # term, in the place that ratio occupies:
        #   splice/deg   = init - unspliced     so the unspliced class becomes init - reduced
        #   init/splice  = unspliced            so the unspliced class becomes the reduced term
        #   elong/splice = unspliced - nascent  so the nascent class becomes unspliced - reduced
        #   elong/deg    = init - nascent       so the nascent class becomes init - reduced
        #   init/elong   = nascent              so the nascent class becomes the reduced term
        # The first two restrict splice/deg, which the nascent class carries too, so there the
        # nascent term is built from the restricted unspliced term rather than from the full one.
        if tested_ratio == TestedRatio.SPLICE_OVER_DEG:
            unspliced_term = self._get_restricted_intron_term(
                unspliced_term, init_over_deg_per_intron - reduced_per_intron)
            nascent_term = unspliced_term - elong_over_splice_term
        elif tested_ratio == TestedRatio.INIT_OVER_SPLICE:
            unspliced_term = self._get_restricted_intron_term(unspliced_term, reduced_per_intron)
            nascent_term = unspliced_term - elong_over_splice_term
        elif tested_ratio == TestedRatio.ELONG_OVER_SPLICE:
            nascent_term = self._get_restricted_intron_term(
                unspliced_term - elong_over_splice_term,
                self._get_tested_intron_column(unspliced_term) - reduced_per_intron)
        elif tested_ratio == TestedRatio.ELONG_OVER_DEG:
            nascent_term = self._get_restricted_intron_term(
                unspliced_term - elong_over_splice_term,
                init_over_deg_per_intron - reduced_per_intron)
        elif tested_ratio == TestedRatio.INIT_OVER_ELONG:
            nascent_term = self._get_restricted_intron_term(
                unspliced_term - elong_over_splice_term, reduced_per_intron)
        else:
            raise ValueError(f'Unhandled tested ratio: {tested_ratio!r}')

        unspliced_term = _link_unspliced_to_nascent(unspliced_term, nascent_term,
                                                    elong_over_splice_term,
                                                    self.coverage_is_plugged_in)
        return init_over_deg_term, nascent_term, unspliced_term

    def _get_tested_intron_column(self, full_term: torch.Tensor) -> torch.Tensor:
        """
        The tested intron's column of an intron-axis term, kept two-dimensional. With LFCs shared
        across introns there is only one column, which is the term itself.
        """
        if not self.intron_specific_lfc:
            return full_term
        intron_index = self.tested_intron_index
        return full_term[:, intron_index:intron_index + 1]

    def _get_restricted_intron_term(self, full_term: torch.Tensor,
                                    restricted_term: torch.Tensor) -> torch.Tensor:
        """
        One intron read class's term, under the restriction.

        With intron-specific LFCs only the tested intron is restricted, so its column comes from
        restricted_term while every other intron keeps its full value. With LFCs shared across
        introns there is a single column, which restricted_term replaces outright. Built with
        torch.cat rather than an in-place write into a matmul output, so autograd's
        version-counter semantics stay off the critical path.
        """
        if not self.intron_specific_lfc:
            return restricted_term
        intron_index = self.tested_intron_index
        return torch.cat([full_term[:, :intron_index],
                          restricted_term,
                          full_term[:, intron_index + 1:]], dim=1)

    def forward(self,
                design_matrix: torch.Tensor,
                log_library_sizes: torch.Tensor,
                isoform_length_offset: torch.Tensor,
                reduced_design_matrix: Optional[torch.Tensor] = None):
        lfc_term_exon, lfc_term_nascent, lfc_term_unspliced = self.get_lfc_terms_per_read_class(
            design_matrix, reduced_design_matrix)

        predicted_log_reads_exon = (self.intercept_exon + log_library_sizes
                                    + isoform_length_offset + lfc_term_exon)

        predicted_pi = _get_predicted_pi_from_read_classes(
            self.intercept_pi_logit, lfc_term_nascent, lfc_term_unspliced,
            self.coverage_is_plugged_in)

        # The two intron read classes share the library-size offset and differ in their
        # intercept. Naming the nascent intercept makes intercept_pi_logit's role explicit: it is
        # how far the nascent baseline sits above the unspliced one, i.e. the baseline logit of
        # the nascent fraction.
        log_library_sizes_per_intron = log_library_sizes.unsqueeze(1)
        intercept_nascent = self.intercept_unspliced + self.intercept_pi_logit

        predicted_reads_intron_nascent = safe_exp(
            intercept_nascent + log_library_sizes_per_intron + lfc_term_nascent)
        predicted_reads_intron_unspliced = safe_exp(
            self.intercept_unspliced + log_library_sizes_per_intron + lfc_term_unspliced)
        predicted_reads_intron = predicted_reads_intron_nascent + predicted_reads_intron_unspliced

        return safe_exp(predicted_log_reads_exon), predicted_reads_intron, predicted_pi

    def get_param_df(self) -> pd.DataFrame:
        lfc_intron_axis = 'intron' if self.intron_specific_lfc else None
        return build_param_df(self, {
            'intercept_exon': (None,),
            'lfc_init_over_deg': ('feature',),
            'lfc_elong_over_splice': ('feature', lfc_intron_axis),
            'lfc_splice_over_deg': ('feature', lfc_intron_axis),
            'intercept_unspliced': ('intron',),
            'intercept_pi_logit': ('intron',),
        })


class IntronCoverageModel(nn.Module):

    def __init__(self,
                 feature_names: list[str],
                 intron_names: list[str]):
        super().__init__()
        self.feature_names = feature_names
        self.intron_names = intron_names

        num_features = len(feature_names)
        num_introns = len(intron_names)

        # One LFC, shared across whatever introns the model was given. The second axis is a
        # singleton so that the term broadcasts against the per-intron intercept; it is never an
        # intron axis. Fitting this model per intron means constructing it with that one intron,
        # and then its LFC is reported under that intron's name -- see get_param_df.
        #
        # The name is shared with `RNAKineticsModel`'s parameter of the same ratio deliberately:
        # that model's nascent fraction responds to exactly this LFC, so the two estimate the same
        # quantity. The coverage term of the likelihood is the same in both; this model has only
        # that term, while `RNAKineticsModel` has the read counts on top of it.
        self.lfc_elong_over_splice = nn.Parameter(torch.zeros(num_features, 1))
        self.intercept_pi_logit = nn.Parameter(torch.zeros(num_introns))

    def initialize_parameters(self,
                              coverage: torch.Tensor,
                              pi_eps: float = 0.01,
                              num_pi_grid_points: int = 20) -> None:
        best_pi = estimate_initial_pi(coverage, pi_eps, num_pi_grid_points)
        self.intercept_pi_logit.data.copy_(torch.logit(best_pi, eps=pi_eps))

    def forward(self, design_matrix: torch.Tensor):
        elong_over_splice_term = design_matrix @ self.lfc_elong_over_splice
        pi = torch.sigmoid(self.intercept_pi_logit - elong_over_splice_term)
        return pi

    def get_param_df(self) -> pd.DataFrame:
        # Given a single intron, the shared LFC is that intron's LFC, so it is reported under its
        # name. Given several, it is shared across them and reported unnamed.
        lfc_intron_axis = 'intron' if len(self.intron_names) == 1 else None
        return build_param_df(self, {
            'lfc_elong_over_splice': ('feature', lfc_intron_axis),
            'intercept_pi_logit': ('intron',),
        })


class GlobalRNAKineticsModel(nn.Module):

    def __init__(self,
                 feature_names: list[str],
                 gene_names: list[str],
                 intron_names: list[str],
                 gene_idx: torch.Tensor,
                 lrt_specification: Optional[LRTSpecification] = None,
                 plug_in_coverage: Optional[PlugInCoverage] = None,
                 ):
        super().__init__()
        self.feature_names = feature_names
        self.gene_names = gene_names
        self.intron_names = intron_names

        num_features = len(feature_names)
        num_genes = len(gene_names)
        num_introns = len(intron_names)

        # As in RNAKineticsModel: a coverage fit turns these two into buffers, and that is the
        # whole of what makes the model a plug-in one. The LFC is shared across the dataset here,
        # so it has no intron axis to lay out and is stored as (num_features,).
        self.coverage_is_plugged_in = plug_in_coverage is not None
        if plug_in_coverage is not None:
            plug_in_coverage.check_shapes((num_features, 1), num_introns)

        self.lfc_init_over_deg = nn.Parameter(torch.zeros(num_genes, num_features))
        if plug_in_coverage is None:
            self.lfc_elong_over_splice = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_buffer('lfc_elong_over_splice',
                                 plug_in_coverage.lfc_elong_over_splice[:, 0].clone())
        self.lfc_splice_over_deg = nn.Parameter(torch.zeros(num_features))
        self.intercept_exon = nn.Parameter(torch.zeros(num_genes))
        self.intercept_unspliced = nn.Parameter(torch.zeros(num_introns))
        if plug_in_coverage is None:
            self.intercept_pi_logit = nn.Parameter(torch.zeros(num_introns))
        else:
            self.register_buffer('intercept_pi_logit',
                                 plug_in_coverage.intercept_pi_logit.clone())

        self.register_buffer('gene_idx', gene_idx)

        self.lrt_specification = lrt_specification
        if lrt_specification is not None:
            # A plugged-in lfc_elong_over_splice leaves two of the three testable, its own test
            # being the coverage model's.
            testable_ratios = (GLOBAL_PLUG_IN_TESTABLE_RATIOS if self.coverage_is_plugged_in
                               else GLOBAL_MODEL_TESTED_RATIOS)
            if lrt_specification.tested_ratio not in testable_ratios:
                raise ValueError(
                    f"GlobalRNAKineticsModel LRT supports only {testable_ratios}, "
                    f"got {lrt_specification.tested_ratio!r}.")
            self.reduced_lfc = nn.Parameter(torch.zeros(lrt_specification.num_features_reduced_matrix))

    def initialize_parameters(self,
                              global_gene_data: GlobalGeneData,
                              library_sizes: torch.Tensor,
                              design_matrix: torch.Tensor,
                              pi_eps: float = 0.01,
                              num_pi_grid_points: int = 20) -> None:
        with torch.no_grad():
            self.intercept_exon.data.copy_(
                torch.log(global_gene_data.exon_reads.mean(dim=0) / library_sizes.mean())
            )

            log_lib = torch.log(library_sizes)
            offset = (self.intercept_exon.unsqueeze(0) + log_lib.unsqueeze(1)
                      + global_gene_data.isoform_length_offset).cpu().numpy()  # (num_samples, num_genes)
            glm_lfc = _fit_poisson_glm(
                design_matrix.cpu().numpy(), global_gene_data.exon_reads.cpu().numpy(), offset
            )  # (num_genes, num_features)
            self.lfc_init_over_deg.data.copy_(torch.from_numpy(glm_lfc).to(self.lfc_init_over_deg.dtype))

            # See RNAKineticsModel.initialize_parameters: the unspliced intercept sits under
            # whichever pi the model will use, and -softplus(logit) is log(1 - pi) written so that
            # a saturated plugged-in logit stays finite.
            mean_intron_reads = global_gene_data.intron_reads.mean(dim=0) / library_sizes.mean()
            if self.coverage_is_plugged_in:
                self.intercept_unspliced.data.copy_(
                    torch.log(mean_intron_reads) - functional.softplus(self.intercept_pi_logit))
            else:
                best_pi = estimate_initial_pi(global_gene_data.coverage, pi_eps,
                                              num_pi_grid_points)
                self.intercept_pi_logit.data.copy_(torch.logit(best_pi, eps=pi_eps))
                self.intercept_unspliced.data.copy_(torch.log(mean_intron_reads * (1 - best_pi)))

    def get_lfc_terms_per_read_class(self,
                                     design_matrix: torch.Tensor,
                                     reduced_design_matrix: Optional[torch.Tensor]
                                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        LFC term of each read class the model predicts, shaped (num_samples, num_genes),
        (num_samples, num_introns) and (num_samples, num_introns).

        Same three terms as RNAKineticsModel, except that lfc_init_over_deg is per gene here while
        the other two are shared, so the exon class is indexed by gene and the intron classes are
        expanded through gene_idx.
        """
        init_over_deg_term = design_matrix @ self.lfc_init_over_deg.T  # (num_samples, num_genes)
        init_over_deg_per_intron = init_over_deg_term[:, self.gene_idx]  # (num_samples, num_introns)

        elong_over_splice_term = (design_matrix @ self.lfc_elong_over_splice).unsqueeze(1)
        unspliced_term = (init_over_deg_per_intron
                          - (design_matrix @ self.lfc_splice_over_deg).unsqueeze(1))
        nascent_term = unspliced_term - elong_over_splice_term

        if self.lrt_specification is not None:
            if reduced_design_matrix is None:
                raise ValueError("reduced_design_matrix must be provided in LRT mode.")
            # Stands in for whichever LFC is under test, used in the place that ratio occupies:
            #   elong/deg    = init - nascent          so nascent becomes init - reduced
            #   splice/deg   = init - unspliced        so unspliced becomes init - reduced
            #   elong/splice = unspliced - nascent     so nascent becomes unspliced - reduced
            # splice/deg is carried by the nascent class too, so there the nascent term is
            # rebuilt from the restricted unspliced term.
            reduced_lfc_term = (reduced_design_matrix @ self.reduced_lfc).unsqueeze(1)
            tested_ratio = self.lrt_specification.tested_ratio
            if tested_ratio == TestedRatio.ELONG_OVER_DEG:
                nascent_term = init_over_deg_per_intron - reduced_lfc_term
            elif tested_ratio == TestedRatio.SPLICE_OVER_DEG:
                unspliced_term = init_over_deg_per_intron - reduced_lfc_term
                nascent_term = unspliced_term - elong_over_splice_term
            elif tested_ratio == TestedRatio.ELONG_OVER_SPLICE:
                nascent_term = unspliced_term - reduced_lfc_term
            else:
                raise ValueError(f'Unhandled tested ratio: {tested_ratio!r}')

            unspliced_term = _link_unspliced_to_nascent(unspliced_term, nascent_term,
                                                        elong_over_splice_term,
                                                        self.coverage_is_plugged_in)

        return init_over_deg_term, nascent_term, unspliced_term

    def forward(self,
                design_matrix: torch.Tensor,
                log_library_sizes: torch.Tensor,
                isoform_length_offset: torch.Tensor,
                reduced_design_matrix: Optional[torch.Tensor] = None):
        lfc_term_exon, lfc_term_nascent, lfc_term_unspliced = self.get_lfc_terms_per_read_class(
            design_matrix, reduced_design_matrix)

        predicted_log_reads_exon = (
                self.intercept_exon
                + log_library_sizes.unsqueeze(1)
                + isoform_length_offset
                + lfc_term_exon
        )

        predicted_pi = _get_predicted_pi_from_read_classes(
            self.intercept_pi_logit, lfc_term_nascent, lfc_term_unspliced,
            self.coverage_is_plugged_in)

        # The two intron read classes share the library-size offset and differ in their
        # intercept. Naming the nascent intercept makes intercept_pi_logit's role explicit: it is
        # how far the nascent baseline sits above the unspliced one, i.e. the baseline logit of
        # the nascent fraction.
        log_library_sizes_per_intron = log_library_sizes.unsqueeze(1)
        intercept_nascent = self.intercept_unspliced + self.intercept_pi_logit

        predicted_reads_intron_nascent = safe_exp(
            intercept_nascent + log_library_sizes_per_intron + lfc_term_nascent)
        predicted_reads_intron_unspliced = safe_exp(
            self.intercept_unspliced + log_library_sizes_per_intron + lfc_term_unspliced)
        predicted_reads_intron = predicted_reads_intron_nascent + predicted_reads_intron_unspliced

        return safe_exp(predicted_log_reads_exon), predicted_reads_intron, predicted_pi

    def get_param_df(self) -> pd.DataFrame:
        return build_param_df(self, {
            'intercept_exon': ('gene',),
            'lfc_init_over_deg': ('gene', 'feature'),
            'lfc_elong_over_splice': ('feature',),
            'lfc_splice_over_deg': ('feature',),
            'intercept_unspliced': ('intron',),
            'intercept_pi_logit': ('intron',),
        })


class RNAKineticsLoss(nn.Module):
    def __init__(self, num_position_coverage: int):
        super().__init__()
        self.loss_function_exon_counts = nn.PoissonNLLLoss(log_input=False, full=True, reduction='sum')
        self.loss_function_intron_counts = nn.PoissonNLLLoss(log_input=False, full=True, reduction='sum')
        self.loss_function_intron_coverage = CoverageLoss(num_position_coverage=num_position_coverage)

    def forward(self,
                reads_exon: torch.Tensor,
                reads_intron: torch.Tensor,
                intron_coverage: torch.Tensor,
                predicted_reads_exon: torch.Tensor,
                predicted_reads_intron: torch.Tensor,
                predicted_pi: Optional[torch.Tensor]):
        """
        The three terms of the likelihood, summed.

        A plug-in model predicts no pi, having taken it from a coverage fit, and then the coverage
        term would be the same constant under every hypothesis. It is left out rather than added,
        so that the reported loss is the objective that was optimized.
        """
        loss_exon_counts = self.loss_function_exon_counts(predicted_reads_exon,
                                                          reads_exon)
        loss_intron_counts = self.loss_function_intron_counts(predicted_reads_intron, reads_intron)

        total_loss = loss_exon_counts + loss_intron_counts
        if predicted_pi is not None:
            total_loss = total_loss + self.loss_function_intron_coverage(predicted_pi,
                                                                         intron_coverage)
        return total_loss
