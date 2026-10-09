import itertools
from enum import StrEnum
from typing import Optional, NamedTuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import torch.nn as nn

from rna_kinetics.data import GeneData, GlobalGeneData


def safe_exp(x: torch.Tensor, output_threshold: float = 1e20) -> torch.Tensor:
    output_threshold_tensor = torch.tensor(output_threshold, dtype=x.dtype, device=x.device)
    input_threshold = torch.log(output_threshold_tensor)
    return torch.where(
        x <= input_threshold,
        torch.exp(x),
        output_threshold_tensor + output_threshold_tensor * (x - input_threshold)
    )


class TestableParameters(StrEnum):
    ALPHA = 'alpha'
    BETA = 'beta'
    GAMMA = 'gamma'


# Stable output vocabulary: the values written to the `parameter_type` column, which
# `merge_regularization.R` joins against `tested_parameter`. Deliberately decoupled from the
# attribute names, so those can be renamed without touching any R script or output schema.
# TODO: 'theta' is a leftover name for intercept_pi_logit; renaming it is a schema change and
# belongs with the LRT refactor, when the R side is being touched anyway.
PARAMETER_WIRE_NAMES = {
    'intercept_exon': 'intercept_exon',
    'lfc_init_over_deg': 'alpha',
    'lfc_elong_over_deg': 'beta',
    'lfc_splice_over_deg': 'gamma',
    'lfc_elong_over_splice': 'lfc',
    'intercept_unspliced': 'intercept_unspliced',
    'intercept_pi_logit': 'theta',
}

TESTED_PARAMETER_TO_ATTRIBUTE = {
    TestableParameters.ALPHA: 'lfc_init_over_deg',
    TestableParameters.BETA: 'lfc_elong_over_deg',
    TestableParameters.GAMMA: 'lfc_splice_over_deg',
}


class LRTSpecification(NamedTuple):
    num_features_reduced_matrix: int
    tested_parameter: TestableParameters
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
    `flatten_hessian_dict`.
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
            row = {'parameter_type': PARAMETER_WIRE_NAMES[param_name]}
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


class RNAKineticsModel(nn.Module):

    def __init__(self,
                 feature_names: list[str],
                 intron_names: list[str],
                 intron_specific_lfc: bool,
                 lrt_specification: Optional[LRTSpecification] = None
                 ):
        super().__init__()
        self.feature_names = feature_names
        self.intron_names = intron_names

        num_features = len(feature_names)
        num_introns = len(intron_names)

        self.lfc_init_over_deg = nn.Parameter(torch.zeros(num_features))
        self.intercept_exon = nn.Parameter(torch.zeros(1))

        if intron_specific_lfc:
            self.lfc_elong_over_deg = nn.Parameter(torch.zeros(num_features, num_introns))
            self.lfc_splice_over_deg = nn.Parameter(torch.zeros(num_features, num_introns))
        else:
            self.lfc_elong_over_deg = nn.Parameter(torch.zeros(num_features, 1))
            self.lfc_splice_over_deg = nn.Parameter(torch.zeros(num_features, 1))
        self.intron_specific_lfc = intron_specific_lfc

        self.intercept_unspliced = nn.Parameter(torch.zeros(num_introns))
        self.intercept_pi_logit = nn.Parameter(torch.zeros(num_introns))

        self.lrt_specification = lrt_specification
        self.tested_intron_index: Optional[int] = None
        if lrt_specification is not None:
            self.reduced_lfc = nn.Parameter(torch.zeros(lrt_specification.num_features_reduced_matrix))
            # Only intron-specific LFC parameters (elongation, splicing) are tested per intron;
            # lfc_init_over_deg is shared across the gene, so it has no tested intron.
            tests_single_intron = (intron_specific_lfc
                                   and lrt_specification.tested_parameter != TestableParameters.ALPHA)
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

            best_pi = estimate_initial_pi(gene_data.coverage, pi_eps, num_pi_grid_points)
            self.intercept_pi_logit.data.copy_(torch.logit(best_pi, eps=pi_eps))

            intercept_unspliced_vector = torch.log(
                gene_data.intron_reads.mean(dim=0) / library_sizes.mean() * (1 - best_pi))
            self.intercept_unspliced.data.copy_(intercept_unspliced_vector)

    def _get_lfc_terms_per_read_class(self,
                                      design_matrix: torch.Tensor,
                                      reduced_design_matrix: Optional[torch.Tensor]
                                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        LFC term of each read class the model predicts. Unrestricted, the three are

            exon      = X @ alpha
            nascent   = X @ (alpha - beta)
            unspliced = X @ (alpha - gamma)

        and every LRT is a single substitution here: the term of the rate ratio under test is
        built from the reduced design matrix, while the other read classes keep the full one.
        Which ratio is tested therefore decides which read class gets substituted.

        The exon term is (num_samples,); the intron terms are (num_samples, num_introns), or
        (num_samples, 1) broadcasting over introns when the LFCs are shared across them.
        """
        init_over_deg_term = design_matrix @ self.lfc_init_over_deg
        elong_over_deg_term = design_matrix @ self.lfc_elong_over_deg
        splice_over_deg_term = design_matrix @ self.lfc_splice_over_deg

        if self.lrt_specification is None:
            return (init_over_deg_term,
                    init_over_deg_term.unsqueeze(1) - elong_over_deg_term,
                    init_over_deg_term.unsqueeze(1) - splice_over_deg_term)

        if reduced_design_matrix is None:
            raise ValueError(
                "reduced_design_matrix must be provided in the LRT mode "
                "(i.e. when lrt_specification is not None).")
        # Stands in for whichever LFC is under test, so below it is used in exactly the position
        # that LFC occupies in the unrestricted terms.
        reduced_lfc_term = reduced_design_matrix @ self.reduced_lfc  # (num_samples,)
        tested_parameter = self.lrt_specification.tested_parameter

        if tested_parameter == TestableParameters.ALPHA:
            # alpha feeds all three read classes and is never intron-specific, so all three are
            # rebuilt. It enters each of them positively.
            reduced_init_per_intron = reduced_lfc_term.unsqueeze(1)
            return (reduced_lfc_term,
                    reduced_init_per_intron - elong_over_deg_term,
                    reduced_init_per_intron - splice_over_deg_term)

        init_over_deg_per_intron = init_over_deg_term.unsqueeze(1)
        nascent_term = init_over_deg_per_intron - elong_over_deg_term
        unspliced_term = init_over_deg_per_intron - splice_over_deg_term

        # beta and gamma are each subtracted from the initiation term in their own read class, so
        # the restricted read class is that same expression with the reduced term in their place.
        if tested_parameter == TestableParameters.BETA:
            nascent_term = self._get_restricted_intron_term(
                nascent_term, init_over_deg_per_intron - reduced_lfc_term.unsqueeze(1))
        elif tested_parameter == TestableParameters.GAMMA:
            unspliced_term = self._get_restricted_intron_term(
                unspliced_term, init_over_deg_per_intron - reduced_lfc_term.unsqueeze(1))
        else:
            raise ValueError(f'Unhandled tested parameter: {tested_parameter!r}')

        return init_over_deg_term, nascent_term, unspliced_term

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
        lfc_term_exon, lfc_term_nascent, lfc_term_unspliced = self._get_lfc_terms_per_read_class(
            design_matrix, reduced_design_matrix)

        predicted_log_reads_exon = (self.intercept_exon + log_library_sizes
                                    + isoform_length_offset + lfc_term_exon)

        # pi is not an independent quantity: it is the nascent fraction implied by the two intron
        # read classes. Deriving it from the same two terms that build the counts is what keeps
        # the coverage loss and the Poisson count loss fitting one pi under every null.
        predicted_pi = torch.sigmoid(
            self.intercept_pi_logit + lfc_term_nascent - lfc_term_unspliced)

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
            'lfc_elong_over_deg': ('feature', lfc_intron_axis),
            'lfc_splice_over_deg': ('feature', lfc_intron_axis),
            'intercept_unspliced': ('intron',),
            'intercept_pi_logit': ('intron',),
        })


class IntronCoverageModel(nn.Module):

    def __init__(self,
                 feature_names: list[str],
                 intron_names: list[str],
                 lfc_is_intron_specific: bool = False):
        super().__init__()
        self.feature_names = feature_names
        self.intron_names = intron_names

        num_features = len(feature_names)
        num_introns = len(intron_names)

        # The model always has a single LFC, shared across whatever introns it was given.
        # When it is fitted to one intron at a time, that LFC belongs to that intron and is
        # reported under its name; when fitted per gene, it is shared and reported unnamed.
        if lfc_is_intron_specific and num_introns != 1:
            raise ValueError(
                f"lfc_is_intron_specific requires exactly one intron, got {num_introns}: {intron_names}.")
        self.lfc_is_intron_specific = lfc_is_intron_specific

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
        return build_param_df(self, {
            'lfc_elong_over_splice': ('feature', 'intron' if self.lfc_is_intron_specific else None),
            'intercept_pi_logit': ('intron',),
        })


class GlobalRNAKineticsModel(nn.Module):

    def __init__(self,
                 feature_names: list[str],
                 gene_names: list[str],
                 intron_names: list[str],
                 gene_idx: torch.Tensor,
                 lrt_specification: Optional[LRTSpecification] = None,
                 ):
        super().__init__()
        self.feature_names = feature_names
        self.gene_names = gene_names
        self.intron_names = intron_names

        num_features = len(feature_names)
        num_genes = len(gene_names)
        num_introns = len(intron_names)

        self.lfc_init_over_deg = nn.Parameter(torch.zeros(num_genes, num_features))
        self.lfc_elong_over_deg = nn.Parameter(torch.zeros(num_features))
        self.lfc_splice_over_deg = nn.Parameter(torch.zeros(num_features))
        self.intercept_exon = nn.Parameter(torch.zeros(num_genes))
        self.intercept_unspliced = nn.Parameter(torch.zeros(num_introns))
        self.intercept_pi_logit = nn.Parameter(torch.zeros(num_introns))

        self.register_buffer('gene_idx', gene_idx)

        self.lrt_specification = lrt_specification
        if lrt_specification is not None:
            if lrt_specification.tested_parameter not in (TestableParameters.BETA, TestableParameters.GAMMA):
                raise ValueError(
                    f"GlobalRNAKineticsModel LRT only supports BETA and GAMMA, "
                    f"got {lrt_specification.tested_parameter!r}.")
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

            best_pi = estimate_initial_pi(global_gene_data.coverage, pi_eps, num_pi_grid_points)
            self.intercept_pi_logit.data.copy_(torch.logit(best_pi, eps=pi_eps))

            self.intercept_unspliced.data.copy_(
                torch.log(global_gene_data.intron_reads.mean(dim=0) / library_sizes.mean() * (1 - best_pi))
            )

    def _get_lfc_terms_per_read_class(self,
                                      design_matrix: torch.Tensor,
                                      reduced_design_matrix: Optional[torch.Tensor]
                                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        LFC term of each read class the model predicts, shaped (num_samples, num_genes),
        (num_samples, num_introns) and (num_samples, num_introns).

        Same three terms as RNAKineticsModel (exon = X @ alpha, nascent = X @ (alpha - beta),
        unspliced = X @ (alpha - gamma)), except that alpha is per gene here while beta and gamma
        are shared, so the exon class is indexed by gene and the intron classes are expanded
        through gene_idx.
        """
        init_over_deg_term = design_matrix @ self.lfc_init_over_deg.T  # (num_samples, num_genes)
        init_over_deg_per_intron = init_over_deg_term[:, self.gene_idx]  # (num_samples, num_introns)

        nascent_term = (init_over_deg_per_intron
                        - (design_matrix @ self.lfc_elong_over_deg).unsqueeze(1))
        unspliced_term = (init_over_deg_per_intron
                          - (design_matrix @ self.lfc_splice_over_deg).unsqueeze(1))

        if self.lrt_specification is not None:
            if reduced_design_matrix is None:
                raise ValueError("reduced_design_matrix must be provided in LRT mode.")
            # Stands in for whichever LFC is under test, used in the position that LFC occupies:
            # beta and gamma are each subtracted from the initiation term in their own read class.
            reduced_lfc_term = (reduced_design_matrix @ self.reduced_lfc).unsqueeze(1)
            tested_parameter = self.lrt_specification.tested_parameter
            if tested_parameter == TestableParameters.BETA:
                nascent_term = init_over_deg_per_intron - reduced_lfc_term
            elif tested_parameter == TestableParameters.GAMMA:
                unspliced_term = init_over_deg_per_intron - reduced_lfc_term
            else:
                raise ValueError(f'Unhandled tested parameter: {tested_parameter!r}')

        return init_over_deg_term, nascent_term, unspliced_term

    def forward(self,
                design_matrix: torch.Tensor,
                log_library_sizes: torch.Tensor,
                isoform_length_offset: torch.Tensor,
                reduced_design_matrix: Optional[torch.Tensor] = None):
        lfc_term_exon, lfc_term_nascent, lfc_term_unspliced = self._get_lfc_terms_per_read_class(
            design_matrix, reduced_design_matrix)

        predicted_log_reads_exon = (
                self.intercept_exon
                + log_library_sizes.unsqueeze(1)
                + isoform_length_offset
                + lfc_term_exon
        )

        # pi is not an independent quantity: it is the nascent fraction implied by the two intron
        # read classes. Deriving it from the same two terms that build the counts is what keeps
        # the coverage loss and the Poisson count loss fitting one pi under every null.
        predicted_pi = torch.sigmoid(
            self.intercept_pi_logit + lfc_term_nascent - lfc_term_unspliced)

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
            'lfc_elong_over_deg': ('feature',),
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
                predicted_pi: torch.Tensor):
        loss_exon_counts = self.loss_function_exon_counts(predicted_reads_exon,
                                                          reads_exon)
        loss_intron_counts = self.loss_function_intron_counts(predicted_reads_intron, reads_intron)
        loss_intron_coverage = self.loss_function_intron_coverage(predicted_pi, intron_coverage)

        total_loss = loss_exon_counts + loss_intron_counts + loss_intron_coverage
        return total_loss
