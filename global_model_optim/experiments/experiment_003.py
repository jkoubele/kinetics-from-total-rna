"""Experiment 003 - map the loss surface in beta and gamma by starting the full fit from a grid.

No reduced models and no LRTs: this fits the full model only, repeatedly, from different starting
values of the two shared LFCs. Those are the only parameters `initialize_parameters` leaves
uninformed -- it derives `intercept_exon`, `lfc_init_over_deg` (Poisson GLM), `intercept_pi_logit`
(grid search over pi) and `intercept_intron` from the data, but `lfc_elong_over_deg` and
`lfc_splice_over_deg` start at exactly zero. Zero is a default, not a principled choice, so varying
it is a fair test rather than a handicap.

The question is whether the optimum experiment_002 reaches is the global one. We already know the
surface has more than one basin -- experiment_002 found better optima than the heuristic start in 15
of 81 full-scale tests, with loss gains up to 18565 -- so this is quantitative, not exploratory.

Each grid point is reported as its own row, with the starting values, the loss reached, and the
fitted LFCs. If every start converges to the same loss and the same LFCs, the optimum is very likely
global. If they scatter, everything resting on the global model is provisional.

The grid perturbs all features of a parameter together: for a design with several columns, beta
starts at (b, b, ...) and gamma at (g, g, ...). That explores the overall-shift directions, which is
where the signature-matrix question lives, and the optimizer is free to separate the components
afterwards. A full per-feature grid is infeasible once a design has more than a couple of columns.
"""

import itertools
from typing import Callable

import pandas as pd
import torch
from torch import nn, optim

from rna_kinetics.data import DatasetMetadata, GlobalGeneData
from rna_kinetics.estimation import (
    TrainingResults,
    _make_global_rna_kinetics_closures,
    _make_intron_coverage_closures,
    make_lbfgs_optimizer,
    train_model,
)
from rna_kinetics.models import GlobalRNAKineticsModel, IntronCoverageModel

EXPERIMENT_DESCRIPTION = ('Full model only, fitted from a grid of starting values for the shared '
                          'LFCs, to test whether the optimum is global.')

# Natural-log LFCs. Fitted values across the nine datasets span roughly 0 to -3.4, so this covers
# the observed range in both directions.
INIT_GRID_VALUES = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)


def _train_two_stage_with_log(
        model: nn.Module,
        global_param_names: set[str],
        make_closures: Callable[[optim.Optimizer], tuple[Callable, Callable]],
        fit_label: str,
        verbose: bool = False,
) -> tuple[nn.Module, TrainingResults, list[dict]]:
    """The same two-stage schedule as experiment_001 and _002, so the runs stay comparable."""
    training_log: list[dict] = []

    def record(stage: str, training_results: TrainingResults) -> None:
        for epoch, loss in enumerate(training_results.losses, start=1):
            training_log.append({'fit_label': fit_label, 'stage': stage, 'epoch': epoch, 'loss': loss})

    non_empty_parameters = [p for p in model.parameters() if p.numel() > 0]
    if non_empty_parameters:
        _, evaluate_initial_loss = make_closures(optim.LBFGS(non_empty_parameters, lr=1.0))
        training_log.append({'fit_label': fit_label, 'stage': 'initial', 'epoch': 0,
                             'loss': evaluate_initial_loss().item()})

    for name, param in model.named_parameters():
        param.requires_grad_(name in global_param_names)
    stage_1_params = [p for p in model.parameters() if p.requires_grad and p.numel() > 0]
    if stage_1_params:
        optimizer_stage_1 = optim.LBFGS(
            stage_1_params,
            lr=1.0, max_iter=20, tolerance_change=1e-9, tolerance_grad=1e-7,
            history_size=100, line_search_fn='strong_wolfe',
        )
        closure_stage_1, evaluate_stage_1 = make_closures(optimizer_stage_1)
        model, results_stage_1 = train_model(model, optimizer_stage_1, closure_stage_1, evaluate_stage_1,
                                             max_epochs=500, verbose=verbose)
        record('stage_1', results_stage_1)

    for param in model.parameters():
        param.requires_grad_(True)
    optimizer_stage_2 = make_lbfgs_optimizer(model)
    closure_stage_2, evaluate_stage_2 = make_closures(optimizer_stage_2)
    model, results_stage_2 = train_model(model, optimizer_stage_2, closure_stage_2, evaluate_stage_2,
                                         max_epochs=500, verbose=verbose)
    record('stage_2', results_stage_2)

    return model, results_stage_2, training_log


def _format_vector(tensor: torch.Tensor) -> str:
    return ','.join(f'{value:.4f}' for value in tensor.detach().flatten().tolist())


def fit_global_rna_kinetics(
        global_gene_data: GlobalGeneData,
        dataset_metadata: DatasetMetadata,
        device: str = 'cpu',
        verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_gene_data = global_gene_data.to(device)
    dataset_metadata = dataset_metadata.to(device)
    training_log: list[dict] = []
    grid_results: list[dict] = []
    best_loss = float('inf')
    best_param_df: pd.DataFrame | None = None

    grid_points = list(itertools.product(INIT_GRID_VALUES, repeat=2))
    for point_index, (init_beta, init_gamma) in enumerate(grid_points):
        model = GlobalRNAKineticsModel(
            feature_names=dataset_metadata.feature_names,
            gene_names=global_gene_data.gene_names,
            intron_names=global_gene_data.intron_names,
            gene_idx=global_gene_data.gene_idx,
        ).to(device)
        model.initialize_parameters(global_gene_data, dataset_metadata.library_sizes,
                                    dataset_metadata.design_matrix)
        with torch.no_grad():
            model.lfc_elong_over_deg.fill_(init_beta)
            model.lfc_splice_over_deg.fill_(init_gamma)

        fit_label = f'init_beta_{init_beta}_gamma_{init_gamma}'
        model, results, log = _train_two_stage_with_log(
            model,
            global_param_names={'lfc_elong_over_deg', 'lfc_splice_over_deg'},
            make_closures=lambda opt: _make_global_rna_kinetics_closures(
                model, opt, global_gene_data, dataset_metadata),
            fit_label=fit_label,
            verbose=verbose,
        )
        training_log.extend(log)

        grid_results.append({
            'test_id': f'grid_{point_index:03d}',
            'tested_parameter': 'full_model_init',
            'init_beta': init_beta,
            'init_gamma': init_gamma,
            'loss_full_model': results.final_loss,
            'final_beta': _format_vector(model.lfc_elong_over_deg),
            'final_gamma': _format_vector(model.lfc_splice_over_deg),
            'num_epochs_full_model': results.num_epochs,
            'training_diverged_full_model': results.training_diverged,
            'training_converged_within_max_epochs_full_model': results.converged_within_max_epochs,
        })
        print(f'  grid {point_index + 1}/{len(grid_points)}  beta0={init_beta:+.1f} '
              f'gamma0={init_gamma:+.1f}  loss={results.final_loss:.3f}', flush=True)

        if results.final_loss < best_loss:
            best_loss = results.final_loss
            best_param_df = model.get_param_df()
            best_param_df['loss_full_model'] = results.final_loss
            best_param_df['training_diverged_full_model'] = results.training_diverged
            best_param_df['training_converged_within_max_epochs_full_model'] = (
                results.converged_within_max_epochs)

    return best_param_df, pd.DataFrame(grid_results), pd.DataFrame(training_log)


def fit_global_intron_coverage(
        coverage: torch.Tensor,
        dataset_metadata: DatasetMetadata,
        intron_names: list[str],
        device: str = 'cpu',
        verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The coverage model has a single shared LFC, so the grid is one-dimensional."""
    coverage = coverage.to(device)
    dataset_metadata = dataset_metadata.to(device)
    training_log: list[dict] = []
    grid_results: list[dict] = []
    best_loss = float('inf')
    best_param_df: pd.DataFrame | None = None

    for point_index, init_value in enumerate(INIT_GRID_VALUES):
        model = IntronCoverageModel(
            feature_names=dataset_metadata.feature_names,
            intron_names=intron_names,
        ).to(device)
        model.initialize_parameters(coverage)
        with torch.no_grad():
            model.lfc_elong_over_splice.fill_(init_value)

        fit_label = f'init_elong_over_splice_{init_value}'
        model, results, log = _train_two_stage_with_log(
            model,
            global_param_names={'lfc_elong_over_splice'},
            make_closures=lambda opt: _make_intron_coverage_closures(
                model, opt, coverage, dataset_metadata.design_matrix),
            fit_label=fit_label,
            verbose=verbose,
        )
        training_log.extend(log)

        grid_results.append({
            'test_id': f'grid_{point_index:03d}',
            'tested_parameter': 'full_model_init',
            'init_beta': init_value,
            'init_gamma': None,
            'loss_full_model': results.final_loss,
            'final_beta': _format_vector(model.lfc_elong_over_splice),
            'final_gamma': None,
            'num_epochs_full_model': results.num_epochs,
            'training_diverged_full_model': results.training_diverged,
            'training_converged_within_max_epochs_full_model': results.converged_within_max_epochs,
        })
        print(f'  grid {point_index + 1}/{len(INIT_GRID_VALUES)}  init={init_value:+.1f}  '
              f'loss={results.final_loss:.3f}', flush=True)

        if results.final_loss < best_loss:
            best_loss = results.final_loss
            best_param_df = model.get_param_df()
            best_param_df['loss_full_model'] = results.final_loss
            best_param_df['training_diverged_full_model'] = results.training_diverged
            best_param_df['training_converged_within_max_epochs_full_model'] = (
                results.converged_within_max_epochs)

    return best_param_df, pd.DataFrame(grid_results), pd.DataFrame(training_log)
