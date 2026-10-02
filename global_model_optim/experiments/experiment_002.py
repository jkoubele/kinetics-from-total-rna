"""Experiment 002 - restart the full model whenever a reduced model beats it.

A negative chi2 means a reduced (more constrained) model reached a lower loss than the full one,
which cannot happen at an optimum: the full model's parameter space contains the reduced one's. So
whenever that happens the full fit is the one that failed, and the reduced fit is a strictly better
starting point for it than the initialisation was.

The loop is:

  1. fit the full model
  2. fit every reduced model, warm-started from the current best full fit
  3. if some reduced fit beats the full one by more than CHI2_TOLERANCE, refit the full model
     warm-started from that reduced fit, and go back to step 2
  4. stop when no reduced fit beats the full one, when the full refit stops improving, or after
     MAX_FULL_MODEL_RESTARTS restarts

Both sides keep the best fit they have ever reached, since a refit can come out worse than the one
it replaced. That also means chi2 is computed from the best full and best reduced losses, so it can
only move towards being non-negative as the loop runs.

Warm-starting the full model from a reduced one needs the reverse of the projection the reduced fits
already use: the reduced model constrains the tested term to `reduced_matrix @ reduced_lfc`, and the
full coefficients reproducing that contribution are the least-squares solution against the full
design matrix. The reduced matrix's column space is a subspace of the design matrix's, so this is
exact rather than an approximation.
"""

from typing import Callable

import pandas as pd
import torch
from scipy import stats
from torch import nn, optim

from rna_kinetics.data import DatasetMetadata, GlobalGeneData
from rna_kinetics.estimation import (
    TrainingResults,
    _make_global_rna_kinetics_closures,
    _make_intron_coverage_closures,
    make_lbfgs_optimizer,
    train_model,
)
from rna_kinetics.models import (
    PARAMETER_WIRE_NAMES,
    GlobalRNAKineticsModel,
    IntronCoverageModel,
    LRTSpecification,
    TestableParameters,
)

EXPERIMENT_DESCRIPTION = ('Refit the full model warm-started from any reduced fit that beats it, '
                          'repeating until no reduced fit does.')

MAX_FULL_MODEL_RESTARTS = 10
# A restart is triggered when chi2 drops below -CHI2_TOLERANCE, i.e. when the reduced model's loss is
# more than CHI2_TOLERANCE/2 below the full model's. Anything smaller is numerical noise rather than
# a failed fit, and chasing it would just burn restarts.
CHI2_TOLERANCE = 0.1


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _train_two_stage_with_log(
        model: nn.Module,
        global_param_names: set[str],
        make_closures: Callable[[optim.Optimizer], tuple[Callable, Callable]],
        fit_label: str,
        verbose: bool = False,
) -> tuple[nn.Module, TrainingResults, list[dict]]:
    """As in experiment_001: the unchanged two-stage schedule, with the loss curves kept."""
    training_log: list[dict] = []

    def record(stage: str, training_results: TrainingResults) -> None:
        for epoch, loss in enumerate(training_results.losses, start=1):
            training_log.append({'fit_label': fit_label, 'stage': stage, 'epoch': epoch, 'loss': loss})

    # train_model steps before it evaluates, so the loss at the starting point is never recorded.
    # For a warm-started fit that is the number worth seeing, so take it here.
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


def _project_reduced_contribution(design_matrix: torch.Tensor,
                                  reduced_matrix: torch.Tensor,
                                  reduced_lfc: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Split the reduced model's contribution into a design-matrix part and a constant.

    The design matrices carry no intercept column -- the model holds its own intercepts -- so the
    reduced design is nested in [1, X], not in X. Projecting onto X alone is therefore exact only
    when the reduced matrix happens to avoid the constant direction, which is the case for a
    two-level factor (where the reduced matrix is empty) but not in general: relevelling to group_2
    and dropping group_1's column can leave the original reference level's indicator, and that is
    outside col(X). The leftover constant is returned so the caller can absorb it into the
    intercepts, which makes the warm start reproduce the reduced model's predictions exactly.

    `reduced_lfc` is 1D for the RNA kinetics model and (num_reduced_features, 1) for the coverage
    model; the returned coefficients match whichever was given.
    """
    is_one_dimensional = reduced_lfc.dim() == 1
    reduced_lfc_2d = reduced_lfc.unsqueeze(-1) if is_one_dimensional else reduced_lfc
    reduced_contribution = reduced_matrix @ reduced_lfc_2d  # (num_samples, 1)

    num_samples = design_matrix.shape[0]
    intercept_column = torch.ones(num_samples, 1, dtype=design_matrix.dtype, device=design_matrix.device)
    design_with_intercept = torch.cat([intercept_column, design_matrix], dim=1)
    solution = torch.linalg.lstsq(design_with_intercept, reduced_contribution).solution

    constant_offset = solution[0, 0].item()
    full_lfc = solution[1:]
    return (full_lfc.squeeze(-1) if is_one_dimensional else full_lfc), constant_offset



def fit_global_rna_kinetics(
        global_gene_data: GlobalGeneData,
        dataset_metadata: DatasetMetadata,
        device: str = 'cpu',
        verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_gene_data = global_gene_data.to(device)
    dataset_metadata = dataset_metadata.to(device)
    feature_names = dataset_metadata.feature_names
    training_log: list[dict] = []

    def build_model(lrt_specification: LRTSpecification | None = None) -> GlobalRNAKineticsModel:
        return GlobalRNAKineticsModel(
            feature_names=feature_names,
            gene_names=global_gene_data.gene_names,
            intron_names=global_gene_data.intron_names,
            gene_idx=global_gene_data.gene_idx,
            lrt_specification=lrt_specification,
        ).to(device)

    model_full = build_model()
    model_full.initialize_parameters(global_gene_data, dataset_metadata.library_sizes,
                                     dataset_metadata.design_matrix)
    model_full, results_full, log_full = _train_two_stage_with_log(
        model_full,
        global_param_names={'lfc_elong_over_deg', 'lfc_splice_over_deg'},
        make_closures=lambda opt: _make_global_rna_kinetics_closures(
            model_full, opt, global_gene_data, dataset_metadata),
        fit_label='full_model|round_0',
        verbose=verbose,
    )
    training_log.extend(log_full)
    best_full_loss = results_full.final_loss
    best_full_state = _clone_state(model_full)
    best_full_results = results_full

    reduced_specifications = [(lrt_row, tested_parameter)
                              for _, lrt_row in dataset_metadata.lrt_metadata.iterrows()
                              for tested_parameter in (TestableParameters.BETA, TestableParameters.GAMMA)]
    best_reduced: dict[tuple[str, str], dict] = {}
    num_full_model_restarts = 0

    for round_index in range(MAX_FULL_MODEL_RESTARTS + 1):
        for lrt_row, tested_parameter in reduced_specifications:
            reduced_matrix = dataset_metadata.reduced_matrices[lrt_row['test_id']].to(device)
            lrt_specification = LRTSpecification(num_features_reduced_matrix=reduced_matrix.shape[1],
                                                 tested_parameter=tested_parameter)
            model_reduced = build_model(lrt_specification)

            reduced_state = model_reduced.state_dict()
            for key in best_full_state:
                reduced_state[key] = best_full_state[key]
            if reduced_matrix.shape[1] > 0:
                tested_attribute = ('lfc_elong_over_deg' if tested_parameter == TestableParameters.BETA
                                    else 'lfc_splice_over_deg')
                full_contribution = dataset_metadata.design_matrix @ best_full_state[tested_attribute]
                reduced_state['reduced_lfc'] = torch.linalg.lstsq(
                    reduced_matrix, full_contribution.unsqueeze(-1)).solution.squeeze(-1)
            model_reduced.load_state_dict(reduced_state)

            fit_label = f'{lrt_row["test_id"]}|{tested_parameter}|round_{round_index}'
            model_reduced, results_reduced, log_reduced = _train_two_stage_with_log(
                model_reduced,
                global_param_names={'lfc_elong_over_deg', 'lfc_splice_over_deg', 'reduced_lfc'},
                make_closures=lambda opt: _make_global_rna_kinetics_closures(
                    model_reduced, opt, global_gene_data, dataset_metadata, reduced_matrix),
                fit_label=fit_label,
                verbose=verbose,
            )
            training_log.extend(log_reduced)

            key = (lrt_row['test_id'], tested_parameter)
            if key not in best_reduced or results_reduced.final_loss < best_reduced[key]['loss']:
                best_reduced[key] = {
                    'loss': results_reduced.final_loss,
                    'results': results_reduced,
                    'reduced_lfc': model_reduced.reduced_lfc.detach().clone(),
                    'state': _clone_state(model_reduced),
                    'reduced_matrix': reduced_matrix,
                    'tested_parameter': tested_parameter,
                    'lrt_row': lrt_row,
                }

        # The reduced fit that beats the full model by the most is the best restart point.
        winning_key = min(best_reduced, key=lambda k: best_reduced[k]['loss'])
        chi2_of_winner = 2 * (best_reduced[winning_key]['loss'] - best_full_loss)
        if chi2_of_winner >= -CHI2_TOLERANCE or round_index == MAX_FULL_MODEL_RESTARTS:
            break

        winner = best_reduced[winning_key]
        tests_beta = winner['tested_parameter'] == TestableParameters.BETA
        tested_attribute = 'lfc_elong_over_deg' if tests_beta else 'lfc_splice_over_deg'
        restart_state = {name: tensor.detach().clone() for name, tensor in winner['state'].items()
                         if name != 'reduced_lfc'}
        full_lfc, constant_offset = _project_reduced_contribution(
            dataset_metadata.design_matrix, winner['reduced_matrix'], winner['reduced_lfc'])
        restart_state[tested_attribute] = full_lfc
        # Absorb the constant so the warm start reproduces the reduced model's predictions exactly.
        # Beta enters only through (intercept_pi_logit - elong_term), so shifting the pi intercept is
        # enough. Gamma enters through both (intercept_pi_logit + splice_term) and
        # (intercept_intron - splice_term), so both intercepts move, in opposite directions.
        if tests_beta:
            restart_state['intercept_pi_logit'] = restart_state['intercept_pi_logit'] - constant_offset
        else:
            restart_state['intercept_pi_logit'] = restart_state['intercept_pi_logit'] + constant_offset
            restart_state['intercept_intron'] = restart_state['intercept_intron'] - constant_offset

        model_full = build_model()
        model_full.load_state_dict(restart_state)
        model_full, results_full, log_full = _train_two_stage_with_log(
            model_full,
            global_param_names={'lfc_elong_over_deg', 'lfc_splice_over_deg'},
            make_closures=lambda opt: _make_global_rna_kinetics_closures(
                model_full, opt, global_gene_data, dataset_metadata),
            fit_label=f'full_model|round_{round_index + 1}',
            verbose=verbose,
        )
        training_log.extend(log_full)
        num_full_model_restarts += 1

        if results_full.final_loss < best_full_loss:
            best_full_loss = results_full.final_loss
            best_full_state = _clone_state(model_full)
            best_full_results = results_full
        else:
            # The restart did not help, so further rounds would repeat the same work.
            break

    model_full = build_model()
    model_full.load_state_dict(best_full_state)
    model_param_df = model_full.get_param_df()
    model_param_df['loss_full_model'] = best_full_loss
    model_param_df['training_diverged_full_model'] = best_full_results.training_diverged
    model_param_df['training_converged_within_max_epochs_full_model'] = (
        best_full_results.converged_within_max_epochs)

    test_results_list: list[dict] = []
    for lrt_row, tested_parameter in reduced_specifications:
        entry = best_reduced[(lrt_row['test_id'], tested_parameter)]
        results_reduced = entry['results']
        full_lfc_values = (best_full_state['lfc_elong_over_deg']
                           if tested_parameter == TestableParameters.BETA
                           else best_full_state['lfc_splice_over_deg'])
        lfc_positive = (0.0 if pd.isna(lrt_row['lfc_column_positive'])
                        else full_lfc_values[feature_names.index(lrt_row['lfc_column_positive'])].item())
        lfc_negative = (0.0 if pd.isna(lrt_row['lfc_column_negative'])
                        else full_lfc_values[feature_names.index(lrt_row['lfc_column_negative'])].item())

        chi2_statistic = 2 * (entry['loss'] - best_full_loss)
        test_result = lrt_row.to_dict()
        test_result['tested_parameter'] = tested_parameter
        test_result['lfc'] = lfc_positive - lfc_negative
        test_result['loss_full_model'] = best_full_loss
        test_result['loss_reduced_model'] = entry['loss']
        test_result['chi2_test_statistics'] = chi2_statistic
        test_result['p_value'] = 1 - stats.chi2.cdf(chi2_statistic, df=lrt_row['lrt_df'])
        test_result['training_diverged_reduced_model'] = results_reduced.training_diverged
        test_result['training_converged_within_max_epochs_reduced_model'] = (
            results_reduced.converged_within_max_epochs)
        test_result['num_epochs_full_model'] = best_full_results.num_epochs
        test_result['num_epochs_reduced_model'] = results_reduced.num_epochs
        test_result['num_full_model_restarts'] = num_full_model_restarts
        test_results_list.append(test_result)

    return model_param_df, pd.DataFrame(test_results_list), pd.DataFrame(training_log)


def fit_global_intron_coverage(
        coverage: torch.Tensor,
        dataset_metadata: DatasetMetadata,
        intron_names: list[str],
        device: str = 'cpu',
        verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coverage = coverage.to(device)
    dataset_metadata = dataset_metadata.to(device)
    training_log: list[dict] = []

    model_full = IntronCoverageModel(
        feature_names=dataset_metadata.feature_names,
        intron_names=intron_names,
    ).to(device)
    model_full.initialize_parameters(coverage)
    model_full, results_full, log_full = _train_two_stage_with_log(
        model_full,
        global_param_names={'lfc_elong_over_splice'},
        make_closures=lambda opt: _make_intron_coverage_closures(
            model_full, opt, coverage, dataset_metadata.design_matrix),
        fit_label='full_model|round_0',
        verbose=verbose,
    )
    training_log.extend(log_full)
    best_full_loss = results_full.final_loss
    best_full_state = _clone_state(model_full)
    best_full_results = results_full

    best_reduced: dict[str, dict] = {}
    num_full_model_restarts = 0

    for round_index in range(MAX_FULL_MODEL_RESTARTS + 1):
        for _, lrt_row in dataset_metadata.lrt_metadata.iterrows():
            reduced_matrix = dataset_metadata.reduced_matrices[lrt_row['test_id']].to(device)
            num_reduced_features = reduced_matrix.shape[1]
            placeholder_names = [f'reduced_feature_{i}' for i in range(num_reduced_features)]

            model_reduced = IntronCoverageModel(
                feature_names=placeholder_names,
                intron_names=intron_names,
            ).to(device)
            with torch.no_grad():
                model_reduced.intercept_pi_logit.data.copy_(best_full_state['intercept_pi_logit'])
                if num_reduced_features > 0:
                    full_contribution = (dataset_metadata.design_matrix
                                         @ best_full_state['lfc_elong_over_splice'])
                    model_reduced.lfc_elong_over_splice.data.copy_(
                        torch.linalg.lstsq(reduced_matrix, full_contribution).solution)

            fit_label = f'{lrt_row["test_id"]}|round_{round_index}'
            model_reduced, results_reduced, log_reduced = _train_two_stage_with_log(
                model_reduced,
                global_param_names={'lfc_elong_over_splice'},
                make_closures=lambda opt: _make_intron_coverage_closures(
                    model_reduced, opt, coverage, reduced_matrix),
                fit_label=fit_label,
                verbose=verbose,
            )
            training_log.extend(log_reduced)

            key = lrt_row['test_id']
            if key not in best_reduced or results_reduced.final_loss < best_reduced[key]['loss']:
                best_reduced[key] = {
                    'loss': results_reduced.final_loss,
                    'results': results_reduced,
                    'state': _clone_state(model_reduced),
                    'reduced_lfc': model_reduced.lfc_elong_over_splice.detach().clone(),
                    'reduced_matrix': reduced_matrix,
                    'lrt_row': lrt_row,
                }

        winning_key = min(best_reduced, key=lambda k: best_reduced[k]['loss'])
        chi2_of_winner = 2 * (best_reduced[winning_key]['loss'] - best_full_loss)
        if chi2_of_winner >= -CHI2_TOLERANCE or round_index == MAX_FULL_MODEL_RESTARTS:
            break

        winner = best_reduced[winning_key]
        model_full = IntronCoverageModel(
            feature_names=dataset_metadata.feature_names,
            intron_names=intron_names,
        ).to(device)
        full_lfc, constant_offset = _project_reduced_contribution(
            dataset_metadata.design_matrix, winner['reduced_matrix'], winner['reduced_lfc'])
        with torch.no_grad():
            # pi = sigmoid(intercept_pi_logit - elong_over_splice_term), so the constant moves into
            # the intercept with the opposite sign.
            model_full.intercept_pi_logit.data.copy_(
                winner['state']['intercept_pi_logit'] - constant_offset)
            model_full.lfc_elong_over_splice.data.copy_(full_lfc)

        model_full, results_full, log_full = _train_two_stage_with_log(
            model_full,
            global_param_names={'lfc_elong_over_splice'},
            make_closures=lambda opt: _make_intron_coverage_closures(
                model_full, opt, coverage, dataset_metadata.design_matrix),
            fit_label=f'full_model|round_{round_index + 1}',
            verbose=verbose,
        )
        training_log.extend(log_full)
        num_full_model_restarts += 1

        if results_full.final_loss < best_full_loss:
            best_full_loss = results_full.final_loss
            best_full_state = _clone_state(model_full)
            best_full_results = results_full
        else:
            break

    model_full = IntronCoverageModel(
        feature_names=dataset_metadata.feature_names,
        intron_names=intron_names,
    ).to(device)
    model_full.load_state_dict(best_full_state)
    model_param_df = model_full.get_param_df()
    model_param_df['loss_full_model'] = best_full_loss
    model_param_df['training_diverged_full_model'] = best_full_results.training_diverged
    model_param_df['training_converged_within_max_epochs_full_model'] = (
        best_full_results.converged_within_max_epochs)
    model_param_df = model_param_df.set_index(['parameter_type', 'feature_name', 'intron_name'], drop=False)

    test_results_list: list[dict] = []
    for _, lrt_row in dataset_metadata.lrt_metadata.iterrows():
        entry = best_reduced[lrt_row['test_id']]
        results_reduced = entry['results']

        lfc_positive = 0.0 if pd.isna(lrt_row['lfc_column_positive']) else model_param_df.loc[
            (PARAMETER_WIRE_NAMES['lfc_elong_over_splice'], lrt_row['lfc_column_positive'], None)]['value']
        lfc_negative = 0.0 if pd.isna(lrt_row['lfc_column_negative']) else model_param_df.loc[
            (PARAMETER_WIRE_NAMES['lfc_elong_over_splice'], lrt_row['lfc_column_negative'], None)]['value']

        chi2_statistic = 2 * (entry['loss'] - best_full_loss)
        test_result = lrt_row.to_dict()
        test_result['tested_parameter'] = PARAMETER_WIRE_NAMES['lfc_elong_over_splice']
        test_result['intron_name'] = None
        test_result['lfc'] = lfc_positive - lfc_negative
        test_result['loss_full_model'] = best_full_loss
        test_result['loss_reduced_model'] = entry['loss']
        test_result['chi2_test_statistics'] = chi2_statistic
        test_result['p_value'] = 1 - stats.chi2.cdf(chi2_statistic, df=lrt_row['lrt_df'])
        test_result['training_diverged_reduced_model'] = results_reduced.training_diverged
        test_result['training_converged_within_max_epochs_reduced_model'] = (
            results_reduced.converged_within_max_epochs)
        test_result['num_epochs_full_model'] = best_full_results.num_epochs
        test_result['num_epochs_reduced_model'] = results_reduced.num_epochs
        test_result['num_full_model_restarts'] = num_full_model_restarts
        test_results_list.append(test_result)

    return model_param_df.reset_index(drop=True), pd.DataFrame(test_results_list), pd.DataFrame(training_log)
