"""Experiment 001 - the current fitting strategy, unchanged, with per-epoch logging added.

This is the baseline every later experiment is compared against, so the fitting behaviour has to
match `rna_kinetics.estimation` exactly. The only difference is bookkeeping: `train_model` already
records a loss per epoch in `TrainingResults.losses`, and `_train_two_stage` already runs two of
those, but both are currently thrown away. Here they are collected and returned.

An experiment module exposes two functions, `fit_global_rna_kinetics` and
`fit_global_intron_coverage`, each returning (model_param_df, test_results_df, training_log_df).
Everything it does not change is imported from `rna_kinetics`, so a diff between two experiment
modules is the actual difference in strategy.
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

EXPERIMENT_DESCRIPTION = 'Baseline: two-stage LBFGS exactly as in rna_kinetics.estimation.'


def _train_two_stage_with_log(
        model: nn.Module,
        global_param_names: set[str],
        make_closures: Callable[[optim.Optimizer], tuple[Callable, Callable]],
        fit_label: str,
        verbose: bool = False,
) -> tuple[nn.Module, TrainingResults, list[dict]]:
    """A copy of rna_kinetics.estimation._train_two_stage that also returns the loss curves.

    Stage 1 optimizes only the shared parameters with the per-gene ones frozen; stage 2 unfreezes
    everything and refines from that warm start. Stage 1's TrainingResults are discarded upstream,
    which is why its losses have to be captured here rather than read off the return value.
    """
    training_log: list[dict] = []

    def record(stage: str, training_results: TrainingResults) -> None:
        for epoch, loss in enumerate(training_results.losses, start=1):
            training_log.append({'fit_label': fit_label, 'stage': stage, 'epoch': epoch, 'loss': loss})

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


def fit_global_rna_kinetics(
        global_gene_data: GlobalGeneData,
        dataset_metadata: DatasetMetadata,
        device: str = 'cpu',
        verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_gene_data = global_gene_data.to(device)
    dataset_metadata = dataset_metadata.to(device)
    training_log: list[dict] = []

    model = GlobalRNAKineticsModel(
        feature_names=dataset_metadata.feature_names,
        gene_names=global_gene_data.gene_names,
        intron_names=global_gene_data.intron_names,
        gene_idx=global_gene_data.gene_idx,
    ).to(device)
    model.initialize_parameters(global_gene_data, dataset_metadata.library_sizes, dataset_metadata.design_matrix)

    model, results_full, log_full = _train_two_stage_with_log(
        model,
        global_param_names={'lfc_elong_over_deg', 'lfc_splice_over_deg'},
        make_closures=lambda opt: _make_global_rna_kinetics_closures(model, opt, global_gene_data, dataset_metadata),
        fit_label='full_model',
        verbose=verbose,
    )
    training_log.extend(log_full)

    feature_names = dataset_metadata.feature_names
    model_param_df = model.get_param_df()
    model_param_df['loss_full_model'] = results_full.final_loss
    model_param_df['training_diverged_full_model'] = results_full.training_diverged
    model_param_df['training_converged_within_max_epochs_full_model'] = results_full.converged_within_max_epochs

    test_results_list: list[dict] = []
    for _, lrt_row in dataset_metadata.lrt_metadata.iterrows():
        reduced_matrix = dataset_metadata.reduced_matrices[lrt_row['test_id']].to(device)

        for tested_parameter in (TestableParameters.BETA, TestableParameters.GAMMA):
            lrt_specification = LRTSpecification(
                num_features_reduced_matrix=reduced_matrix.shape[1],
                tested_parameter=tested_parameter,
            )
            model_reduced = GlobalRNAKineticsModel(
                feature_names=feature_names,
                gene_names=global_gene_data.gene_names,
                intron_names=global_gene_data.intron_names,
                gene_idx=global_gene_data.gene_idx,
                lrt_specification=lrt_specification,
            ).to(device)

            full_state = model.state_dict()
            reduced_state = model_reduced.state_dict()
            for key in full_state:
                reduced_state[key] = full_state[key]

            if reduced_matrix.shape[1] > 0:
                full_lfc = (model.lfc_elong_over_deg if tested_parameter == TestableParameters.BETA
                            else model.lfc_splice_over_deg)
                full_contribution = dataset_metadata.design_matrix @ full_lfc.detach()
                reduced_state['reduced_lfc'] = torch.linalg.lstsq(reduced_matrix, full_contribution).solution

            model_reduced.load_state_dict(reduced_state)

            fit_label = f'{lrt_row["test_id"]}|{tested_parameter}'
            model_reduced, results_reduced, log_reduced = _train_two_stage_with_log(
                model_reduced,
                global_param_names={'lfc_elong_over_deg', 'lfc_splice_over_deg', 'reduced_lfc'},
                make_closures=lambda opt: _make_global_rna_kinetics_closures(
                    model_reduced, opt, global_gene_data, dataset_metadata, reduced_matrix,
                ),
                fit_label=fit_label,
                verbose=verbose,
            )
            training_log.extend(log_reduced)

            full_lfc_values = (model.lfc_elong_over_deg.detach()
                               if tested_parameter == TestableParameters.BETA
                               else model.lfc_splice_over_deg.detach())
            lfc_positive = (0.0 if pd.isna(lrt_row['lfc_column_positive'])
                            else full_lfc_values[feature_names.index(lrt_row['lfc_column_positive'])].item())
            lfc_negative = (0.0 if pd.isna(lrt_row['lfc_column_negative'])
                            else full_lfc_values[feature_names.index(lrt_row['lfc_column_negative'])].item())

            chi2_statistic = 2 * (results_reduced.final_loss - results_full.final_loss)
            test_result = lrt_row.to_dict()
            test_result['tested_parameter'] = tested_parameter
            test_result['lfc'] = lfc_positive - lfc_negative
            test_result['loss_full_model'] = results_full.final_loss
            test_result['loss_reduced_model'] = results_reduced.final_loss
            test_result['chi2_test_statistics'] = chi2_statistic
            test_result['p_value'] = 1 - stats.chi2.cdf(chi2_statistic, df=lrt_row['lrt_df'])
            test_result['training_diverged_reduced_model'] = results_reduced.training_diverged
            test_result['training_converged_within_max_epochs_reduced_model'] = (
                results_reduced.converged_within_max_epochs)
            test_result['num_epochs_full_model'] = results_full.num_epochs
            test_result['num_epochs_reduced_model'] = results_reduced.num_epochs
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
        fit_label='full_model',
        verbose=verbose,
    )
    training_log.extend(log_full)

    model_param_df = model_full.get_param_df()
    model_param_df['loss_full_model'] = results_full.final_loss
    model_param_df['training_diverged_full_model'] = results_full.training_diverged
    model_param_df['training_converged_within_max_epochs_full_model'] = results_full.converged_within_max_epochs
    model_param_df = model_param_df.set_index(['parameter_type', 'feature_name', 'intron_name'], drop=False)

    test_results_list: list[dict] = []
    for _, lrt_row in dataset_metadata.lrt_metadata.iterrows():
        reduced_matrix = dataset_metadata.reduced_matrices[lrt_row['test_id']].to(device)
        num_reduced_features = reduced_matrix.shape[1]
        placeholder_names = [f'reduced_feature_{i}' for i in range(num_reduced_features)]

        model_reduced = IntronCoverageModel(
            feature_names=placeholder_names,
            intron_names=intron_names,
        ).to(device)

        with torch.no_grad():
            model_reduced.intercept_pi_logit.data.copy_(model_full.intercept_pi_logit.data)
            if num_reduced_features > 0:
                full_lfc_contribution = dataset_metadata.design_matrix @ model_full.lfc_elong_over_splice
                model_reduced.lfc_elong_over_splice.data.copy_(
                    torch.linalg.lstsq(reduced_matrix, full_lfc_contribution).solution
                )

        model_reduced, results_reduced, log_reduced = _train_two_stage_with_log(
            model_reduced,
            global_param_names={'lfc_elong_over_splice'},
            make_closures=lambda opt: _make_intron_coverage_closures(model_reduced, opt, coverage, reduced_matrix),
            fit_label=str(lrt_row['test_id']),
            verbose=verbose,
        )
        training_log.extend(log_reduced)

        lfc_positive = 0.0 if pd.isna(lrt_row['lfc_column_positive']) else model_param_df.loc[
            (PARAMETER_WIRE_NAMES['lfc_elong_over_splice'], lrt_row['lfc_column_positive'], None)]['value']
        lfc_negative = 0.0 if pd.isna(lrt_row['lfc_column_negative']) else model_param_df.loc[
            (PARAMETER_WIRE_NAMES['lfc_elong_over_splice'], lrt_row['lfc_column_negative'], None)]['value']

        chi2_statistic = 2 * (results_reduced.final_loss - results_full.final_loss)
        test_result = lrt_row.to_dict()
        test_result['tested_parameter'] = PARAMETER_WIRE_NAMES['lfc_elong_over_splice']
        test_result['intron_name'] = None
        test_result['lfc'] = lfc_positive - lfc_negative
        test_result['loss_full_model'] = results_full.final_loss
        test_result['loss_reduced_model'] = results_reduced.final_loss
        test_result['chi2_test_statistics'] = chi2_statistic
        test_result['p_value'] = 1 - stats.chi2.cdf(chi2_statistic, df=lrt_row['lrt_df'])
        test_result['training_diverged_reduced_model'] = results_reduced.training_diverged
        test_result['training_converged_within_max_epochs_reduced_model'] = (
            results_reduced.converged_within_max_epochs)
        test_result['num_epochs_full_model'] = results_full.num_epochs
        test_result['num_epochs_reduced_model'] = results_reduced.num_epochs
        test_results_list.append(test_result)

    return model_param_df.reset_index(drop=True), pd.DataFrame(test_results_list), pd.DataFrame(training_log)
