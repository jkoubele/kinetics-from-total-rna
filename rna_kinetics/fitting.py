from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn, optim



LOSS_CLAMP_VALUE = 1e30


@dataclass
class TrainingResults:
    final_loss: float
    converged: bool
    num_epochs: int
    losses: list[float]
    training_diverged: bool


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
        loss_change_tolerance=1e-5,
        verbose: bool = False,
) -> tuple[nn.Module, TrainingResults]:
    """
    Train a model in epochs of `optimizer.step(closure)`, and return it at its lowest loss.

    The model is restored to the best state seen, including the starting one, so training never
    returns a model worse than it was given. A non-finite loss stops training and is reported as
    `training_diverged`, with the model restored to its last finite best.

    Training stops once the loss has changed by less than `loss_change_tolerance` (in nats) for
    `max_patience` epochs in a row. The change is absolute rather than relative because the
    losses keep arbitrary constant terms.
    """
    previous_loss = None
    patience_counter = 0

    # Seeded with the starting point, so that training can never return a model worse than the one
    # it was given. Without this the first epoch's loss is accepted unconditionally, even when the
    # optimizer stepped uphill, which would break the warm starts the LRT restarts rely on.
    initial_loss = evaluate_loss().item()
    best_loss = initial_loss if np.isfinite(initial_loss) else np.inf
    best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

    converged = False
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
            if abs(loss - previous_loss) < loss_change_tolerance:
                patience_counter += 1
                if patience_counter >= max_patience:
                    converged = True
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
        converged=converged,
        num_epochs=epoch + 1,
        losses=losses,
        training_diverged=training_diverged,
    )

    return model, training_results


def make_closures(
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


# The global models train in two passes and can need many more epochs than a per-gene fit, which
# `train_model` caps at its own, lower default.
MAX_EPOCHS_GLOBAL_MODEL = 1000
# LBFGS converges slowly on the global models, which hold one intercept per gene or intron, and
# would keep improving the loss by small amounts for far longer than the reported LFCs and LRT
# statistics take to settle. The per-gene fits converge on LBFGS's own criterion and keep the
# tighter default of `train_model`.
LOSS_CHANGE_TOLERANCE_GLOBAL_MODEL = 1e-1
MAX_PATIENCE_GLOBAL_MODEL = 5


def train_global_model(
        model: nn.Module,
        global_param_names: set[str],
        make_closures: Callable[[optim.Optimizer], tuple[Callable, Callable]],
        label: str = '',
        verbose: bool = False,
) -> tuple[nn.Module, TrainingResults]:
    """
    Fit a genome-wide model in two passes: the parameters in `global_param_names` alone, then all
    of them from there.

    A genome-wide variant has tens of thousands of per-unit intercepts standing against a handful
    of shared LFCs. Optimising all of them from the start is badly conditioned and the fit can
    diverge; holding the intercepts at their initial values and moving the shared parameters alone
    is a small, well conditioned problem, and the second pass then starts near the optimum.
    """
    prefix = f'[{label}] ' if label else ''

    # Pass 1: freeze per-gene/per-intron params; optimize only shared params
    if verbose:
        print(f'{prefix}Pass 1: optimizing shared params ({", ".join(sorted(global_param_names))})', flush=True)
    for name, param in model.named_parameters():
        param.requires_grad_(name in global_param_names)
    pass_1_params = [p for p in model.parameters() if p.requires_grad and p.numel() > 0]
    if pass_1_params:
        optimizer_pass_1 = optim.LBFGS(
            pass_1_params,
            lr=1.0, max_iter=20, tolerance_change=1e-9, tolerance_grad=1e-7,
            history_size=100, line_search_fn='strong_wolfe',
        )
        closure_pass_1, eval_pass_1 = make_closures(optimizer_pass_1)
        model, _ = train_model(model, optimizer_pass_1, closure_pass_1, eval_pass_1,
                               max_epochs=MAX_EPOCHS_GLOBAL_MODEL,
                               max_patience=MAX_PATIENCE_GLOBAL_MODEL,
                               loss_change_tolerance=LOSS_CHANGE_TOLERANCE_GLOBAL_MODEL,
                               verbose=verbose)

    # Pass 2: unfreeze all, refine every parameter from the warm start
    if verbose:
        print(f'{prefix}Pass 2: optimizing all params', flush=True)
    for param in model.parameters():
        param.requires_grad_(True)
    optimizer_pass_2 = make_lbfgs_optimizer(model)
    closure_pass_2, eval_pass_2 = make_closures(optimizer_pass_2)
    model, training_results = train_model(model, optimizer_pass_2, closure_pass_2, eval_pass_2,
                                          max_epochs=MAX_EPOCHS_GLOBAL_MODEL,
                                          max_patience=MAX_PATIENCE_GLOBAL_MODEL,
                                          loss_change_tolerance=LOSS_CHANGE_TOLERANCE_GLOBAL_MODEL,
                                          verbose=verbose)
    return model, training_results
