"""Optimizer utilities and configurations."""

import torch
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ConstantLR,
    SequentialLR,
)
from typing import Iterable, Optional
from loguru import logger


def get_optimizer(
    model_parameters: Iterable[torch.nn.Parameter],
    optimizer_type: str = "adamw",
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """Get optimizer instance.

    Args:
        model_parameters: Model parameters to optimize
        optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd')
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        betas: Beta coefficients for Adam optimizers
        eps: Epsilon for numerical stability

    Returns:
        Optimizer instance
    """
    if optimizer_type.lower() == "adamw":
        optimizer = AdamW(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
    elif optimizer_type.lower() == "adam":
        optimizer = Adam(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
    elif optimizer_type.lower() == "sgd":
        optimizer = SGD(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    logger.info(f"Created {optimizer_type} optimizer with lr={learning_rate}")
    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    num_training_steps: int = 1000,
    num_warmup_steps: int = 100,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler.

    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler ('cosine', 'linear', 'constant')
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps

    Returns:
        Learning rate scheduler
    """
    if scheduler_type.lower() == "cosine":
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=num_warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - num_warmup_steps,
            eta_min=1e-7,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[num_warmup_steps],
        )

    elif scheduler_type.lower() == "linear":
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=num_warmup_steps,
        )
        decay_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=num_training_steps - num_warmup_steps,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[num_warmup_steps],
        )

    elif scheduler_type.lower() == "constant":
        scheduler = ConstantLR(optimizer, factor=1.0, total_iters=num_training_steps)

    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    logger.info(f"Created {scheduler_type} scheduler with {num_warmup_steps} warmup steps")
    return scheduler
