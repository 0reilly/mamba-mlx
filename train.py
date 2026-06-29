"""
Training loop for Mamba experiments.

Provides train_model() and evaluate_model() for training/evaluating
Mamba-style models on sequence modeling tasks from tasks.py.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

from tasks import (
    generate_retrieval_batch,
    generate_addition_batch,
    generate_dyck_batch,
    generate_selective_copy_batch,
)


# ---------------------------------------------------------------------------
# Task dispatch
# ---------------------------------------------------------------------------

_TASK_GENERATORS = {
    "retrieval": generate_retrieval_batch,
    "addition": generate_addition_batch,
    "dyck": generate_dyck_batch,
    "selective_copy": generate_selective_copy_batch,
}

# Tasks that need vocab_size passed to the generator
_VOCAB_TASKS = {"retrieval", "selective_copy"}

# Default vocab sizes for vocab-aware tasks
_DEFAULT_VOCAB_SIZES = {
    "retrieval": 256,
    "selective_copy": 64,
}


def _get_batch(task_name, batch_size, seq_len):
    """Generate a fresh (x, y) data batch for a given task."""
    gen = _TASK_GENERATORS[task_name]
    if task_name in _VOCAB_TASKS:
        vocab_size = _DEFAULT_VOCAB_SIZES[task_name]
        return gen(batch_size, seq_len, vocab_size)
    else:
        return gen(batch_size, seq_len)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


def _cross_entropy_loss(logits, targets):
    """
    Compute masked cross-entropy loss.

    Args:
        logits:  (B, L, V) – model output logits.
        targets: (B, L)    – integer token ids, with -100 for padding.

    Returns:
        Scalar loss averaged over non-padding tokens.
    """
    B, L, V = logits.shape

    # Flatten to (B*L, V) and (B*L,)
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)

    # Per-token cross-entropy (no reduction so we can mask)
    ce = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none")

    # Mask out padding positions (target == -100)
    mask = targets_flat != -100
    mask_f = mask.astype(mx.float32)

    total = mx.sum(mask_f)
    loss = mx.sum(ce * mask_f) / mx.maximum(total, 1.0)

    return loss


# ---------------------------------------------------------------------------
# Accuracy helper
# ---------------------------------------------------------------------------


def _compute_accuracy(logits, targets):
    """
    Compute token-level accuracy ignoring padding.

    Args:
        logits:  (B, L, V) – model logits.
        targets: (B, L)    – integer token ids, -100 for padding.

    Returns:
        Scalar accuracy in [0, 1].
    """
    preds = mx.argmax(logits, axis=-1)          # (B, L)
    mask = targets != -100
    correct = mx.sum((preds == targets) * mask)
    total = mx.sum(mask)
    return correct / mx.maximum(total, 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(model, task_name, num_batches=10, batch_size=32, seq_len=128):
    """
    Evaluate a model on a task over multiple batches.

    Args:
        model:       MLX module (e.g. Mamba model).
        task_name:   One of {'retrieval', 'addition', 'dyck', 'selective_copy'}.
        num_batches: Number of fresh batches to average over.
        batch_size:  Batch size.
        seq_len:     Maximum sequence length.

    Returns:
        dict with keys 'loss' and 'accuracy' (scalar floats).
    """
    total_loss = 0.0
    total_acc = 0.0

    for _ in range(num_batches):
        x, y = _get_batch(task_name, batch_size, seq_len)
        logits = model(x)

        loss = _cross_entropy_loss(logits, y)
        acc = _compute_accuracy(logits, y)

        mx.eval(loss, acc)

        total_loss += loss.item()
        total_acc += acc.item()

    return {
        "loss": total_loss / num_batches,
        "accuracy": total_acc / num_batches,
    }


# ---------------------------------------------------------------------------
# Model-aware loss wrapper (needed for nn.value_and_grad)
# ---------------------------------------------------------------------------


def _model_loss(model, x, y):
    """Forward pass + cross-entropy loss, for use with nn.value_and_grad."""
    logits = model(x)
    return _cross_entropy_loss(logits, y)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    model,
    task_name,
    batch_size=32,
    seq_len=128,
    n_steps=2000,
    eval_every=200,
    lr=3e-4,
):
    """
    Train a model on a single task.

    Generates a fresh random batch at each step and evaluates on held-out
    batches every *eval_every* steps.

    Args:
        model:      MLX module to train.
        task_name:  One of {'retrieval', 'addition', 'dyck', 'selective_copy'}.
        batch_size: Number of sequences per batch.
        seq_len:    Maximum sequence length.
        n_steps:    Total training steps.
        eval_every: Interval (in steps) between evaluations and logging.
        lr:         Learning rate for AdamW.

    Returns:
        A single dict with the final metrics:
            {step, train_loss, val_loss, accuracy}
    """

    # ---- Optimizer -------------------------------------------------------
    optimizer = opt.AdamW(learning_rate=lr)

    # ---- Value + grad function -------------------------------------------
    # nn.value_and_grad returns a function that computes (loss, gradients)
    # w.r.t. the model parameters.
    loss_and_grad_fn = nn.value_and_grad(model, _model_loss)

    # ---- Bookkeeping -----------------------------------------------------
    train_loss = 0.0
    eval_loss = 0.0
    eval_acc = 0.0

    # ---- Training loop ---------------------------------------------------
    for step in range(1, n_steps + 1):
        # 1.  Fresh random batch
        x, y = _get_batch(task_name, batch_size, seq_len)

        # 2.  Forward + backward
        loss, grads = loss_and_grad_fn(model, x, y)

        # 3.  Force evaluation of the lazy loss & gradients, then update
        mx.eval(loss, grads)
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        # 4.  Bookkeeping
        train_loss = loss.item()

        # 5.  Periodic evaluation & logging
        if step % eval_every == 0 or step == 1:
            metrics = evaluate_model(
                model, task_name,
                num_batches=5,
                batch_size=batch_size,
                seq_len=seq_len,
            )
            eval_loss = metrics["loss"]
            eval_acc = metrics["accuracy"]

            print(
                f"Step {step:5d}/{n_steps} | "
                f"train_loss: {train_loss:.4f} | "
                f"eval_loss: {eval_loss:.4f} | "
                f"eval_acc: {eval_acc:.4f}"
            )

    # ---- Return final metrics --------------------------------------------
    return {
        "step": n_steps,
        "train_loss": train_loss,
        "val_loss": eval_loss,
        "accuracy": eval_acc,
    }
