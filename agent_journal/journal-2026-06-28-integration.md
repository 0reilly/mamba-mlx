# Mamba Experiment — Integration Complete

**Date**: 2026-06-28

## Summary

Integrated all Mamba experiment components into a working pipeline. Verified end-to-end.

## Files

| File | Lines | Status |
|------|-------|--------|
| `models.py` | 698 | ✅ Verified — forward pass, gradients, both model types |
| `tasks.py` | 207 | ✅ Verified — 3 synthetic tasks generate correctly |
| `train.py` | 229 | ✅ Fixed + Verified — training loop runs, loss decreases |
| `run.py` | 60 | ✅ Fixed + Verified — orchestrates experiments |

## Bugs Fixed

1. **`mx.tree_flatten` → `mlx.utils.tree_flatten`** (`run.py`): MLX doesn't expose `tree_flatten` in `mlx.core`. Fixed import + unpacking pattern.

2. **`nn.value_and_grad` mismatch** (`train.py`): `value_and_grad(model, fn)` calls `fn(model, x, y)`, but `_cross_entropy_loss` expected `(logits, targets)`. Added `_model_loss` wrapper that does forward pass + loss.

3. **`if __name__ == "__main__"` guard** (`run.py`): Missing — importing `run` triggered full training. Added guard.

## Verification Results

- **Mamba**: 1,817,860 params — forward pass OK, gradients flow through selective scan (including B_proj, C_proj, dt_proj, A_log, D_param)
- **Transformer**: 4,327,680 params — forward pass OK
- **Training**: 50-step test on mamba/retrieval: loss 15.92→5.58 (decreasing)
- **All tasks**: retrieval, addition, dyck all generate correctly with correct shapes

## Ready to Run

```bash
cd mamba_experiment && python3 run.py
```

This will train transformer then mamba on all 3 tasks (2000 steps each) and save results to `results.json`.
