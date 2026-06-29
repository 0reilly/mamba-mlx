"""Run all Mamba experiments: Transformer vs Mamba(S6) vs LTI-SSM on 3 synthetic tasks."""
import json
import time
import mlx.core as mx
from mlx.utils import tree_flatten
from models import MambaLM, StandardTransformer
from train import train_model

TASKS = ["retrieval", "addition", "dyck"]
MODELS = {
    "transformer": lambda: StandardTransformer(vocab_size=256, d_model=256, n_layers=4),
    "mamba": lambda: MambaLM(vocab_size=256, d_model=256, n_layers=4),
}


def main():
    results = {}
    for model_name, model_fn in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        model = model_fn()
        mx.eval(model)
        n_params = sum(arr.size for _, arr in tree_flatten(model.parameters()))
        print(f"Parameters: {n_params:,}")

        model_results = {}
        for task in TASKS:
            print(f"\n  --- {task} ---")
            t0 = time.time()
            metrics = train_model(model, task, n_steps=2000, eval_every=200)
            elapsed = time.time() - t0
            model_results[task] = {
                "final_train_loss": float(metrics["train_loss"]),
                "final_val_loss": float(metrics["val_loss"]),
                "final_accuracy": float(metrics["accuracy"]),
                "training_time_s": round(elapsed, 1),
            }
            print(f"  Accuracy: {model_results[task]['final_accuracy']:.2%}")

        results[model_name] = model_results

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for model_name, tasks in results.items():
        print(f"\n{model_name}:")
        for task, metrics in tasks.items():
            print(f"  {task}: {metrics['final_accuracy']:.2%} accuracy "
                  f"(loss={metrics['final_val_loss']:.4f}, {metrics['training_time_s']}s)")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results.json")

    return results


if __name__ == "__main__":
    main()
