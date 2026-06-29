"""Run all Mamba experiments: Mamba(S6) vs Transformer on 4 synthetic tasks."""
import json
import time
import mlx.core as mx
from mlx.utils import tree_flatten
from models import MambaLM, StandardTransformer
from train import train_model

TASKS = ["retrieval", "addition", "dyck", "selective_copy"]

# Vocab size per task (models need to be created with the right vocab for the task)
TASK_VOCAB = {
    "retrieval": 256,
    "addition": 22,          # 0-9 digits + special tokens + sum digits
    "dyck": 16,              # open/close paren + special
    "selective_copy": 64,
}

TRAIN_STEPS = 2000
BATCH_SIZE = 32
SEQ_LEN = 128


def main():
    results = {}

    for task in TASKS:
        vocab_size = TASK_VOCAB[task]
        print(f"\n{'='*60}")
        print(f"Task: {task} (vocab={vocab_size})")
        print(f"{'='*60}")

        for model_name, model_cls in [
            ("transformer", StandardTransformer),
            ("mamba", MambaLM),
        ]:
            t0 = time.time()
            model = model_cls(vocab_size=vocab_size, d_model=256, n_layers=4)
            mx.eval(model)
            n_params = sum(arr.size for _, arr in tree_flatten(model.parameters()))

            print(f"\n  [{model_name}] params={n_params:,}")
            metrics = train_model(
                model, task,
                batch_size=BATCH_SIZE,
                seq_len=SEQ_LEN,
                n_steps=TRAIN_STEPS,
            )
            elapsed = time.time() - t0

            results.setdefault(model_name, {})[task] = {
                "final_train_loss": float(metrics["train_loss"]),
                "final_val_loss": float(metrics["val_loss"]),
                "final_accuracy": float(metrics["accuracy"]),
                "training_time_s": round(elapsed, 1),
                "params": n_params,
            }
            print(f"  Accuracy: {metrics['accuracy']:.2%} | Time: {elapsed:.0f}s")

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    header = f"{'Task':<18}"
    for m in results:
        header += f" {m:>12}"
    print(header)
    print("-" * len(header))
    for task in TASKS:
        row = f"{task:<18}"
        for model_name in results:
            acc = results[model_name][task]["final_accuracy"]
            row += f" {acc:>11.2%}"
        print(row)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results.json")

    return results


if __name__ == "__main__":
    main()
