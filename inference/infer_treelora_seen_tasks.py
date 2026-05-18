#!/usr/bin/env python
"""
Standalone seen-task inference for saved Tree_LoRA checkpoints.

Checkpoint i is loaded from CHECKPOINT_PATH/i and evaluated on all seen tasks
0..i. This is intended for forgetting/BWT reporting after continual training.
"""

import argparse
import gc
import json
import math
import os
import sys
from typing import Dict, List

sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from evaluator.compute_metrics import DATASET_TO_OUTPUT_LANG, compute_metrics
from utils.data.data_collator import DataCollator
from utils.data.data_utils import create_prompt_dataset
from utils.model.model_utils import create_hf_model
from utils.my_peft import PeftModel
from utils.utils import load_hf_tokenizer, set_random_seed, to_device


CODETASK_TASKS = [
    "CONCODE",
    "CodeTrans",
    "CodeSearchNet",
    "BFP",
    "KodCode",
    "RunBugRun",
    "TheVault_Csharp",
    "CoST",
]

# Same order as CODETASK_TASKS.
DEFAULT_MAX_PROMPT_LENS = [320, 320, 256, 130, 512, 256, 256, 256]
DEFAULT_MAX_ANS_LENS = [150, 256, 128, 120, 300, 128, 128, 128]

CODE_METRICS_WITHOUT_CODEBLEU = {
    "CodeSearchNet",
    "CoST",
    "KodCode",
    "RunBugRun",
    "TheVault_Csharp",
}


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def resolve_int_list(value: str, task_count: int, name: str) -> List[int]:
    values = parse_csv(value)
    if len(values) == 1:
        return [int(values[0])] * task_count
    if len(values) != task_count:
        raise ValueError(f"{name} expects either 1 value or {task_count} values, got {len(values)}")
    return [int(item) for item in values]


def task_metric_name(task: str) -> str:
    # compute_metrics returns "bleu" for these generation/code tasks.
    return "bleu"


def metric_to_float(metrics: Dict, metric_name: str) -> float:
    value = metrics[metric_name]
    if isinstance(value, (list, tuple)):
        value = value[0]
    if isinstance(value, dict):
        value = next(iter(value.values()))
    return float(value)


def save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_checkpoint_dir(args, task_id: int) -> str:
    if args.checkpoint is not None:
        return args.checkpoint
    return os.path.join(args.checkpoint_path, str(task_id))


def load_tokenizer_for_checkpoint(checkpoint: str, args):
    tokenizer_config = os.path.join(checkpoint, "tokenizer_config.json")
    tokenizer_path = checkpoint if os.path.isfile(tokenizer_config) else args.model_name_or_path
    tokenizer = load_hf_tokenizer(tokenizer_path, fast_tokenizer=True)
    assert tokenizer.padding_side == "left"
    assert tokenizer.truncation_side == "left"
    return tokenizer


def load_treelora_model(checkpoint: str, tokenizer, args, dtype, device):
    adapter_config = os.path.join(checkpoint, "adapter_config.json")
    adapter_model = os.path.join(checkpoint, "adapter_model.bin")
    if not os.path.isfile(adapter_config) or not os.path.isfile(adapter_model):
        raise FileNotFoundError(
            f"Tree_LoRA checkpoint must contain adapter_config.json and adapter_model.bin: {checkpoint}"
        )

    print(f"Loading base model: {args.model_name_or_path}")
    model = create_hf_model(
        AutoModelForCausalLM,
        args.model_name_or_path,
        tokenizer,
        ds_config=None,
    )

    print(f"Loading Tree_LoRA adapter: {checkpoint}")
    model = PeftModel.from_pretrained(model, checkpoint)
    for _, param in model.named_parameters():
        param.requires_grad = False

    if dtype is not None:
        model.to(dtype=dtype, device=device)
    else:
        model.to(device)
    model.eval()
    return model


def build_seen_task_loaders(tokenizer, args, tasks, max_prompt_lens, max_ans_lens, seen_task_count):
    if seen_task_count < 1 or seen_task_count > len(tasks):
        raise ValueError(f"seen_task_count must be in [1, {len(tasks)}], got {seen_task_count}")

    test_loaders = {}
    for task_id, task in enumerate(tasks[:seen_task_count]):
        dataset_path = os.path.join(args.data_path, task)
        _, _, test_dataset = create_prompt_dataset(
            local_rank=-1,
            data_path=dataset_path,
            output_path=args.data_output_path,
            seed=args.seed,
            distributed=False,
            num_train=-1,
            num_eval=-1,
            num_test=args.num_test,
        )
        collator = DataCollator(
            tokenizer,
            padding="longest",
            max_prompt_len=max_prompt_lens[task_id],
            max_ans_len=max_ans_lens[task_id],
            pad_to_multiple_of=8,
            inference=True,
            task=task,
        )
        test_loaders[task] = DataLoader(
            test_dataset,
            collate_fn=collator,
            sampler=SequentialSampler(test_dataset),
            batch_size=args.per_device_eval_batch_size,
        )
        print(f"  [{task_id}] {task}: {len(test_dataset)} test examples")
    return test_loaders


def generate_predictions(model, tokenizer, dataloader, max_ans_len, device, args):
    sources, predictions, ground_truths = [], [], []
    progress = tqdm(total=len(dataloader), leave=True)
    for step, batch in enumerate(dataloader):
        sources.extend(batch["sources"])
        ground_truths.extend(batch["gts"])
        del batch["sources"]
        del batch["gts"]

        batch = to_device(batch, device)
        prompt_len = batch["input_ids"].shape[1]
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        generate_kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "max_new_tokens": int(max_ans_len),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": pad_token_id,
            "do_sample": args.do_sample,
            "num_return_sequences": 1,
            "use_cache": True,
        }
        if args.do_sample:
            generate_kwargs["temperature"] = args.temperature

        with torch.no_grad():
            generated_ids = model.generate(**generate_kwargs)

        predictions.extend(
            tokenizer.batch_decode(
                generated_ids[:, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
        progress.update(1)
        progress.set_description(f"Step {step}", refresh=False)
    progress.close()
    return sources, predictions, ground_truths


def evaluate_task(task, sources, predictions, ground_truths):
    calc_codebleu = task not in CODE_METRICS_WITHOUT_CODEBLEU
    metrics = compute_metrics(
        predictions,
        ground_truths,
        calc_codebleu=calc_codebleu,
        language=DATASET_TO_OUTPUT_LANG.get(task, None),
    )
    # Keep compatibility with old collect_results.py configs that expect bleu-1.
    if "bleu" in metrics and "bleu-1" not in metrics:
        metrics["bleu-1"] = metrics["bleu"]
    return metrics


def evaluate_checkpoint(task_id, checkpoint, output_dir, tasks, max_prompt_lens, max_ans_lens, base_args, dtype, device):
    seen_task_count = base_args.seen_task_count or task_id + 1
    print(f"\n***** Evaluating checkpoint {task_id}: {checkpoint} on seen tasks 0..{seen_task_count - 1} *****")

    tokenizer = load_tokenizer_for_checkpoint(checkpoint, base_args)
    model = load_treelora_model(checkpoint, tokenizer, base_args, dtype, device)
    test_loaders = build_seen_task_loaders(
        tokenizer,
        base_args,
        tasks,
        max_prompt_lens,
        max_ans_lens,
        seen_task_count,
    )

    checkpoint_metrics = {}
    checkpoint_scores = {}
    for eval_task_id, task in enumerate(tasks[:seen_task_count]):
        print(f"***** Inference checkpoint {task_id} on task {eval_task_id}: {task} *****")
        sources, predictions, ground_truths = generate_predictions(
            model,
            tokenizer,
            test_loaders[task],
            max_ans_lens[eval_task_id],
            device,
            base_args,
        )
        metrics = evaluate_task(task, sources, predictions, ground_truths)
        primary_metric = task_metric_name(task)
        primary_score = metric_to_float(metrics, primary_metric)

        checkpoint_metrics[task] = metrics
        checkpoint_scores[task] = primary_score
        rows = [
            {"source": source, "ground-truth": gt, "prediction": pred}
            for source, gt, pred in zip(sources, ground_truths, predictions)
        ]

        result_name = f"results-{task_id}-{eval_task_id}-{task}"
        save_json(
            os.path.join(output_dir, f"{result_name}.json"),
            {
                "eval": metrics,
                "prompts": sources,
                "results": predictions,
                "labels": ground_truths,
                "primary_metric": primary_metric,
                "primary_score": primary_score,
            },
        )
        save_jsonl(os.path.join(output_dir, f"predictions-{task_id}-{eval_task_id}-{task}.jsonl"), rows)
        print(f"{task}: {primary_metric}={primary_score:.4f}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return checkpoint_metrics, checkpoint_scores


def compute_report(tasks, score_matrix):
    task_count = len(tasks)
    valid_cols = [col for col in range(task_count) if not np.isnan(score_matrix[:, col]).all()]
    final_col = valid_cols[-1] if valid_cols else task_count - 1
    summary = {
        "final_average": float(np.nanmean(score_matrix[:, final_col])),
        "all_seen_average": float(np.nanmean(score_matrix)),
    }

    forgetting = []
    bwt = []
    for task_id in range(final_col):
        best_before_final = float(np.nanmax(score_matrix[task_id, task_id:final_col]))
        final_score = float(score_matrix[task_id, final_col])
        forgetting.append(max(best_before_final - final_score, 0.0))
        bwt.append(final_score - float(score_matrix[task_id, task_id]))

    summary["average_forgetting"] = float(np.mean(forgetting)) if forgetting else 0.0
    summary["bwt"] = float(np.mean(bwt)) if bwt else 0.0
    summary["negative_bwt_only"] = float(np.mean([min(value, 0.0) for value in bwt])) if bwt else 0.0
    return summary


def write_report(output_dir, tasks, score_matrix, summary):
    lines = ["Tree_LoRA seen-task inference matrix", ""]
    lines.append("\t".join(["task/checkpoint"] + [f"ckpt{i}:{task}" for i, task in enumerate(tasks)]))
    for row_id, task in enumerate(tasks):
        row = [task]
        for col_id in range(len(tasks)):
            value = score_matrix[row_id, col_id]
            row.append("" if math.isnan(value) else f"{value:.4f}")
        lines.append("\t".join(row))
    lines.append("")
    for key, value in summary.items():
        lines.append(f"{key}: {value:.4f}")

    with open(os.path.join(output_dir, "forgetting_report.txt"), "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="Single checkpoint dir, for example .../7.")
    parser.add_argument("--checkpoint_path", default=None, help="Root dir containing checkpoint folders 0..7.")
    parser.add_argument("--start_task_id", type=int, default=0)
    parser.add_argument("--end_task_id", type=int, default=7)
    parser.add_argument("--data_path", default="CODETASK_HF")
    parser.add_argument("--data_output_path", default="./tmp/data_files/")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-Coder-1.5B")
    parser.add_argument("--inference_tasks", default=",".join(CODETASK_TASKS))
    parser.add_argument("--max_prompt_len", default=",".join(str(x) for x in DEFAULT_MAX_PROMPT_LENS))
    parser.add_argument("--max_ans_len", default=",".join(str(x) for x in DEFAULT_MAX_ANS_LENS))
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--num_test", type=int, default=-1, help="-1 means full test set.")
    parser.add_argument("--seen_task_count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    if args.checkpoint is None and args.checkpoint_path is None:
        parser.error("one of --checkpoint or --checkpoint_path is required")
    if args.checkpoint is not None and args.checkpoint_path is not None:
        parser.error("use either --checkpoint or --checkpoint_path, not both")
    return args


def resolve_dtype(dtype_arg):
    if dtype_arg == "fp32" or not torch.cuda.is_available():
        return torch.float32
    if dtype_arg == "fp16":
        return torch.float16
    if dtype_arg == "bf16":
        return torch.bfloat16
    return torch.float16


def main():
    args = parse_args()
    set_random_seed(args.seed)

    tasks = parse_csv(args.inference_tasks)
    max_prompt_lens = resolve_int_list(args.max_prompt_len, len(tasks), "max_prompt_len")
    max_ans_lens = resolve_int_list(args.max_ans_len, len(tasks), "max_ans_len")
    dtype = resolve_dtype(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint_path is not None:
        output_dir = args.output_dir or os.path.join(os.path.abspath(args.checkpoint_path), "seen_task_eval")
        first_task_id = args.start_task_id
        last_task_id = args.end_task_id
    else:
        output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "seen_task_eval")
        checkpoint_name = os.path.basename(os.path.normpath(args.checkpoint))
        inferred_task_id = int(checkpoint_name) if checkpoint_name.isdigit() else args.start_task_id
        if args.seen_task_count is None:
            args.seen_task_count = inferred_task_id + 1
        first_task_id = inferred_task_id
        last_task_id = inferred_task_id

    os.makedirs(output_dir, exist_ok=True)
    score_matrix = np.full((len(tasks), len(tasks)), np.nan, dtype=np.float64)
    all_metrics = {}

    for task_id in range(first_task_id, last_task_id + 1):
        checkpoint = get_checkpoint_dir(args, task_id)
        if not os.path.isdir(checkpoint):
            print(f"[WARN] skipping missing checkpoint: {checkpoint}")
            continue

        checkpoint_metrics, checkpoint_scores = evaluate_checkpoint(
            task_id=task_id,
            checkpoint=checkpoint,
            output_dir=output_dir,
            tasks=tasks,
            max_prompt_lens=max_prompt_lens,
            max_ans_lens=max_ans_lens,
            base_args=args,
            dtype=dtype,
            device=device,
        )
        all_metrics[str(task_id)] = checkpoint_metrics
        for task, score in checkpoint_scores.items():
            score_matrix[tasks.index(task), task_id] = score

    summary = compute_report(tasks, score_matrix)
    save_json(os.path.join(output_dir, "metrics_summary.json"), all_metrics)
    save_json(
        os.path.join(output_dir, "score_matrix.json"),
        {"tasks": tasks, "primary_metric": "bleu", "matrix": score_matrix.tolist(), "summary": summary},
    )
    write_report(output_dir, tasks, score_matrix, summary)

    print("\nFinished Tree_LoRA seen-task inference.")
    print(f"Saved outputs to: {output_dir}")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
