"""实验一主入口：准备数据 → 提取特征 → 训练/搜索 → 最终测试预测。"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from data_preparation import prepare_data
from model_training import search_parameters, train_model
from reproducibility import (
    BASE_DIR, environment, read_json, sha256_file, source_hashes, write_json,
)
from test_prediction import predict_test
from text_features import build_features


# 默认路径相对于源码；在 PyCharm 中直接运行也不依赖工作目录。
DATA_DIR = BASE_DIR / "data"
DEFAULT_MODE = "all"
FINAL_MODEL = "nb_weighted_svm"
MODEL_NAMES = ("multinomial_nb", "logistic_regression", "linear_svm", FINAL_MODEL)


def parse_args():
    """四种模式共用一个入口；模型与轮次选项仅用于历史搜索。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "final", "compare", "search"),
                        default=DEFAULT_MODE)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", choices=MODEL_NAMES, default=FINAL_MODEL,
                        help="只用于 search 模式")
    parser.add_argument("--round", choices=("1", "2", "3", "all"), default="3",
                        help="只用于 search 模式")
    return parser.parse_args()


def train_selected_models(mode, matrices, frames, config, reference, output):
    """在相同特征上训练锁定参数的模型，保存比较表并返回最终模型。"""
    model_names = [FINAL_MODEL] if mode == "final" else config["best_parameters"]
    comparison = []
    final_model = None
    for name in model_names:
        parameters = config["best_parameters"][name]
        model, metrics = train_model(
            name, parameters, matrices, frames, config,
            output / name, reference["models"][name],
        )
        validation_metrics = metrics["validation"]
        comparison.append({
            "model": name,
            **parameters,
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_accuracy": validation_metrics["accuracy"],
            "correct": validation_metrics["correct"],
            "samples": validation_metrics["samples"],
        })
        if name == FINAL_MODEL:
            final_model = model
    pd.DataFrame(comparison).to_csv(
        output / "model_comparison.csv", index=False, lineterminator="\n",
    )
    return final_model


def run_pipeline(args, config, reference, output, manifest):
    """组织实验阶段；搜索和比较模式始终不读取测试集。"""
    train, validation, assignment, preparation = prepare_data(args.data_dir, config)
    excluded_count = len(preparation["excluded_validation_row_ids"])
    print(f"Fixed split: train={len(train)}, validation={len(validation)}, "
          f"quarantined={excluded_count}", flush=True)
    assignment.to_csv(output / "split_assignments.csv", index=False, lineterminator="\n")
    write_json(output / "preparation_report.json", preparation)

    vectorizer, train_matrix, validation_matrix = build_features(
        train, validation, config, reference,
    )
    print(f"TF-IDF: train={train_matrix.shape}, validation={validation_matrix.shape}",
          flush=True)
    matrices = (train_matrix, validation_matrix)
    frames = (train, validation)

    if args.mode == "search":
        manifest["search"] = search_parameters(
            args.model, args.round, matrices, frames, config, output, reference,
        )
        return

    final_model = train_selected_models(args.mode, matrices, frames, config, reference, output)
    joblib.dump(vectorizer, output / "tfidf_vectorizer.joblib")
    if args.mode in ("all", "final"):
        # 最终参数已锁定。这里仅预测，不合并验证集重训，也不重新拟合特征。
        manifest["test_data_read"] = True
        manifest["test_prediction"] = predict_test(
            args.data_dir, vectorizer, final_model, output, config, reference,
        )


def save_completed_run(output, manifest):
    """保存配置快照和产物指纹，让输出可独立核验。"""
    for filename in ("experiment_config.json", "reference_results.json", "requirements.txt"):
        shutil.copy2(BASE_DIR / filename, output / filename)
    manifest.update(status="completed", reference_predictions_match=True)
    manifest["artifact_sha256"] = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    }
    write_json(output / "run_manifest.json", manifest)


def main():
    args = parse_args()
    config = read_json(BASE_DIR / "experiment_config.json")
    reference = read_json(BASE_DIR / "reference_results.json")
    runtime = environment(config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output_dir or BASE_DIR / "outputs" / f"{args.mode}_{stamp}").resolve()
    # 只允许新输出目录，防止覆盖上一轮实验结果。
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "mode": args.mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config,
        "environment": runtime,
        "source_sha256": source_hashes(),
        "invocation": sys.argv,
        "test_data_read": False,
        "selection_status": "user_locked_before_test",
        "source_data_dir": str(args.data_dir.resolve()),
    }
    write_json(output / "run_manifest.json", manifest)
    try:
        with threadpool_limits(limits=config["thread_limit"]):
            run_pipeline(args, config, reference, output, manifest)
            save_completed_run(output, manifest)
    except Exception as error:
        # 失败也留下原因；独立核验不会将这种运行计为完成。
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "run_manifest.json", manifest)
        raise

    print(f"Complete: {output}", flush=True)
    if args.mode in ("all", "final"):
        print(f"Submission: {output / 'predictions.csv'}; test has no labels, no test metrics.",
              flush=True)


if __name__ == "__main__":
    main()
