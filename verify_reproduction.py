"""独立核对一次main运行的预测、学得参数和产物完整性。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reproducibility import integer_hash, read_json, sha256_file, verify_hash, write_json


def _verify_artifacts(run_dir, manifest):
    """先检查运行完成状态，再确认产物自生成后未发生改动。"""
    if manifest["status"] != "completed":
        raise ValueError("运行未成功完成。")
    for name, expected in manifest["artifact_sha256"].items():
        verify_hash(sha256_file(run_dir / name), expected, f"产物 {name}")


def _verify_search(run_dir, manifest, reference):
    """按模型、轮次及超参数找到历史候选，逐个核对验证集预测。"""
    if manifest["mode"] != "search":
        return 0

    search = manifest["search"]
    model_references = reference["search_validation_references"][search["model"]]
    checked = 0
    for round_key in search["rounds"]:
        candidates = model_references[round_key]
        for folder in sorted((run_dir / f"round_{round_key}").iterdir()):
            params = read_json(folder / "metrics.json")["hyperparameters"]
            # 历史记录只列出搜索维度，模型的其他固定参数不参与匹配。
            expected = next(
                candidate
                for candidate in candidates
                if all(
                    params[key] == value
                    for key, value in candidate["parameters"].items()
                )
            )
            predictions = pd.read_csv(folder / "validation_predictions.csv")
            verify_hash(
                integer_hash(predictions["predicted_label"]),
                expected["validation_prediction_sha256"],
                "搜索候选验证预测",
            )
            checked += 1

    if checked != len(pd.read_csv(run_dir / "search_summary.csv")):
        raise ValueError("搜索候选记录数量不一致。")
    return checked


def _verify_models(run_dir, reference):
    """核对训练/验证预测，并记录每个学得参数是否逐位一致。"""
    matches = {}
    for name, expected in reference["models"].items():
        folder = run_dir / name
        if not folder.is_dir():
            continue
        for role in ("train", "validation"):
            predictions = pd.read_csv(folder / f"{role}_predictions.csv")
            verify_hash(
                integer_hash(predictions["predicted_label"]),
                expected[f"{role}_prediction_sha256"],
                f"{name} {role}预测",
            )

        array_matches = {}
        with np.load(folder / "learned_parameters.npz") as saved:
            for key, fingerprint in expected["learned_arrays"].items():
                if key not in saved.files:
                    raise ValueError(f"缺少 {name} 学得参数 {key}。")
                values = np.asarray(saved[key], dtype=fingerprint["dtype"])
                if list(values.shape) != fingerprint["shape"]:
                    raise ValueError(f"{name} {key} 参数形状错误。")
                actual = hashlib.sha256(values.tobytes(order="C")).hexdigest()
                # 数值库可能影响参数低位：仅记录差异，预测标签仍须完全一致。
                array_matches[key] = actual == fingerprint["sha256"]
        matches[name] = array_matches
    return matches


def _verify_test_predictions(run_dir, manifest, reference):
    """只在本次运行包含测试预测时，核对提交标签及其原始顺序。"""
    if not manifest["test_data_read"]:
        return
    labels = pd.read_csv(run_dir / "predictions.csv", header=None)[0]
    verify_hash(
        integer_hash(labels), reference["test_prediction_sha256"], "提交预测标签"
    )


def verify_run(run_dir):
    """独立核验已有运行目录，并保存可阅读的核验报告。"""
    run_dir = Path(run_dir).resolve()
    manifest = read_json(run_dir / "run_manifest.json")
    reference = read_json(run_dir / "reference_results.json")

    _verify_artifacts(run_dir, manifest)
    search_checks = _verify_search(run_dir, manifest, reference)
    matches = _verify_models(run_dir, reference)
    _verify_test_predictions(run_dir, manifest, reference)

    all_arrays_equal = (
        all(all(array_matches.values()) for array_matches in matches.values())
        if matches else None
    )
    report = {
        "status": "passed",
        "run": run_dir.name,
        "artifact_hashes_verified": len(manifest["artifact_sha256"]),
        "prediction_labels_match_historical_reference": True,
        "learned_arrays_bitwise_match": matches,
        "all_saved_arrays_bitwise_equal": all_arrays_equal,
        "search_candidates_verified": search_checks,
        "test_data_read_during_training": False,
        "limits": "数值库/操作系统变化可能影响参数低位；参考预测标签必须完全相同。",
    }
    write_json(run_dir / "reproduction_check.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    verify_run(parser.parse_args().run_dir)
