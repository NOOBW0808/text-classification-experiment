"""四模型共用的训练、评估和历史网格搜索。只接收训练/验证数据。"""

from itertools import product
from pathlib import Path
import random
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from nb_weighted_linear import NBWeightedLinearClassifier
from reproducibility import integer_hash, verify_hash, write_json


# 固定十类使各模型、各轮搜索的指标口径一致。
CLASS_LABELS = np.arange(10)
LEARNED_ATTRIBUTES = (
    "coef_", "intercept_", "ratios_", "base_coef_", "class_count_", "n_iter_",
    "feature_log_prob_", "class_log_prior_", "feature_count_",
)


def create_model(name, parameters, config):
    """把超参数与锁定的求解器设置组合起来，尚不学习任何参数。"""
    if name == "multinomial_nb":
        return MultinomialNB(alpha=parameters["alpha"], fit_prior=True, force_alpha=True)

    common = {
        "C": parameters["C"],
        "max_iter": config.get("max_iter_per_model", {}).get(name, config["max_iter"]),
        "tol": config["tol"],
        "random_state": config["seed"],
    }
    if name == "linear_svm":
        return LinearSVC(**common, penalty="l2", loss="squared_hinge", dual="auto", class_weight=None)
    if name == "logistic_regression":
        return LogisticRegression(**common, penalty="l2", solver="lbfgs", class_weight=None)
    if name == "nb_weighted_svm":
        return NBWeightedLinearClassifier(**common, classifier="svm", alpha=parameters["alpha"], use_nb=True)
    raise ValueError(f"未知模型：{name}")


def evaluate(labels, predictions):
    """Macro-F1用于选择超参数；同时保存准确率和其他辅助指标。"""
    return {
        "samples": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=CLASS_LABELS,
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, labels=CLASS_LABELS,
                                      average="weighted", zero_division=0)),
        "correct": int(np.sum(np.asarray(labels) == predictions)),
    }


def _prediction_table(frame, prediction, include_correct=True):
    """保持原始行号，便于不同模型逐样本比较及复现核验。"""
    table = frame[["row_id", "target"]].rename(columns={"target": "true_label"}).copy()
    table["predicted_label"] = prediction
    if include_correct:
        table["correct"] = table["true_label"] == prediction
    return table


def save_evaluation(folder, role, frame, prediction):
    """保存逐行预测、分类报告及两种混淆矩阵，不复制全文。"""
    _prediction_table(frame, prediction).to_csv(
        folder / f"{role}_predictions.csv", index=False, lineterminator="\n"
    )
    report = classification_report(
        frame["target"], prediction, labels=CLASS_LABELS, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).T.to_csv(folder / f"{role}_classification_report.csv", lineterminator="\n")

    counts = confusion_matrix(frame["target"], prediction, labels=CLASS_LABELS)
    columns = [f"predicted_{label}" for label in CLASS_LABELS]
    # 行归一化表示每个真实类别被预测到各类的比例。
    for suffix, values in (
        ("counts", counts),
        ("row_normalized", counts / counts.sum(axis=1, keepdims=True)),
    ):
        pd.DataFrame(values, columns=columns).rename_axis("true_label").to_csv(
            folder / f"{role}_confusion_{suffix}.csv", lineterminator="\n"
        )


def _fit_and_predict(model, matrices, train_labels, seed):
    """固定随机状态，记录拟合耗时和警告；验证数据只参与predict。"""
    train_matrix, validation_matrix = matrices
    random.seed(seed)
    np.random.seed(seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = perf_counter()
        model.fit(train_matrix, train_labels.to_numpy())
        fit_seconds = perf_counter() - started
        train_prediction = model.predict(train_matrix)
        validation_prediction = model.predict(validation_matrix)
    return (train_prediction, validation_prediction), fit_seconds, caught


def _verify_reference(name, metrics, expected):
    """逐标签指纹核对历史运行，不能只靠舍入后的指标判断复现。"""
    for role in ("validation", "train"):
        key = f"{role}_prediction_sha256"
        description = "验证" if role == "validation" else "训练"
        verify_hash(metrics[key], expected[key], f"{name}{description}预测")
    for metric in ("macro_f1", "accuracy"):
        if not np.isclose(metrics["validation"][metric], expected["validation"][metric], rtol=0, atol=1e-12):
            raise ValueError(f"{name}验证{metric}与参考不一致。")
    metrics["reference_predictions_match"] = True


def _save_model_artifacts(model, folder, matrices, frames, predictions):
    """保存可加载模型和可独立检查的学得参数，再核对加载后预测。"""
    joblib.dump(model, folder / "model.joblib")
    learned = {"classes": model.classes_}
    for attribute in LEARNED_ATTRIBUTES:
        if hasattr(model, attribute):
            learned[attribute.removesuffix("_")] = np.asarray(getattr(model, attribute))
    np.savez_compressed(folder / "learned_parameters.npz", **learned)

    for role, frame, prediction in zip(("train", "validation"), frames, predictions):
        save_evaluation(folder, role, frame, prediction)
    reloaded = joblib.load(folder / "model.joblib")
    if not np.array_equal(reloaded.predict(matrices[1]), predictions[1]):
        raise ValueError("模型加载后的验证预测不一致。")


def train_model(name, parameters, matrices, frames, config, output_dir, expected=None, save_model=True):
    """训练一个候选；正式比较保存模型，搜索候选仅保存轻量核验结果。"""
    model = create_model(name, parameters, config)
    train, validation = frames
    predictions, fit_seconds, caught = _fit_and_predict(model, matrices, train["target"], config["seed"])
    train_prediction, validation_prediction = predictions
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    if not converged:
        raise RuntimeError(f"{name} 未收敛，不能把该候选当作可复现的已完成实验。")

    metrics = {
        "model": name,
        "hyperparameters": model.get_params(deep=False),
        "train": evaluate(train["target"], train_prediction),
        "validation": evaluate(validation["target"], validation_prediction),
        "fit_seconds": fit_seconds,
        "converged": converged,
        "warnings": [{"category": item.category.__name__, "message": str(item.message)} for item in caught],
        "train_prediction_sha256": integer_hash(train_prediction),
        "validation_prediction_sha256": integer_hash(validation_prediction),
        "fit_scope": "training_rows_only",
        "test_data_read": False,
    }
    if expected:
        _verify_reference(name, metrics, expected)

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "metrics.json", metrics)
    if save_model:
        _save_model_artifacts(model, folder, matrices, frames, predictions)
    else:
        # 90个候选共用输入；不为每个候选保存数MB的模型和特征副本。
        _prediction_table(validation, validation_prediction, include_correct=False).to_csv(
            folder / "validation_predictions.csv", index=False, lineterminator="\n"
        )
    print(f"{name} {parameters}: validation Macro-F1={metrics['validation']['macro_f1']:.9f}, "
          f"Accuracy={metrics['validation']['accuracy']:.9f}", flush=True)
    return model, metrics


def search_parameters(name, round_name, matrices, frames, config, output_dir, reference):
    """重跑已记录的网格，不改写最终参数，也不接触测试集。"""
    rounds = config["search_grids"][name]
    chosen = sorted(rounds) if round_name == "all" else [round_name]
    if any(key not in rounds for key in chosen):
        raise ValueError("不存在该调参轮次。")

    records = []
    for round_key in chosen:
        grid = rounds[round_key]
        keys = list(grid)
        candidates = reference["search_validation_references"][name][round_key]
        # 按配置中的键和值顺序生成候选，保持历史运行编号不变。
        for index, combination in enumerate(product(*(grid[key] for key in keys))):
            parameters = dict(zip(keys, combination))
            expected = next(row for row in candidates if row["parameters"] == parameters)
            folder = Path(output_dir) / f"round_{round_key}" / f"candidate_{index:03d}"
            _, metrics = train_model(
                name, parameters, matrices, frames, config, folder, expected=expected, save_model=False
            )
            records.append({
                "model": name,
                "round": round_key,
                **parameters,
                "train_macro_f1": metrics["train"]["macro_f1"],
                "validation_macro_f1": metrics["validation"]["macro_f1"],
                "validation_accuracy": metrics["validation"]["accuracy"],
                "converged": metrics["converged"],
            })

    summary = pd.DataFrame(records).sort_values(["validation_macro_f1", "validation_accuracy"], ascending=False)
    summary.to_csv(Path(output_dir) / "search_summary.csv", index=False, lineterminator="\n")
    return {
        "candidate_runs": len(records),
        "model": name,
        "rounds": chosen,
        "test_data_read": False,
        "final_parameters_modified": False,
    }

