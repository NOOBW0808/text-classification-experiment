"""固定分层划分、无状态清洗与重叠隔离；数据仅保存在内存中。"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from reproducibility import integer_hash, json_hash, sha256_file, verify_hash
from text_cleaning import CLEANING_RULES, clean_with_trace


def load_labeled_data(data_dir, config):
    """先核验原始数据身份，再检查列、样本数、类别与空文本。"""
    source = Path(data_dir) / "train_data.csv"
    dataset = config["dataset"]
    verify_hash(sha256_file(source), dataset["train_sha256"], "原始训练CSV")
    frame = pd.read_csv(source, dtype={"text": str}, keep_default_na=False)

    if not {"text", "target"}.issubset(frame.columns):
        raise ValueError("训练数据必须包含 text 和 target 列。")
    if len(frame) != dataset["labeled_rows"]:
        raise ValueError("训练样本数不符合固定实验配置。")
    valid_labels = (
        pd.api.types.is_integer_dtype(frame["target"])
        and set(frame["target"]) == set(range(10))
    )
    if not valid_labels or frame["text"].str.strip().eq("").any():
        raise ValueError("训练数据的类别或文本不符合固定实验配置。")

    # 原始行号贯穿划分、评估和预测，便于追溯每条样本。
    frame.insert(0, "row_id", np.arange(len(frame)))
    return frame


def split_row_ids(frame, config):
    """固定种子分层抽样，再恢复原始行序以保持历史结果一致。"""
    train_ids, validation_ids = train_test_split(
        frame.index.to_numpy(),
        test_size=config["validation_ratio"],
        random_state=config["seed"],
        stratify=frame["target"],
    )
    train_ids, validation_ids = np.sort(train_ids), np.sort(validation_ids)
    verify_hash(integer_hash(train_ids), config["split"]["train_ids_sha256"], "训练行号")
    verify_hash(
        integer_hash(validation_ids),
        config["split"]["initial_validation_ids_sha256"],
        "初始验证行号",
    )
    return train_ids, validation_ids


def clean_frame(frame):
    """应用相同的固定规则并汇总操作次数；不学习任何数据统计量。"""
    cleaned_texts = []
    operation_counts = Counter()
    for text in frame["text"]:
        cleaned, trace = clean_with_trace(text)
        cleaned_texts.append(cleaned)
        operation_counts.update({name: int(value) for name, value in trace.items()})
    if any(not text for text in cleaned_texts):
        raise ValueError("清洗出现空文本；不静默删除或改写样本。")
    frame["text"] = cleaned_texts
    return dict(operation_counts)


def exclude_validation_overlap(train, initial_validation, split_config):
    """仅隔离已冻结的验证重复行；这些行不能被加入训练集。"""
    overlap_mask = initial_validation["text"].isin(set(train["text"]))
    overlap_ids = initial_validation.loc[overlap_mask, "row_id"].tolist()
    excluded_ids = split_config["excluded_validation_row_ids"]
    if sorted(overlap_ids) != sorted(excluded_ids):
        raise ValueError("清洗后的重叠隔离名单变化，不能声称复现了原始实验。")

    keep_mask = ~initial_validation["row_id"].isin(excluded_ids)
    validation = initial_validation.loc[keep_mask].reset_index(drop=True)
    if set(train["text"]).intersection(validation["text"]):
        raise ValueError("训练与有效验证存在重复清洗全文。")
    verify_hash(
        integer_hash(validation["row_id"]),
        split_config["validation_ids_sha256"],
        "有效验证行号",
    )
    return validation


def prepare_data(data_dir, config):
    """返回训练集、有效验证集、行号划分表及清洗报告，不读取测试集。"""
    frame = load_labeled_data(data_dir, config)
    train_ids, validation_ids = split_row_ids(frame, config)
    operation_counts = clean_frame(frame)
    train = frame.loc[train_ids].reset_index(drop=True)
    initial_validation = frame.loc[validation_ids].reset_index(drop=True)
    validation = exclude_validation_overlap(train, initial_validation, config["split"])

    for role, subset in (("train", train), ("validation", validation)):
        verify_hash(
            json_hash(subset["text"].tolist()),
            config["split"][f"{role}_texts_sha256"],
            f"{role}清洗文本",
        )

    excluded_ids = config["split"]["excluded_validation_row_ids"]
    assignment = frame[["row_id", "target"]].copy()
    assignment["split"] = "validation"
    assignment.loc[assignment["row_id"].isin(train_ids), "split"] = "train"
    assignment.loc[assignment["row_id"].isin(excluded_ids), "split"] = "validation_overlap_excluded"
    report = {
        "train_rows": len(train),
        "initial_validation_rows": len(initial_validation),
        "validation_rows": len(validation),
        "excluded_validation_row_ids": excluded_ids,
        "seed": config["seed"],
        "initial_validation_ratio": config["validation_ratio"],
        "method": "stratified_seeded_split_then_frozen_exact_overlap_quarantine",
        "cleaning_rules": CLEANING_RULES,
        "cleaning_operations": operation_counts,
        "fit_scope": "rules_are_stateless",
        "test_data_read": False,
    }
    return train, validation, assignment, report
