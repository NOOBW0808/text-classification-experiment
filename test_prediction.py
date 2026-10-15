"""只在最终参数已锁定后读测试集，复用相同清洗及固定模型。"""

from pathlib import Path

import numpy as np
import pandas as pd

from reproducibility import integer_hash, sha256_file, verify_hash
from text_cleaning import clean_text
from text_features import transform_texts


def predict_test(data_dir, vectorizer, model, output_dir, config, reference):
    """保持原始行序，生成提交文件与可追溯预测表；无标签则不计算指标。"""
    path = Path(data_dir) / "test_data_unlabeled.csv"
    verify_hash(sha256_file(path), config["dataset"]["test_sha256"], "原始无标签测试CSV")
    frame = pd.read_csv(path, dtype={"text": str}, keep_default_na=False)
    if "text" not in frame or {"target", "label"}.intersection(frame.columns):
        raise ValueError("测试输入格式不符合无标签实验数据。")
    if len(frame) != config["dataset"]["test_rows"] or frame["text"].str.strip().eq("").any():
        raise ValueError("测试样本数或文本不符合固定实验配置。")
    texts = frame["text"].map(clean_text)
    if texts.eq("").any():
        raise ValueError("测试清洗后出现空文本，不静默删除行。")
    # 复用训练向量器和模型，禁止在测试文本上 fit 或 fit_transform。
    matrix = transform_texts(vectorizer, texts, "测试")
    prediction = model.predict(matrix)
    prediction_hash = integer_hash(prediction)
    verify_hash(prediction_hash, reference["test_prediction_sha256"], "测试预测标签")
    # 一行一个标签，与原draft_main.py要求一致；固定LF换行利于跨平台比较。
    pd.Series(prediction).to_csv(output_dir / "predictions.csv", index=False, header=False,
                                 lineterminator="\n")
    pd.DataFrame({"row_id": np.arange(len(prediction)), "predicted_label": prediction}).to_csv(
        output_dir / "test_predictions.csv", index=False, lineterminator="\n")
    return {
        "rows": len(prediction),
        "prediction_sha256": prediction_hash,
        "zero_feature_rows": int((matrix.getnnz(axis=1) == 0).sum()),
        "row_order_preserved": True,
        "labels_available": False,
        "test_metrics": None,
        "model_refitted": False,
        "vectorizer_refitted": False,
        "nb_weights_refitted": False,
    }
