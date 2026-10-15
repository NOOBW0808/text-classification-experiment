"""仅训练文本拟合 TF-IDF；验证和测试共用固定的映射函数。"""

import hashlib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from reproducibility import json_hash, verify_hash


def transform_texts(vectorizer, texts, role):
    """只调用 transform，并核验映射没有改变训练词表或 IDF。"""
    vocabulary = dict(vectorizer.vocabulary_)
    idf = vectorizer.idf_.copy()
    matrix = vectorizer.transform(texts).tocsr()
    if vocabulary != vectorizer.vocabulary_ or not np.array_equal(idf, vectorizer.idf_):
        raise ValueError(f"{role}映射改变了词表或IDF。")
    return matrix


def build_features(train, validation, config, reference):
    """返回已拟合向量器、训练矩阵与验证矩阵，并核对历史特征指纹。"""
    parameters = dict(config["tfidf"])
    parameters["ngram_range"] = tuple(parameters["ngram_range"])
    parameters["dtype"] = np.float64
    vectorizer = TfidfVectorizer(**parameters)

    # 词表和 IDF 只来自训练文本；验证信息不能参与拟合。
    train_matrix = vectorizer.fit_transform(train["text"]).tocsr()
    validation_matrix = transform_texts(vectorizer, validation["text"], "验证")
    feature_names = vectorizer.get_feature_names_out().tolist()
    idf_bytes = np.asarray(vectorizer.idf_, dtype="<f8").tobytes()
    verify_hash(json_hash(feature_names), reference["feature_names_sha256"], "TF-IDF有序特征名")
    verify_hash(hashlib.sha256(idf_bytes).hexdigest(), reference["idf_sha256"], "训练集IDF")

    for matrix in (train_matrix, validation_matrix):
        if matrix.shape[1] != config["tfidf"]["max_features"]:
            raise ValueError("TF-IDF特征维度错误。")
        if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
            raise ValueError("TF-IDF必须是有限的非负数值。")
    return vectorizer, train_matrix, validation_matrix
