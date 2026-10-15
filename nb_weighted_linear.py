"""NB加权线性分类器：训练集估计词项权重，再训练十个“本类对其他类”模型。

这是受Wang和Manning（2012）启发的多类别变体，不是原论文的完整复现。
分类使用未校准的决策分数，分数不能直接解释为类别概率。
"""

from numbers import Integral, Real

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_array, check_is_fitted


class NBWeightedLinearClassifier(ClassifierMixin, BaseEstimator):
    """接受非负稀疏TF-IDF，类别固定为0～9。

    alpha控制NB平滑，C控制线性分类器的正则化强度；它们是超参数。
    ratios_与base_coef_是仅由fit中的训练样本学习的参数。
    coef_ = base_coef_ * ratios_，用于直接计算最终决策分数。
    use_nb=False保留普通OVR对照；classifier支持SVM或逻辑回归。
    """

    def __init__(self, classifier="svm", C=2.0, alpha=1.0, max_iter=5000,
                 tol=1e-4, random_state=42, use_nb=True):
        self.classifier = classifier
        self.C = C
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.use_nb = use_nb

    def _validate_hyperparameters(self):
        """拒绝非法设置，避免得到不可解释或不能复现的训练结果。"""
        if self.classifier not in ("svm", "logreg"):
            raise ValueError("classifier must be 'svm' or 'logreg'.")
        if not isinstance(self.use_nb, (bool, np.bool_)):
            raise ValueError("use_nb must be a boolean.")
        for name in ("C", "alpha", "tol"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be a finite positive real number.")
        if (isinstance(self.max_iter, (bool, np.bool_))
                or not isinstance(self.max_iter, Integral) or self.max_iter <= 0):
            raise ValueError("max_iter must be a positive integer.")
        if (isinstance(self.random_state, (bool, np.bool_))
                or not isinstance(self.random_state, Integral)
                or not 0 <= self.random_state <= 2**32 - 1):
            raise ValueError("random_state must be an integer in [0, 2**32-1].")

    @staticmethod
    def _validate_features(X):
        """规范稀疏矩阵副本，不修改调用方传入的TF-IDF。"""
        if not sparse.issparse(X):
            raise ValueError("X must be a sparse nonnegative TF-IDF matrix.")
        checked = check_array(X, accept_sparse="csr", dtype=np.float64,
                              ensure_all_finite=True)
        if (checked.data < 0).any():
            raise ValueError("X must contain only nonnegative TF-IDF values.")
        # 合并重复项并删除显式零，保证“词项出现一次”的计数不被重复存储影响。
        checked = sparse.csr_matrix(checked, copy=True)
        checked.sum_duplicates()
        checked.eliminate_zeros()
        if not np.isfinite(checked.data).all():
            raise ValueError("Canonical sparse feature values must be finite.")
        return checked

    def _count_ratio(self, positive_counts, negative_counts, feature_count):
        """计算本类/其他类的平滑log-count-ratio，保留历史运算顺序。"""
        # 分母为整张词表的出现计数之和，不是正类或负类的文档数量。
        return (
            np.log((positive_counts + self.alpha)
                   / (positive_counts.sum() + self.alpha * feature_count))
            - np.log((negative_counts + self.alpha)
                     / (negative_counts.sum() + self.alpha * feature_count))
        )

    def _create_binary_model(self):
        """各类别共用求解器设置，分别学习自己的二分类系数。"""
        common = {
            "C": self.C,
            "penalty": "l2",
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
        }
        if self.classifier == "svm":
            return LinearSVC(**common, loss="squared_hinge", dual="auto")
        return LogisticRegression(**common, solver="lbfgs")

    def fit(self, X, y):
        """先由训练集估计NB权重，再逐类别拟合二分类器。"""
        self._validate_hyperparameters()
        features = self._validate_features(X)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != features.shape[0]:
            raise ValueError("y must be one-dimensional and match the X rows.")
        if labels.dtype.kind not in "iu":
            raise ValueError("y must contain integer class labels 0 through 9.")
        classes = np.unique(labels)
        if not np.array_equal(classes, np.arange(10)):
            raise ValueError("Training y must contain all and only classes 0 through 9.")

        feature_count = features.shape[1]
        if self.use_nb:
            # 估计NB权重只关心词项是否出现；线性模型仍使用原始TF-IDF数值。
            presence = features.copy()
            presence.data.fill(1.0)
            total_counts = np.asarray(presence.sum(axis=0)).ravel()
        ratios, estimators, base_coefficients = [], [], []
        intercepts, iterations, class_counts = [], [], []
        for label in classes:
            positive = labels == label
            if self.use_nb:
                positive_counts = np.asarray(presence[positive].sum(axis=0)).ravel()
                negative_counts = total_counts - positive_counts
                ratio = self._count_ratio(positive_counts, negative_counts, feature_count)
            else:
                ratio = np.ones(feature_count, dtype=np.float64)
            if not np.isfinite(ratio).all():
                raise ValueError("NB ratios are not finite; check alpha and input scale.")
            # 加权后不再L2归一化；额外归一化会改变已锁定的实验模型。
            weighted = features.multiply(ratio).tocsr()
            if not np.isfinite(weighted.data).all():
                raise ValueError("NB-weighted features must be finite.")
            estimator = self._create_binary_model()
            estimator.fit(weighted, positive.astype(np.int8))
            if (not np.isfinite(estimator.coef_).all()
                    or not np.isfinite(estimator.intercept_).all()):
                raise ValueError("Fitted linear coefficients must be finite.")
            ratios.append(ratio)
            estimators.append(estimator)
            base_coefficients.append(estimator.coef_[0].copy())
            intercepts.append(float(estimator.intercept_[0]))
            iterations.append(int(np.asarray(estimator.n_iter_).max()))
            class_counts.append(int(positive.sum()))

        self.classes_ = classes.copy()
        self.use_nb_ = bool(self.use_nb)
        self.ratios_ = np.vstack(ratios)
        self.estimators_ = estimators
        self.base_coef_ = np.vstack(base_coefficients)
        # 把NB权重并入有效系数，预测时无需逐类重建加权特征矩阵。
        self.coef_ = self.base_coef_ * self.ratios_
        self.intercept_ = np.asarray(intercepts)
        self.n_features_in_ = feature_count
        self.n_iter_ = np.asarray(iterations, dtype=np.int64)
        self.class_count_ = np.asarray(class_counts, dtype=np.int64)
        return self

    def decision_function(self, X):
        """复用学得参数，返回形状为[样本数, 10]的决策分数。"""
        check_is_fitted(self, attributes=["coef_", "intercept_", "classes_", "ratios_"])
        features = self._validate_features(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {features.shape[1]} features; expected {self.n_features_in_}.")
        return np.asarray(features @ self.coef_.T) + self.intercept_

    def predict(self, X):
        """选择分数最高的类别；同分时使用固定的类别顺序。"""
        scores = self.decision_function(X)
        return self.classes_[scores.argmax(axis=1)]
