"""统一记录环境、读写 JSON，并生成不依赖文件路径的数据指纹。"""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent


def sha256_file(path):
    """分块计算文件指纹，避免一次性将大文件读入内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integer_hash(values):
    """按原顺序编码为小端 int64，避免 CSV 换行格式影响标签核验。"""
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def json_hash(value):
    """使用固定 JSON 编码规则；列表顺序保留，字典键顺序统一。"""
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path):
    """配置、参考结果和运行记录统一使用 UTF-8。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_value(value):
    """将 NumPy 数值和路径转换为 JSON 可以保存的类型。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def write_json(path, value):
    """保存便于阅读的 JSON，保留中文并以一个换行结尾。"""
    content = json.dumps(value, ensure_ascii=False, indent=2, default=_json_value)
    Path(path).write_text(content + "\n", encoding="utf-8")


def verify_hash(actual, expected, description):
    """数据或预测指纹发生变化时立即停止，防止混用不同实验结果。"""
    if actual != expected:
        raise ValueError(f"复现核验失败：{description}。实际={actual}，参考={expected}")


def environment(config):
    """检查锁定的 Python 和依赖版本，并记录实际运行平台。"""
    expected = config["environment"]
    versions = {
        name: importlib.metadata.version(name) for name in expected["packages"]
    }
    if versions != expected["packages"]:
        raise RuntimeError("实验依赖版本不匹配，请用 requirements.txt 安装锁定版本。")
    python_version = platform.python_version()
    if python_version != expected["python"]:
        raise RuntimeError(f"请使用 Python {expected['python']}。")
    return {
        "python": python_version,
        "packages": versions,
        "platform": platform.platform(),
        "thread_limit": config["thread_limit"],
    }


def source_hashes():
    """记录本次运行使用的源码、配置和依赖清单，方便追溯版本。"""
    paths = [
        *BASE_DIR.glob("*.py"),
        BASE_DIR / "experiment_config.json",
        BASE_DIR / "reference_results.json",
        BASE_DIR / "requirements.txt",
    ]
    return {path.name: sha256_file(path) for path in sorted(paths)}
