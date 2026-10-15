# 实验一：文本分类

从原 `draft_main.py` 的加载、特征、分类、输出框架整理而来，主入口为 `main.py`。默认从原始 CSV 开始，一次运行重现整个已锁定的实验：固定划分与清洗 → 训练集拟合 TF-IDF → 四模型各自最佳参数训练和验证 → 最终 NB 加权 SVM 测试预测。

**默认运行不重新选择超参数，不合并训练集与验证集重训。** 最终模型保持 `C=0.2, alpha=1`，只用5894条训练样本学习参数，以便和已经提交的实验结果一致。验证集1471条只评价，测试2457条只在最后预测。

## 运行

使用 **Python 3.10.22**，安装完整锁定依赖后直接运行：

如果需要新建环境，可先执行 `conda create -n textclass_reproduce python=3.10.22 -y`，然后 `conda activate textclass_reproduce`。已有相同版本环境时直接使用即可。

```powershell
python -m pip install -r requirements.txt
python main.py
```

在 PyCharm 中选择相同版本的解释器，以 `main.py` 为运行文件即可。代码按自身所在位置寻找 `data/`，不依赖终端或IDE的当前工作目录，不含本机绝对路径。

原始 `data/train_data.csv` 与 `data/test_data_unlabeled.csv` 是复现输入，包含在交付文件夹中，约17.4 MiB，应随源码提交。`.gitattributes` 禁止Git自动改变原始CSV换行，确保输入SHA256不变。其他目录的缓存、模型快照和虚拟环境均不需要复制。

每次运行输出到新的 `outputs/<模式>_<UTC时间戳>/`，不会覆盖此前结果。`outputs/` 被 `.gitignore` 排除，不提交到GitHub。

## 四种运行模式

| 命令 | 执行范围 | 读取测试集 |
| --- | --- | --- |
| `python main.py` | 默认all：准备数据、四模型最佳参数比较、最终NB-SVM预测 | 仅最后预测 |
| `python main.py --mode final` | 准备数据、最终NB-SVM训练/验证/测试预测 | 仅最后预测 |
| `python main.py --mode compare` | 准备数据、四模型最佳参数比较 | 否 |
| `python main.py --mode search --model nb_weighted_svm --round 3` | 重现指定模型的第三轮实际搜索 | 否 |

搜索模型可选 `multinomial_nb`、`logistic_regression`、`linear_svm`、`nb_weighted_svm`，轮次可选1、2、3或all。例如：

```powershell
python main.py --mode search --model linear_svm --round all
python main.py --mode search --model nb_weighted_svm --round 2
```

网格来自实际保存的三轮搜索，记录在 `experiment_config.json`。NB-SVM第三轮实际只运行5个C、alpha固定1，代码没有把建议但未执行的组合算入历史结果。搜索保存候选指标和验证预测，不自动改写最终参数或继续预测测试集。

四模型历史90次候选运行的训练/验证预测指纹已保存在 `reference_results.json`，搜索模式会逐候选核对预测与验证指标。参考指纹是小型复现依据，不是模型或中间数据副本。

这样安排是因为开发阶段的比较/调参与最终预测职责不同；统一入口便于复现，独立模式避免每次调参都接触测试集。

## 文件与引用关系

```text
main.py
 ├─ data_preparation.py ─ text_cleaning.py
 ├─ text_features.py
 ├─ model_training.py ─ nb_weighted_linear.py
 ├─ test_prediction.py ─ text_cleaning.py
 └─ reproducibility.py

verify_reproduction.py    独立核对产物、预测和学得参数
experiment_config.json   数据身份、冻结划分、TF-IDF、最佳参数、历史搜索网格
reference_results.json   历史参考指标及数值指纹，用于检验重现结果
requirements.txt         Python依赖版本锁
data/                    两个原始CSV
```

| 模块 | 作用 |
| --- | --- |
| `main.py` | 组织执行次序，限制测试集读取阶段，保存运行记录 |
| `data_preparation.py` | 8:2分层划分、冻结重叠隔离、调用清洗，在内存中返回数据 |
| `text_cleaning.py` | 保留Subject，删除连续邮件头、历史引用、明确回复来源及作者匹配署名；统一NFC、小写、空白 |
| `text_features.py` | 仅训练集拟合词表/IDF，验证数据只transform |
| `model_training.py` | 共用四模型的训练、评价、参数保存和历史网格搜索 |
| `nb_weighted_linear.py` | 多类别NB权重加线性分类器，分开权重计算与二分类器构造 |
| `test_prediction.py` | 固定清洗、固定TF-IDF与最终NB-SVM预测，保持测试行顺序 |
| `reproducibility.py` | 锁定环境、输入/源码/数值指纹、统一记录格式 |
| `verify_reproduction.py` | 独立检查保存的预测、学得参数和完整性 |

三种基线保留在共用训练模块内，因此不需要三个重复的命令行训练文件。保留真实超参数搜索代码和网格，是为了能重现为什么选出最终参数。未纳入ModernBERT等未完成真实数据训练的扩展；它们不属于本实验最终四模型比较。

阅读代码时，可以先看 `main.py` 的 `run_pipeline()`，了解各阶段的执行顺序，再进入对应模块。数据准备依次加载、分层划分、清洗、隔离重复验证行；训练依次拟合与预测、核对参考、保存产物。验证和测试共用 `transform_texts()`，JSON读写与指纹函数集中在 `reproducibility.py`。注释解释数据边界、清洗判断和NB权重含义，固定实验设置集中在配置中。

## 数据独立性与冻结划分

原始有标签数据7368条，`stratify=target, random_state=42, test_size=0.2`，各子集按原始行号排序。得到5894条训练和1474条初始验证。按已冻结规则隔离与训练清洗全文相同的验证行 `[2004, 2009, 6131]`，有效验证1471条；隔离行不加入训练。

这里用数据文件SHA256、分割算法、行号指纹和固定隔离名单替代旧的大型 `split_reference.csv`。运行时检查行号、清洗文本、词序和IDF指纹，遇到不一致直接停止，而非静默变更实验。

词表、IDF、NB权重、分类器系数全部来自训练集。验证与测试的词表/IDF禁止变化。清洗规则本身不拟合词频。保持不做增强、不截断、不统一删除数字标点，避免改变已接受的实验方案。

NB加权SVM是受Wang和Manning 2012论文启发的多类别OVR变体：训练TF-IDF转为词项出现指示来估计各类NB log-count-ratio，再对该类TF-IDF加权训练LinearSVC。比例分母是整张词表的词项出现次数之和，并非类别文档数。推断取十类未校准分数最大者，不把分数当概率。这不是原论文二分类插值模型的逐字复现。论文：https://aclanthology.org/P12-2018/

## 参考结果及核验

| 模型 | 已搜索范围内最佳参数 | 验证Macro-F1 | 验证Accuracy |
| --- | --- | ---: | ---: |
| 多项式朴素贝叶斯 | alpha=0.2 | 0.894775253 | 0.893949694 |
| 逻辑回归 | C=5 | 0.908163936 | 0.907545887 |
| 线性SVM | C=2 | 0.916025086 | 0.915703603 |
| NB加权SVM | C=0.2, alpha=1 | 0.928180661 | 0.927940177 |

运行会自动和历史参考的训练、验证预测逐标签指纹核对；最终测试预测也自动核对。四模型使用同一组清洗文本和20000维TF-IDF。逻辑回归max_iter=2000，两种SVM为5000，tol=0.0001，种子42，线程限制1。模型参数和完整配置都在输出中记录。

输出包括各模型学得参数 `.npz`、模型 `.joblib`、分类报告、混淆矩阵、预测，及只保存一次的TF-IDF向量器、划分记录和运行manifest。**不保存重复原始/清洗全文、候选模型全集或训练特征矩阵**，这些均能由原始CSV重建。

`predictions.csv` 每行一个0～9整数，无表头和索引。新入口统一LF换行，与历史Windows文件的CRLF可能有字节差异；2457个标签内容及顺序必须完全相同。测试无真实标签，不能计算测试Accuracy、Macro-F1或混淆矩阵。

进一步核验保存的学得参数和产物：

```powershell
python verify_reproduction.py outputs/<具体运行目录名>
```

`reference_results.json` 包含历史关键数组的形状、类型和SHA256。本机锁定环境中可以核对参数逐位相同；跨操作系统/BLAS实现可能产生数值低位差异，独立核验会记录该差异，不会把它隐藏成逐位复现。主流程仍要求参考预测标签完全一致，否则停止。

验证集已经用于多轮超参数选择，这些分数是验证比较结果，不是独立泛化测试。各模型搜索预算不同，最佳指已保存搜索范围内的最佳。

## GitHub交付范围

提交本文件夹内源码、配置、参考指纹、依赖、README和两个原始数据CSV。不要提交 `outputs/`、`__pycache__/`、IDE设置或虚拟环境。最终学得模型和图表可作为实验报告附件保留在本地，仓库通过入口重新生成。

本代码结构的目标是让获得相同原始数据的人，从空输出目录直接运行 `main.py` 重现结果，且能单独检查训练和调参过程。

`.github/workflows/reproduce.yml` 已配置GitHub Actions：推送或提交PR后，自动在Windows、Python3.10.22环境安装锁定依赖，从原始数据运行主入口并独立核验。工作流文件已经准备好，本地交付时尚未在GitHub实际执行。
