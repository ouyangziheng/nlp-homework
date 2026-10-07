# 作业一：文本分类实现

在 NYT 新闻数据集上实现并比较三类文本表示方法：Bag-of-Words、
100 维词向量平均、微调 BERT。

**实验报告：`report_cvpr/main.pdf`**

## 结果（NYT Test Set）

### 任务一：Bag of Words

| 表示方法 | Accuracy | Macro-F1 |
|---|---:|---:|
| Binary Bag of Words | **0.9887** | **0.9738** |
| Word Frequency | 0.9879 | 0.9737 |
| TF-IDF | 0.9809 | 0.9613 |

### 任务二：Word2Vec（100 维）

| 词向量来源 | Accuracy | Macro-F1 |
|---|---:|---:|
| 预训练 GloVe (glove.6B.100d) | **0.9896** | **0.9770** |
| Word2Vec on NYT | 0.9870 | 0.9713 |
| Word2Vec on AG News | 0.9818 | 0.9577 |

### 任务三：BERT（max_length=64, 3 epochs）

| 模型 | Accuracy | Macro-F1 |
|---|---:|---:|
| bert-base-uncased | 0.9748 | 0.9433 |

参考：多数类基线 Accuracy 0.7285、Macro-F1 0.2810。

## 主要分析结论

**1. Binary 与 Word Frequency 几乎相同**（0.9738 / 0.9737）。长文档主题分类中，
词的"出现"承载主要信号，词频的额外信息很少。

**2. TF-IDF 反而低于原始词频**（0.9613）。按类别拆开看，损失集中在 business
（F1 0.9586→0.9324）：sports 占 75%，IDF 把 business/politics 的普通词当作
"稀有词"提权，却压低了真正区分两个少数类的中频词。

**3. 预训练 GloVe 效果最好**（0.9770），100 维即超过 127,767 维的词袋模型。
自训练词向量之间，领域匹配比语料规模更重要：NYT 训练（0.9713）优于
AG News 训练（0.9577）——后者文本平均仅 39 词（标题+首句），语域与 NYT 长文不匹配。

**4. BERT 最低（0.9433）是 `max_length=64` 导致的，不是模型能力问题。**
NYT 文档中位 847 个 word piece，测试集 **100% 的文档被截断**，
BERT 只读到每篇中位 **7.6%** 的内容，而其余方法用全文。
把词袋与 GloVe 也截断到 64 token 后排序反转：

| 方法 | 全文 | 前 64 token | Δ |
|---|---:|---:|---:|
| Binary BoW | 0.9738 | 0.9138 | −0.0600 |
| GloVe 平均 | 0.9770 | 0.9377 | −0.0393 |
| BERT | — | **0.9433** | — |

**5. Macro-F1 比 Accuracy 更适合本任务。** 所有方法在多数类 sports 上 F1 ≥ 0.9935，
差异全部来自两个少数类；Accuracy 把七种方法压在 0.975–0.990，Macro-F1 拉开到 0.943–0.977。

## 目录结构

```
HW-1/                      nyt.csv（11,519 篇 / 3 类）、ag.csv（90,000 条）
src/
  common.py                数据加载、80/10/10 划分、分词缓存、评价指标
  task1_bow.py             任务一：binary / word frequency / tf-idf + LogisticRegression
  task2_word2vec.py        任务二：GloVe、Word2Vec(AG News)、Word2Vec(NYT) 平均
  task3_bert.py            任务三：bert-base-uncased 微调（max_length=64, 3 epochs）
  analysis.py              跨任务比较 + 截断对照实验
  make_figures_cvpr.py     报告图（矢量 PDF，Type 42 字体）
report_cvpr/
  main.tex / main.pdf      实验报告（CVPR 格式，中文）
  refs.bib                 参考文献
  cvpr.sty 等              CVPR 官方模板文件
  fig/*.pdf                图
results/
  task1_bow.json           任务一完整指标（含每类 F1、classification_report）
  task2_word2vec.json      任务二完整指标
  task3_bert.json          任务三完整指标 + 每轮训练记录
  analysis.json            跨任务比较 + 截断分析
scripts/
  setup.sh                 环境准备
  download_assets.sh       下载 GloVe 与 BERT 权重
  run_all.sh               全流程
logs/                      各次运行的 stdout
```

## 代码运行说明

### 1. 准备环境

```bash
bash scripts/setup.sh              # ~2 分钟
```

在已有 torch / transformers / scikit-learn 的 Python 环境上建立 venv，
额外安装 `gensim` 与 `nltk`，并下载 `punkt` 分词模型到 `data/nltk_data`。
可用 `BASE_PYTHON=/path/to/python` 指定基础解释器。

### 2. 下载预训练资源

```bash
bash scripts/download_assets.sh    # ~3 分钟（822MB GloVe + 440MB BERT）
```

下载 `glove.6B.zip` 并解压出 `glove.6B.100d.txt`，
以及 `google-bert/bert-base-uncased`。

### 3. 运行全部实验

```bash
bash scripts/run_all.sh            # ~8 分钟
```

依次执行三个任务、比较分析、作图、编译 PDF。

### 单独运行某个任务

```bash
export PYTHONPATH=$PWD/src NLTK_DATA=$PWD/data/nltk_data HF_HOME=$PWD/data/hf_cache
.venv/bin/python src/task1_bow.py                              # 任务一
.venv/bin/python src/task2_word2vec.py                         # 任务二
.venv/bin/python src/task3_bert.py --max-length 64 --epochs 3   # 任务三
.venv/bin/python src/analysis.py                               # 比较分析
```

### 编译报告

需要 **xelatex**（中文用 xeCJK）与 Noto CJK 字体：

```bash
cd report_cvpr && xelatex main && bibtex main && xelatex main && xelatex main
```

## 实现要点

**统一的数据划分。** `common.get_splits()` 用固定种子（42）整体打乱一次后按
80/10/10 划分（9215 / 1151 / 1153），三个任务全部导入它，
保证不同方法之间可比；任务一与任务二还共享同一份缓存分词结果，
使词表来自完全相同的 token。

**避免测试集信息泄漏。** 词袋词表只在训练集上构建；
NYT 的 Word2Vec 只用**训练集**文本训练（若用全部 11519 篇，
测试集的词共现统计会影响词向量本身）。

**任务一的第三种方法。** 作业要求三种 Bag-of-Words 方法，
正文明确给出 Binary 与 Word Frequency 两种，第三种取该族标准变体 TF-IDF。

**任务三严格按作业设定。** `max_length=64`、3 epochs；
其余（batch 32、AdamW 2e-5、10% warmup、梯度裁剪 1.0、fp16）
为标准 BERT 微调配方。使用显式 PyTorch 训练循环而非 `Trainer`。

**可复现性。** 词袋、GloVe、BERT 均可逐位复现。
Word2Vec 需 `workers=1`：多线程下 gensim 更新顺序不确定，
固定种子后 Macro-F1 仍波动最多 0.007（与本文讨论的部分差异同量级）。
已由两次独立运行验证。

## 环境说明

- 本集群部分 conda 环境的 `_sqlite3.so` 是坏的，而 `nltk` 导入时即加载 `sqlite3`，
  所以 `setup.sh` 的基础解释器必须是 `sqlite3` 能正常导入的那个。
- 两个脚本默认导出 star-proxy 代理，可用 `USE_PROXY=` 关闭。
- `punkt` 直接用 curl 下载，因为 `nltk.download()` 不走代理环境变量。
- GloVe 从 HuggingFace 的 `stanfordnlp/glove` 镜像拉取（与作业给的 Stanford 链接
  是同一文件，但本网络下快约 100 倍），失败时回落到 Stanford 原始地址。
- 报告编译额外需要 `silence.sty` 与 `cleveref.sty`（已随 `report_cvpr/` 提交）
  以及 Noto CJK 字体（`yum install google-noto-serif-cjk-sc-fonts`）。

## 已知数据问题

`nyt.csv` 含 72 篇内容重复的文档，其中 8 篇在指定的随机划分下横跨训练集与测试集。
为不偏离作业规定的划分流程，保留原样，报告中已说明。
