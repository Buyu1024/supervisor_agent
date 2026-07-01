# AgentDemoProject

> 一个模块化的 AI Agent 开发项目，**感知模块（Perception Module）** 是第一个完成的子模块。

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📖 项目简介

本项目从零开始构建一个完整的 AI Agent，采用**模块化**的分层架构设计，共 5 个核心模块：

| 模块 | 状态 | 职责 |
|------|:---:|------|
| **感知模块** | ✅ 已完成 | 多源输入 → 格式识别 → 预处理 → 标准化输出 |
| LLM 模块 | 🔲 待开发 | 大语言模型调用与管理 |
| 记忆模块 | 🔲 待开发 | 对话历史、长期记忆、向量检索 |
| 规划模块 | 🔲 待开发 | 任务分解、推理路径规划 |
| 工具模块 | 🔲 待开发 | 工具注册、调用、结果解析 |

---

## 🧠 感知模块

### 核心能力

感知模块是 Agent 的"感官层"，负责接收一切外部输入，清洗、校验、标准化后交给下游模块。屏蔽了输入来源和文件格式的差异，下游模块只需消费统一的 `Message` 对象。

```
┌─────────────────────────────────────────────────────┐
│                   感知模块架构                        │
│                                                      │
│  输入                                                │
│  ├── 纯文本  ──→ TextProcessor                       │
│  └── 文件路径 ──→ FileProcessor ──→ 格式 Reader       │
│                      │                               │
│                      ▼                               │
│              PreprocessorPipeline                     │
│              去噪 → 敏感词 → 截断 → 语言检测          │
│                      │                               │
│                      ▼                               │
│               OutputAdapter → Message                │
└─────────────────────────────────────────────────────┘
```

### 文件格式支持（7 种 Reader）

| Reader | 支持格式 | 提取能力 |
|--------|---------|---------|
| `TextFileReader` | `.txt` `.json` `.csv` `.py` `.yaml` `.yml` `.log` `.xml` `.html` `.toml` `.cfg` `.ini` `.env` | 编码读取（UTF-8 / GBK） |
| `MarkdownReader` | `.md` `.markdown` | 正文保留 + 本地图片引用 OCR + base64 图片解码 OCR |
| `DocxReader` | `.docx` | 段落文本 + 嵌入图片 OCR |
| `PptxReader` | `.pptx` | 递归组合形状 + 表格 + 图表 + 嵌入图片 OCR + XY-Cut 空间排序 + 演讲者备注 |
| `PdfReader` | `.pdf` | 结构化 blocks 提取 + `find_tables()` 表格检测 → Markdown + 页眉页脚过滤 + XY-Cut 阅读顺序 + 嵌入图片 OCR + 扫描页 OCR 回退 |
| `ImageReader` | `.png` `.jpg` `.jpeg` `.bmp` `.webp` | 整图 OCR（RapidOCR） |

### 预处理管道

| 过滤器 | 职责 | 配置项 |
|--------|------|--------|
| `DenoiseFilter` | 控制字符移除、换行统一、空行压缩、首尾空白清理 | — |
| `SensitiveFilter` | 敏感词命中 → 整条拒绝 + 友好提示 | 外部 `sensitive_words.txt` |
| `TruncationFilter` | 超长文本截断 + metadata 记录原始长度 | `max_length`（默认 4000） |
| `LanguageFilter` | 语种检测 → 写入 metadata | — |

### 输出 Message 结构

```python
@dataclass
class Message:
    role: str           # "user"
    content: str        # 清洗后的文本
    source_type: str    # "text" / "file"
    metadata: dict      # { language, original_length, truncated, ... }
    is_rejected: bool   # 敏感词拦截标记
    reject_reason: str  # 拦截原因
    attachments: list   # 保留扩展字段
```

---

## 🚀 快速开始

### 环境要求

- **Python**: 3.12+
- **包管理器**: [uv](https://github.com/astral-sh/uv)

### 安装

```bash
# 克隆项目
git clone https://github.com/your-username/AgentDemoProject.git
cd AgentDemoProject

# 安装依赖
uv sync
```

### 使用示例

```python
from agent_demo.perception import PerceptionModule

# 初始化
pm = PerceptionModule(
    max_length=4000,
    sensitive_words_path="sensitive_words.txt",
)

# ---- 纯文本输入 ----
msg = pm.process("你好，这是一个测试消息")
print(msg.content)       # "你好，这是一个测试消息"
print(msg.metadata)      # {'language': 'zh-cn', 'truncated': False, ...}

# ---- 文件输入 ----
msg = pm.process("path/to/report.pdf")
msg = pm.process("path/to/slides.pptx")
msg = pm.process("path/to/doc.docx")
msg = pm.process("path/to/screenshot.png")
msg = pm.process("path/to/note.md")    # Markdown 含图片引用会自动 OCR

# ---- 敏感词拦截 ----
msg = pm.process("包含敏感词的内容")
if msg.is_rejected:
    print(msg.content)   # "您输入的内容包含违规信息，已被系统拦截..."
```

### 运行测试

```bash
uv run python tests/test_perception.py
# 23 个用例，全部通过
```

---

## 📁 项目结构

```
AgentDemoProject/
├── pyproject.toml                          # 项目配置与依赖
├── sensitive_words.txt                     # 敏感词库（示例）
├── README.md                               # 本文档
│
├── src/agent_demo/perception/              # 感知模块
│   ├── __init__.py                         # 公开 API
│   ├── module.py                           # PerceptionModule 主入口
│   ├── message.py                          # Message 数据类
│   ├── router.py                           # InputRouter 输入路由器
│   ├── pipeline.py                         # PreprocessorPipeline
│   ├── adapter.py                          # OutputAdapter
│   │
│   ├── processors/                         # 输入处理器
│   │   ├── base.py                         # BaseProcessor 抽象基类
│   │   ├── text.py                         # TextProcessor
│   │   ├── file.py                         # FileProcessor（按后缀委派 Reader）
│   │   │
│   │   └── readers/                        # 文件格式读取器
│   │       ├── base.py                     # BaseFileReader 抽象基类
│   │       ├── text_reader.py              # 纯文本类文件
│   │       ├── markdown_reader.py          # Markdown + 图片 OCR
│   │       ├── docx_reader.py              # Word（段落 + 嵌入图片 OCR）
│   │       ├── pptx_reader.py              # PPT（递归形状 + 表格/图表 + OCR）
│   │       ├── pdf_reader.py               # PDF（结构化 + 表格 + 扫描件 OCR）
│   │       └── image_reader.py             # 图片 OCR
│   │
│   └── filters/                            # 预处理过滤器
│       ├── base.py                         # BaseFilter 抽象基类
│       ├── denoise.py                      # 去噪过滤器
│       ├── sensitive.py                    # 敏感词过滤器
│       ├── truncation.py                   # 长度截断过滤器
│       └── language.py                     # 语言检测过滤器
│
└── tests/                                  # 测试
    ├── test_perception.py                  # 23 个测试用例
    └── samples/                            # 测试样本文件
        ├── ocr_01.pptx
        ├── ocr_02.docx
        ├── ocr_03.pdf
        ├── ocr_04.png
        └── ocr_05.md
```

---

## 📦 依赖

| 包 | 用途 | 体积 |
|---|------|------|
| `rapidocr-onnxruntime` | 图片 / 嵌入图片 OCR | 轻量（ONNX Runtime） |
| `python-docx` | Word 文档解析 | — |
| `python-pptx` | PPT 解析（形状/表格/图表） | — |
| `pymupdf` | PDF 解析 + 表格检测 + 页面渲染 | — |
| `langdetect` | 语言检测 | — |

> 设计原则：**不引入 PaddlePaddle**（体积大、Windows 兼容性差），OCR 使用基于 ONNX Runtime 的 RapidOCR。

---

## 🏗️ 设计原则

1. **模块化**：每个 Reader / Filter 独立，可插拔、可替换
2. **优先文件识别**：`InputRouter` 先匹配文件处理器，文件不存在才退化为文本
3. **责任链**：预处理管道按序执行，敏感词命中立即中断
4. **延迟加载**：OCR 模型仅在首次使用时初始化
5. **参考业界方案**：PPTX/PDF 处理参考 MinerU 的架构思路，但不照搬重量级模型

---

## 📄 许可

MIT License

---

## 🔮 后续计划

- [ ] LLM 模块：对接 Claude / OpenAI / 本地模型
- [ ] 记忆模块：对话历史 + 向量检索 + 长期记忆
- [ ] 规划模块：任务分解与推理规划
- [ ] 工具模块：工具注册、调用、结果解析
