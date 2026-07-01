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
| **LLM 模块** | ✅ 已完成 | LLM 调用 + Function Calling 闭环管理 |
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

## 🤖 LLM 模块

### 核心能力

LLM 模块封装了 qwen3.7-plus 模型调用，支持纯对话和 Function Calling 工具调用两种模式。**工具调用闭环完全在模块内部管理**，对外只暴露"输入消息 → 最终回复"。

```
┌─────────────────────────────────────────────────────┐
│                 LLM 模块架构                          │
│                                                      │
│  messages ──→ PromptBuilder ──→ QwenClient           │
│  tools ────→                    (DashScope API)      │
│  context ──→                          │              │
│                                       ▼              │
│                               ToolLoopManager        │
│                               finish_reason?         │
│                               ├── "stop" → 返回文本  │
│                               └── "tool_calls"       │
│                                    → 执行工具        │
│                                    → 回填结果        │
│                                    → 循环            │
│                                       │              │
│                                       ▼              │
│                                  LLMResponse         │
└─────────────────────────────────────────────────────┘
```

### 组件

| 组件 | 文件 | 职责 |
|------|------|------|
| `QwenClient` | `client.py` | DashScope OpenAI 兼容接口封装 + 指数退避重试 |
| `PromptBuilder` | `prompt_builder.py` | 内部 Message/dict → OpenAI messages 格式转换，上下文注入 |
| `ToolLoopManager` | `tool_loop.py` | Function Calling 闭环：轮询 finish_reason、执行工具、回填结果 |
| `LLMModule` | `module.py` | 主入口，组装上述组件 + 多轮对话历史管理 |

### LLMResponse 结构

```python
@dataclass
class LLMResponse:
    content: str                # 最终回复文本
    finish_reason: str          # "stop" / "tool_calls" / "error" / "max_rounds"
    token_usage: dict           # {prompt_tokens, completion_tokens, total_tokens}
    tool_calls_log: list[dict]  # 工具调用记录 [{round, name, arguments, result}]
```

### 使用示例

```python
from agent_demo.llm import LLMModule
from agent_demo.perception import PerceptionModule, Message

# 纯对话模式
llm = LLMModule(
    api_key="sk-xxx",                       # None → 读环境变量 DASHSCOPE_API_KEY
    system_prompt="用中文简短回答",
)

# 单轮对话
msg = Message(content="你好，请介绍一下自己", role="user")
response = llm.chat(messages=[msg])
print(response.content)

# 多轮对话（历史自动管理）
llm.chat(messages=[Message(content="我叫张三", role="user")])
response = llm.chat(messages=[Message(content="我叫什么名字？", role="user")])
print(response.content)  # "你叫张三。"
llm.clear_history()

# 感知模块 → LLM 模块串联
pm = PerceptionModule()
msg = pm.process("path/to/file.txt")
response = llm.chat(messages=[msg])

# 带工具的调用（注入 ToolsModule executor）
# tools_mod = ToolsModule()
# llm = LLMModule(tool_executor=tools_mod.get_executor())
# response = llm.chat(messages=[msg], tools=tools_mod.get_schemas())
```

**关键设计**：
- **内部闭环**：工具调用循环在 ToolLoopManager 中自动管理，调用方只需传入工具 schema
- **历史管理**：`_history` 自动累积多轮对话，支持 context 动态更新
- **回调注入**：`tool_executor` 通过构造函数注入，LLM 模块不持有 ToolsModule

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
# 感知模块测试（23 个用例）
uv run python tests/test_perception.py

# LLM 模块测试（15 个用例，需要 DASHSCOPE_API_KEY 环境变量）
uv run python tests/test_llm.py
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
├── src/agent_demo/llm/                     # LLM 模块
│   ├── __init__.py                         # 公开 API
│   ├── types.py                            # LLMResponse 数据结构
│   ├── client.py                           # QwenClient（DashScope 封装 + 重试）
│   ├── prompt_builder.py                   # Message → OpenAI messages 格式转换
│   ├── tool_loop.py                        # Function Calling 闭环管理
│   └── module.py                           # LLMModule 主入口
│
└── tests/                                  # 测试
    ├── test_perception.py                  # 感知模块（23 个测试用例）
    ├── test_llm.py                         # LLM 模块（15 个测试用例）
    └── samples/                            # 测试样本文件
        ├── ocr_01.pptx
        ├── ocr_02.docx
        ├── ocr_03.pdf
        ├── ocr_04.png
        └── ocr_05.md
```

---

## 📦 依赖

| 包 | 用途 | 所属模块 |
|---|------|:---:|
| `rapidocr-onnxruntime` | 图片 / 嵌入图片 OCR | 感知 |
| `python-docx` | Word 文档解析 | 感知 |
| `python-pptx` | PPT 解析（形状/表格/图表） | 感知 |
| `pymupdf` | PDF 解析 + 表格检测 + 页面渲染 | 感知 |
| `langdetect` | 语言检测 | 感知 |
| `openai` | LLM API 调用（OpenAI 兼容协议） | LLM |

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

- [x] LLM 模块：qwen3.7-plus + Function Calling 闭环
- [ ] 工具模块：工具注册、Schema 管理、执行器
- [ ] 记忆模块：对话历史 + 向量检索 + 长期记忆
- [ ] 规划模块：任务分解与推理规划
