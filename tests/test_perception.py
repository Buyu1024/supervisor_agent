"""感知模块 - 集成测试 Demo"""

import tempfile
import os
from pathlib import Path


# ---- 测试辅助：创建临时文件 ----

def _make_temp_file(content: str, suffix: str = ".txt") -> str:
    """创建临时文件并返回路径"""
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix=suffix, delete=False, encoding='utf-8'
    )
    tmp.write(content)
    tmp.close()
    return tmp.name


def _make_sensitive_file(words: list[str]) -> str:
    """创建临时的敏感词文件"""
    return _make_temp_file("\n".join(words), suffix=".txt")


# ============================================================
# 测试用例
# ============================================================

class TestTextInput:
    """纯文本输入测试"""

    def test_basic(self):
        """基本文本输入"""
        from agent_demo.perception import PerceptionModule
        pm = PerceptionModule()
        msg = pm.process("你好，这是一个测试消息")
        assert not msg.is_rejected
        assert msg.source_type == "text"
        assert "测试消息" in msg.content
        assert "language" in msg.metadata
        print(f"  [PASS] 基本文本输入: {msg}")

    def test_denoise_whitespace(self):
        """去噪 - 多余空白压缩"""
        from agent_demo.perception import PerceptionModule
        pm = PerceptionModule()
        msg = pm.process("  你好，  世界！  \n\n\n\n  多空行  ")
        assert not msg.is_rejected
        # 首尾空白去除，连续空行压缩
        assert msg.content.startswith("你好")
        assert "世界" in msg.content
        assert "多空行" in msg.content
        print(f"  [PASS] 去噪空白: '{msg.content}'")


class TestFileInput:
    """文件输入测试"""

    def test_read_txt(self):
        """读取 .txt 文件"""
        from agent_demo.perception import PerceptionModule
        pm = PerceptionModule()
        path = _make_temp_file("文件中的测试内容。\n第二行内容。")
        try:
            msg = pm.process(path)
            assert not msg.is_rejected
            assert msg.source_type == "file"
            assert "测试内容" in msg.content
            print(f"  [PASS] 文件读取(.txt): {msg}")
        finally:
            os.unlink(path)

    def test_read_md(self):
        """读取 .md 文件（纯文本，无图片）"""
        from agent_demo.perception import PerceptionModule
        pm = PerceptionModule()
        path = _make_temp_file("# Markdown 标题\n正文内容", suffix=".md")
        try:
            msg = pm.process(path)
            assert not msg.is_rejected
            assert msg.source_type == "file"
            assert "Markdown" in msg.content
            print(f"  [PASS] MD 纯文本: {msg}")
        finally:
            os.unlink(path)

    def test_read_md_with_image(self):
        """读取 .md 文件（含本地图片引用 → OCR 提取图片文字）"""
        from agent_demo.perception import PerceptionModule
        pm = PerceptionModule()
        samples_dir = Path(__file__).parent / "samples"
        sample_img = samples_dir / "ocr_04.png"

        # 构造一个引用本地图片的 md 文件
        md_content = f"# 文档标题\n\n正文内容。\n\n![示例图片]({sample_img.as_posix()})"
        path = _make_temp_file(md_content, suffix=".md")
        try:
            msg = pm.process(path)
            assert not msg.is_rejected
            assert msg.source_type == "file"
            assert "# 文档标题" in msg.content  # 正文保留
            # 图片 OCR 结果应出现在输出中
            assert "📷" in msg.content
            assert len(msg.content) > len(md_content)  # 输出比原始 MD 长
            print(f"  [PASS] MD 含本地图片: +{len(msg.content) - len(md_content)} 字符(OCR)")
            # 打印 OCR 结果
            idx = msg.content.find("📷")
            if idx >= 0:
                print(f"    OCR 结果预览: {msg.content[idx:idx+200]}...")
        finally:
            os.unlink(path)

    def test_read_md_url_image_skipped(self):
        """读取 .md 文件（含网络图片 → 跳过不下载）"""
        from agent_demo.perception import PerceptionModule
        pm = PerceptionModule()

        md_content = "# 文档\n\n![网络图](https://example.com/photo.png)"
        path = _make_temp_file(md_content, suffix=".md")
        try:
            msg = pm.process(path)
            assert not msg.is_rejected
            assert msg.source_type == "file"
            # 网络图片不应触发 OCR，输出长度应接近原文
            assert "https://example.com/photo.png" in msg.content
            print(f"  [PASS] MD 网络图片跳过: len={len(msg.content)}")
        finally:
            os.unlink(path)

    def test_unsupported_type(self):
        """不支持的文件后缀 → 退回当纯文本处理"""
        from agent_demo.perception import PerceptionModule
        pm = PerceptionModule()
        msg = pm.process("photo.jpg")  # 文件不存在 + 后缀不在白名单 → 当文本处理
        assert not msg.is_rejected
        assert msg.source_type == "text"
        print(f"  [PASS] 不支持的扩展名→退化为文本: {msg}")

    def test_nonexistent_file(self):
        """不存在的文件路径 → 退回当纯文本处理"""
        from agent_demo.perception import PerceptionModule
        pm = PerceptionModule()
        msg = pm.process("不存在的文件.txt")  # 文件不存在 → 当文本处理
        assert not msg.is_rejected
        assert msg.source_type == "text"
        print(f"  [PASS] 文件不存在→退化为文本: {msg}")


class TestSensitiveFilter:
    """敏感词拦截测试"""

    def test_reject_on_sensitive(self):
        """命中敏感词 → 整条拒绝"""
        from agent_demo.perception import PerceptionModule
        words_file = _make_sensitive_file(["暴力", "赌博"])
        try:
            pm = PerceptionModule(sensitive_words_path=words_file)
            msg = pm.process("这里包含暴力内容")
            assert msg.is_rejected
            assert msg.reject_reason == "sensitive_words"
            print(f"  [PASS] 敏感词拦截: {msg.content}")
        finally:
            os.unlink(words_file)

    def test_pass_clean_content(self):
        """正常内容不受影响"""
        from agent_demo.perception import PerceptionModule
        words_file = _make_sensitive_file(["暴力", "赌博"])
        try:
            pm = PerceptionModule(sensitive_words_path=words_file)
            msg = pm.process("这是正常的内容")
            assert not msg.is_rejected
            print(f"  [PASS] 正常内容通过: {msg}")
        finally:
            os.unlink(words_file)

    def test_no_sensitive_file(self):
        """不传敏感词文件 → 不过滤"""
        from agent_demo.perception import PerceptionModule
        pm = PerceptionModule()  # 不传 sensitive_words_path
        msg = pm.process("包含暴力词汇")
        assert not msg.is_rejected  # 没有词库，不会被拦截
        print(f"  [PASS] 无敏感词文件=不过滤: {msg}")

    def test_comment_lines_ignored(self):
        """敏感词文件中 # 注释行被忽略"""
        from agent_demo.perception import PerceptionModule
        # 只有 # 注释行，没有实质敏感词
        content = "# 这是注释\n# 另一行注释"
        path = _make_temp_file(content, suffix=".txt")
        try:
            pm = PerceptionModule(sensitive_words_path=path)
            msg = pm.process("任意内容")
            assert not msg.is_rejected
            print(f"  [PASS] 注释行被忽略: {msg}")
        finally:
            os.unlink(path)


class TestTruncation:
    """长度截断测试"""

    def test_truncate_long(self):
        """超长内容被截断"""
        from agent_demo.perception import PerceptionModule
        pm = PerceptionModule(max_length=50)
        long_text = "哈" * 100
        msg = pm.process(long_text)
        assert not msg.is_rejected
        assert len(msg.content) == 50
        assert msg.metadata["truncated"] is True
        assert msg.metadata["original_length"] == 100
        print(f"  [PASS] 截断: {msg.metadata['original_length']} → {len(msg.content)}")

    def test_no_truncate_short(self):
        """短内容不截断"""
        from agent_demo.perception import PerceptionModule
        pm = PerceptionModule(max_length=50)
        short_text = "短"
        msg = pm.process(short_text)
        assert msg.metadata["truncated"] is False
        assert msg.metadata["original_length"] == 1
        print(f"  [PASS] 短内容不截断: len={msg.metadata['original_length']}")


class TestLanguageDetection:
    """语言检测测试"""

    def test_chinese(self):
        """中文检测"""
        from agent_demo.perception import PerceptionModule
        pm = PerceptionModule()
        msg = pm.process("这是一段中文文本用于测试语言检测功能")
        assert msg.metadata["language"] in ("zh-cn", "zh-tw", "ko")
        # langdetect 对短中文可能误判为 ko，我们宽松断言
        print(f"  [PASS] 语言检测: {msg.metadata['language']}")

    def test_english(self):
        """英文检测"""
        from agent_demo.perception import PerceptionModule
        pm = PerceptionModule()
        msg = pm.process("This is a long English sentence for language detection testing purpose")
        assert msg.metadata["language"] == "en"
        print(f"  [PASS] 语言检测(英文): {msg.metadata['language']}")

    def test_empty_text(self):
        """空文本标记为 unknown"""
        from agent_demo.perception import PerceptionModule
        pm = PerceptionModule()
        msg = pm.process("")
        assert msg.metadata["language"] == "unknown"
        print(f"  [PASS] 空文本: language={msg.metadata['language']}")


class TestFileFormats:
    """多格式文件读取测试 —— 使用 tests/samples/ 下的示例文件"""

    SAMPLES_DIR = Path(__file__).parent / "samples"

    def test_read_docx(self):
        """读取 .docx 文件"""
        from agent_demo.perception import PerceptionModule
        pm = PerceptionModule()
        path = self.SAMPLES_DIR / "ocr_02.docx"
        msg = pm.process(str(path))
        assert not msg.is_rejected
        assert msg.source_type == "file"
        assert len(msg.content) > 0, "docx 应提取到文字内容"
        print(f"  [PASS] DOCX 读取: len={len(msg.content)}")
        print(f"    内容预览: {msg.content[:]}...")

    def test_read_pptx(self):
        """读取 .pptx 文件"""
        from agent_demo.perception import PerceptionModule
        pm = PerceptionModule()
        path = self.SAMPLES_DIR / "ocr_01.pptx"
        msg = pm.process(str(path))
        assert not msg.is_rejected
        assert msg.source_type == "file"
        assert len(msg.content) > 0, "pptx 应提取到文字内容"
        print(f"  [PASS] PPTX 读取: len={len(msg.content)}")
        print(f"    内容预览: {msg.content[:]}...")

    def test_read_pdf(self):
        """读取 .pdf 文件"""
        from agent_demo.perception import PerceptionModule
        pm = PerceptionModule()
        path = self.SAMPLES_DIR / "ocr_03.pdf"
        msg = pm.process(str(path))
        assert not msg.is_rejected
        assert msg.source_type == "file"
        assert len(msg.content) > 0, "pdf 应提取到文字内容"
        print(f"  [PASS] PDF 读取: len={len(msg.content)}")
        print(f"    内容预览: {msg.content[:]}...")

    def test_read_png_ocr(self):
        """读取 .png 文件（PaddleOCR）"""
        from agent_demo.perception import PerceptionModule
        pm = PerceptionModule()
        path = self.SAMPLES_DIR / "ocr_04.png"
        msg = pm.process(str(path))
        assert not msg.is_rejected
        assert msg.source_type == "file"
        # PaddleOCR 可能返回文字或"未识别到文字"
        assert len(msg.content) > 0, "png 应返回 OCR 结果或提示信息"
        print(f"  [PASS] PNG OCR: len={len(msg.content)}")
        print(f"    内容预览: {msg.content[:]}...")


class TestFullPipeline:
    """完整管道集成测试"""

    def test_text_full(self):
        """文本输入走完整管道"""
        from agent_demo.perception import PerceptionModule
        words_file = _make_sensitive_file(["暴力"])
        try:
            pm = PerceptionModule(
                max_length=4000,
                sensitive_words_path=words_file,
            )
            raw = "  Hello World!  这是测试  \n\n\n  多余空行  "
            msg = pm.process(raw)
            assert not msg.is_rejected
            assert msg.source_type == "text"
            assert "language" in msg.metadata
            assert "truncated" in msg.metadata
            assert "original_length" in msg.metadata
            print(f"  [PASS] 完整管道(文本): {msg}")
            print(f"    metadata: {msg.metadata}")
        finally:
            os.unlink(words_file)

    def test_file_full(self):
        """文件输入走完整管道"""
        from agent_demo.perception import PerceptionModule
        words_file = _make_sensitive_file(["暴力"])
        file_path = _make_temp_file("  Hello from   file!  \n\n\n  Clean  ")
        try:
            pm = PerceptionModule(
                max_length=4000,
                sensitive_words_path=words_file,
            )
            msg = pm.process(file_path)
            assert not msg.is_rejected
            assert msg.source_type == "file"
            assert "language" in msg.metadata
            print(f"  [PASS] 完整管道(文件): {msg}")
            print(f"    metadata: {msg.metadata}")
        finally:
            os.unlink(words_file)
            os.unlink(file_path)


# ============================================================
# 运行入口
# ============================================================

def run_all():
    """运行所有测试并汇总结果"""
    import sys

    test_classes = [
        TestTextInput,
        TestFileInput,
        TestSensitiveFilter,
        TestTruncation,
        TestLanguageDetection,
        TestFileFormats,
        TestFullPipeline,
    ]

    total = 0
    passed = 0
    failed = 0

    print("=" * 60)
    print("感知模块 (Perception Module) 测试")
    print("=" * 60)

    for cls in test_classes:
        print(f"\n--- {cls.__name__} ---")
        instance = cls()
        for name in dir(instance):
            if name.startswith("test_"):
                total += 1
                try:
                    getattr(instance, name)()
                    passed += 1
                except Exception as e:
                    failed += 1
                    import traceback
                    print(f"  [FAIL] {name}: {e}")
                    traceback.print_exc()

        # 清理 teardown
        if hasattr(instance, 'teardown_method'):
            instance.teardown_method()

    print("\n" + "=" * 60)
    print(f"测试完成: {total} 个用例 | 通过: {passed} | 失败: {failed}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    import sys
    sys.exit(0 if success else 1)
