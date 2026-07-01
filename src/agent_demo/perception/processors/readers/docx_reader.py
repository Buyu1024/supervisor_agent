""".docx Word 文档读取器 —— 段落文本 + 嵌入图片 OCR"""

import tempfile
import os
from pathlib import Path
from .base import BaseFileReader


class DocxReader(BaseFileReader):
    """
    读取 .docx 文件，双通道提取：
    1. 段落文本（python-docx）
    2. 嵌入图片 → OCR（RapidOCR）
    """

    suffixes = {'.docx'}

    def __init__(self):
        self._ocr = None

    def _get_ocr(self):
        """延迟初始化 RapidOCR"""
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr = RapidOCR()
        return self._ocr

    def read(self, path: Path) -> str:
        from docx import Document

        doc = Document(str(path))
        parts: list[str] = []

        # ---- 通道 1：段落文本 ----
        for p in doc.paragraphs:
            text = p.text.strip()
            if text:
                parts.append(text)

        # ---- 通道 2：嵌入图片 OCR ----
        image_texts = self._ocr_embedded_images(doc)
        if image_texts:
            parts.append("--- [文档内嵌图片识别] ---")
            parts.extend(image_texts)

        return '\n'.join(parts)

    def _ocr_embedded_images(self, doc) -> list[str]:
        """
        提取 .docx 中所有嵌入图片，OCR 后返回文字列表

        python-docx 的图片存储在 document part 的 relationships 中，
        reltype 包含 "image" 即为嵌入图片。
        """
        texts: list[str] = []
        for rel in doc.part.rels.values():
            if "image" not in rel.reltype:
                continue

            image_part = rel.target_part
            suffix = Path(image_part.partname).suffix or '.png'
            # 将图片 blob 写入临时文件
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(image_part.blob)
                tmp_path = tmp.name

            try:
                ocr = self._get_ocr()
                result, _ = ocr(tmp_path)
                if result:
                    lines = [r[1] for r in result if r[1].strip()]
                    if lines:
                        texts.extend(lines)
            finally:
                os.unlink(tmp_path)

        return texts
