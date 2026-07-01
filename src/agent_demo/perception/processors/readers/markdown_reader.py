"""Markdown 文件读取器 —— 正文 + 本地图片引用 OCR"""

import re
import tempfile
import os
from pathlib import Path
from .base import BaseFileReader

# 匹配 Markdown 图片语法: ![alt](path)
_MD_IMAGE_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
# 匹配 HTML img 标签: <img src="path" ...>
_HTML_IMG_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>', re.IGNORECASE)


class MarkdownReader(BaseFileReader):
    """
    读取 .md / .markdown 文件，增强处理：
    - 正文原样保留（Markdown 结构交给下游 LLM 理解）
    - 本地图片引用 → 解析路径 → OCR → 追加文字
    - 网络图片 → 保留引用（不下载，避免网络依赖）
    - base64 内嵌图片 → 解码 → OCR
    """

    suffixes = {'.md', '.markdown'}

    def __init__(self):
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr = RapidOCR()
        return self._ocr

    def read(self, path: Path) -> str:
        raw = path.read_text(encoding='utf-8')
        md_dir = path.parent  # 图片相对路径的基准目录

        # 收集所有图片引用
        img_refs: list[tuple[str, str]] = []  # [(alt_text, ref_path)]

        # Markdown 语法: ![alt](path)
        for m in _MD_IMAGE_RE.finditer(raw):
            alt = m.group(1) or ""
            ref = m.group(2)
            img_refs.append((alt, ref))

        # HTML 语法: <img src="path">
        for m in _HTML_IMG_RE.finditer(raw):
            ref = m.group(1)
            img_refs.append(("", ref))

        if not img_refs:
            return raw

        # OCR 每个可解析的图片
        ocr_results: list[str] = []
        for alt, ref in img_refs:
            ocr_text = self._resolve_and_ocr(ref, alt, md_dir)
            if ocr_text:
                label = f"📷 {alt}" if alt else f"📷 {ref}"
                ocr_results.append(f"\n> [{label}]\n> {ocr_text.replace(chr(10), chr(10) + '> ')}")

        if not ocr_results:
            return raw

        return raw + '\n' + '\n'.join(ocr_results)

    def _resolve_and_ocr(
        self, ref: str, alt: str, base_dir: Path
    ) -> str:
        """
        解析图片引用并 OCR：
        - 本地路径 → 读取文件 → OCR
        - data: URI → 解码 → OCR
        - http(s) URL → 跳过（标注）
        """
        # ---- HTTP(S) 链接：跳过 ----
        if ref.startswith(('http://', 'https://')):
            return ""  # 保持简洁，不引入网络依赖

        # ---- Base64 内嵌图片 ----
        if ref.startswith('data:image/'):
            return self._ocr_base64(ref)

        # ---- 本地文件路径 ----
        # 去除 URL 片段和查询参数
        clean_ref = ref.split('?')[0].split('#')[0]
        img_path = base_dir / clean_ref

        if not img_path.exists():
            # 尝试当前工作目录
            img_path = Path(clean_ref)
            if not img_path.exists():
                return ""

        return self._ocr_file(img_path)

    def _ocr_file(self, img_path: Path) -> str:
        """OCR 本地图片文件"""
        try:
            ocr = self._get_ocr()
            result, _ = ocr(str(img_path))
            if result:
                lines = [r[1] for r in result if r[1].strip()]
                return '\n'.join(lines)
        except Exception:
            pass
        return ""

    def _ocr_base64(self, data_uri: str) -> str:
        """解码 base64 内嵌图片并 OCR"""
        import base64
        # 格式: data:image/png;base64,iVBORw0...
        try:
            header, encoded = data_uri.split(',', 1)
            # 推断扩展名
            ext = 'png'
            if 'jpeg' in header or 'jpg' in header:
                ext = 'jpg'
            elif 'gif' in header:
                ext = 'gif'
            elif 'webp' in header:
                ext = 'webp'
            elif 'bmp' in header:
                ext = 'bmp'

            img_bytes = base64.b64decode(encoded)
            with tempfile.NamedTemporaryFile(
                suffix=f'.{ext}', delete=False
            ) as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name

            try:
                ocr = self._get_ocr()
                result, _ = ocr(tmp_path)
                if result:
                    lines = [r[1] for r in result if r[1].strip()]
                    return '\n'.join(lines)
            finally:
                os.unlink(tmp_path)
        except Exception:
            pass
        return ""
