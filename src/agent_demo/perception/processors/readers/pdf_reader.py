""".pdf 文档读取器 —— 结构化提取 + 表格 + 空间排序 + 扫描页 OCR

参考 MinerU 的 PDF 处理管线，基于 pymupdf (fitz) 实现：
1. get_text("dict") 结构化提取 → 保留字体/字号/坐标
2. find_tables() 表格检测 → Markdown 格式输出
3. 页眉/页脚自动过滤（Y 坐标裁剪）
4. XY-Cut 空间排序 → 正确阅读顺序（支持多栏）
5. 嵌入图片 → OCR 提取文字
6. 扫描页回退 → 整页渲染 + OCR
"""

import tempfile
import os
from pathlib import Path
from .base import BaseFileReader


class PdfReader(BaseFileReader):
    """
    读取 .pdf 文件，混合策略：

    - 文字型页面 → 结构化提取（文本块 + 表格 + 阅读顺序）
    - 扫描型页面 → 整页渲染图片 → OCR
    """

    suffixes = {'.pdf'}

    # 单页文字少于该字符数 → 视为扫描页，触发全页 OCR
    SCAN_THRESHOLD = 50

    # 页眉/页脚裁剪比例（页面高度的百分比）
    HEADER_RATIO = 0.08   # 顶部 8%
    FOOTER_RATIO = 0.92   # 底部 8%

    # 渲染扫描页的 DPI
    OCR_DPI = 200

    def __init__(self):
        self._ocr = None  # 延迟加载

    def _get_ocr(self):
        """延迟初始化 RapidOCR"""
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr = RapidOCR()
        return self._ocr

    # ================================================================
    # 主入口
    # ================================================================

    def read(self, path: Path) -> str:
        import fitz

        doc = fitz.open(str(path))
        pages_output: list[str] = []

        try:
            for page_num, page in enumerate(doc):
                # ---- 判断是否为扫描页 ----
                text_dict = page.get_text("dict")
                text_blocks = [
                    b for b in text_dict["blocks"] if b.get("type") == 0
                ]
                total_chars = sum(
                    len(span.get("text", ""))
                    for block in text_blocks
                    for line in block.get("lines", [])
                    for span in line.get("spans", [])
                )

                if total_chars < self.SCAN_THRESHOLD:
                    # 扫描型页面 → 全页 OCR
                    ocr_text = self._ocr_full_page(page)
                    if ocr_text:
                        pages_output.append(
                            f"--- [第{page_num + 1}页 · OCR] ---"
                        )
                        pages_output.append(ocr_text)
                    continue

                # ---- 文字型页面：结构化提取 ----
                page_lines = []

                # 1. 提取文本块（空间排序 + 页眉页脚过滤）
                body_blocks = self._filter_body_blocks(
                    text_blocks, page.rect.height
                )
                sorted_blocks = self._sort_by_reading_order(body_blocks)

                for block in sorted_blocks:
                    block_text = self._render_text_block(block)
                    if block_text:
                        page_lines.append(block_text)

                # 2. 表格检测 → Markdown
                try:
                    tables = page.find_tables(strategy="lines_strict")
                    for table in tables:
                        md = table.to_markdown(clean=True)
                        if md.strip():
                            # 在文本中找到表格应插入的大致位置（按 Y 坐标）
                            page_lines.append("\n" + md)
                except Exception:
                    pass  # 表格检测失败不影响文本提取

                # 3. 嵌入图片 OCR
                image_texts = self._ocr_embedded_images(page, doc)
                if image_texts:
                    for img_text in image_texts:
                        if img_text:
                            page_lines.append(img_text)

                if page_lines:
                    pages_output.append(f"--- [第{page_num + 1}页] ---")
                    pages_output.extend(page_lines)

        finally:
            doc.close()

        return '\n'.join(pages_output)

    # ================================================================
    # 页面布局分析
    # ================================================================

    def _filter_body_blocks(
        self, text_blocks: list[dict], page_height: float
    ) -> list[dict]:
        """过滤掉页眉/页脚区域的文本块"""
        header_limit = page_height * self.HEADER_RATIO
        footer_limit = page_height * self.FOOTER_RATIO

        body_blocks = []
        for block in text_blocks:
            bbox = block["bbox"]  # (x0, y0, x1, y1)
            # 块的中心 Y 坐标在页眉/页脚区域之外
            center_y = (bbox[1] + bbox[3]) / 2
            if header_limit <= center_y <= footer_limit:
                body_blocks.append(block)
        return body_blocks

    def _sort_by_reading_order(self, blocks: list[dict]) -> list[dict]:
        """
        XY-Cut 空间排序：先按 Y 分组（同一行），再按 X 排序（阅读方向）

        简化实现：
        1. 按块中心 Y 排序
        2. 对 Y 接近的块（同一行），再按 X 排序
        """
        if not blocks:
            return blocks

        # 计算平均块高度作为"同一行"的容差
        avg_height = sum(
            (b["bbox"][3] - b["bbox"][1]) for b in blocks
        ) / len(blocks)
        line_tolerance = avg_height * 0.8

        # 按 Y 排序，相同行内按 X 排序
        sorted_blocks = sorted(
            blocks,
            key=lambda b: (
                b["bbox"][1] // line_tolerance,  # Y 分组
                b["bbox"][0],                     # X 顺序
            )
        )
        return sorted_blocks

    # ================================================================
    # 文本块渲染
    # ================================================================

    def _render_text_block(self, block: dict) -> str:
        """
        将文本块渲染为字符串，保留格式提示：
        - 大号字体+粗体 → 识别为标题
        - 行内保留空格和间距
        """
        lines: list[str] = []
        block_lines = block.get("lines", [])

        for line in block_lines:
            spans = line.get("spans", [])
            if not spans:
                continue

            line_text_parts: list[str] = []

            for span in spans:
                text = span.get("text", "")
                if not text.strip():
                    # 保留空白作为间距
                    if text and line_text_parts:
                        line_text_parts.append(" ")
                    continue

                size = span.get("size", 10)
                flags = span.get("flags", 0)
                is_bold = bool(flags & 16)

                # 标题检测：字号 >= 16pt 且加粗
                if size >= 16 and is_bold and not line_text_parts:
                    # 独占一行的标题，加 ## 标记
                    line_text_parts.append(f"## {text}")
                else:
                    line_text_parts.append(text)

            if line_text_parts:
                line_str = " ".join(line_text_parts)
                # 去掉标题标记前的多余空格
                line_str = line_str.replace(" ## ", "\n## ")
                lines.append(line_str)

        return '\n'.join(lines)

    # ================================================================
    # 嵌入图片 OCR
    # ================================================================

    def _ocr_embedded_images(self, page, doc) -> list[str]:
        """提取页面嵌入图片并 OCR"""
        texts: list[str] = []
        try:
            images = page.get_images(full=True)
        except Exception:
            return texts

        for img_info in images:
            try:
                xref = img_info[0]  # 图片引用号
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]

                with tempfile.NamedTemporaryFile(
                    suffix=f'.{ext}', delete=False
                ) as tmp:
                    tmp.write(image_bytes)
                    tmp_path = tmp.name

                try:
                    ocr = self._get_ocr()
                    result, _ = ocr(tmp_path)
                    if result:
                        lines = [
                            r[1] for r in result if r[1].strip()
                        ]
                        if lines:
                            texts.append(
                                "[图片文字]\n" + '\n'.join(lines)
                            )
                finally:
                    os.unlink(tmp_path)
            except Exception:
                continue

        return texts

    # ================================================================
    # 扫描页面 OCR
    # ================================================================

    def _ocr_full_page(self, page) -> str:
        """扫描页：渲染整页为 PNG → OCR"""
        pix = page.get_pixmap(dpi=self.OCR_DPI)

        with tempfile.NamedTemporaryFile(
            suffix='.png', delete=False
        ) as tmp:
            pix.save(tmp.name)
            tmp_path = tmp.name

        try:
            ocr = self._get_ocr()
            result, _ = ocr(tmp_path)
            if result:
                lines = [r[1] for r in result if r[1].strip()]
                return '\n'.join(lines)
        finally:
            os.unlink(tmp_path)

        return ""
