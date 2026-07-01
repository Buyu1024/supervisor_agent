"""图片读取器 —— 基于 RapidOCR (ONNX Runtime) 提取图片中的文字

选择 RapidOCR 而非 PaddleOCR 的原因：
RapidOCR 基于 ONNX Runtime，无需 PaddlePaddle 依赖，体积小、安装快，
在 Windows 上兼容性更好，对中文识别效果同样出色。
"""

from pathlib import Path
from .base import BaseFileReader


class ImageReader(BaseFileReader):
    """
    读取图片文件，使用 RapidOCR 提取文字

    首次使用时会自动下载 ONNX 模型文件，后续使用缓存。
    """

    suffixes = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

    def __init__(self):
        self._ocr = None  # 延迟加载

    def _get_ocr(self):
        """延迟初始化 RapidOCR（首次调用时加载 ONNX 模型）"""
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr = RapidOCR()
        return self._ocr

    def read(self, path: Path) -> str:
        ocr = self._get_ocr()
        result, _ = ocr(str(path))

        # RapidOCR 返回格式: [[box, text, confidence], ...]
        if not result:
            return "[图片文件] 未识别到文字"

        lines = [item[1] for item in result if item[1].strip()]
        if not lines:
            return "[图片文件] 未识别到文字"

        return '\n'.join(lines)
