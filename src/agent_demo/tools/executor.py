"""ToolExecutor —— 工具执行器

职责：
    1. 参数校验（JSON Schema）
    2. 安全确认（回调注入）
    3. 超时控制
    4. 执行 handler + 异常捕获
    5. 结果序列化为 ToolResult

确认回调签名: Callable[[str, dict], bool]
    - 参数: (tool_name, arguments)
    - 返回: True = 允许执行, False = 拦截
"""

import time
import logging
import threading
from typing import Callable
from .types import ToolDef, ToolResult

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    工具执行器 —— 校验 → 确认 → 执行 → 包装

    确认回调通过构造函数注入，由外部（CLI / GUI / Web）决定如何向用户展示确认弹窗。
    不注入回调 = 所有工具自动执行，不拦截。
    """

    def __init__(self, confirm_callback: Callable | None = None):
        """
        Args:
            confirm_callback: (tool_name, arguments) -> bool
                              True 放行 / False 拦截
                              None 表示所有工具自动放行
        """
        self._confirm_callback = confirm_callback

    def execute(self, tool: ToolDef, arguments: dict) -> ToolResult:
        """
        执行工具并返回统一结果

        Args:
            tool: 工具定义（从 Registry 获取）
            arguments: LLM 传入的参数字典

        Returns:
            ToolResult（success=True 表示执行成功）
        """
        start_time = time.perf_counter()

        # 1. 参数校验
        validated, error_msg = self._validate_args(tool, arguments)
        if validated is None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                tool_name=tool.name,
                success=False,
                content=f"参数校验失败: {error_msg}",
                error=error_msg,
                metadata={"elapsed_ms": elapsed_ms},
            )

        # 2. 安全确认
        if tool.require_confirm and self._confirm_callback:
            try:
                allowed = self._confirm_callback(tool.name, validated)
            except Exception as e:
                allowed = False
                logger.warning(f"确认回调异常: {e}，默认拦截")
            if not allowed:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return ToolResult(
                    tool_name=tool.name,
                    success=False,
                    content=f"工具 '{tool.name}' 执行被用户取消。",
                    error="user_cancelled",
                    metadata={"elapsed_ms": elapsed_ms, "confirmed": False},
                )

        # 3. 超时控制执行
        try:
            result_text = self._run_with_timeout(tool, validated)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                tool_name=tool.name,
                success=True,
                content=str(result_text),
                metadata={"elapsed_ms": elapsed_ms},
            )

        except TimeoutError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                tool_name=tool.name,
                success=False,
                content=f"工具 '{tool.name}' 执行超时({tool.timeout}s)。",
                error="timeout",
                metadata={"elapsed_ms": elapsed_ms},
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"工具 '{tool.name}' 执行异常: {e}", exc_info=True)
            return ToolResult(
                tool_name=tool.name,
                success=False,
                content=f"工具 '{tool.name}' 执行出错: {e}",
                error=str(e),
                metadata={"elapsed_ms": elapsed_ms},
            )

    # ---- 内部方法 ----

    def _validate_args(self, tool: ToolDef, arguments: dict) -> tuple[dict | None, str | None]:
        """
        参数校验（简化版 JSON Schema 校验）

        校验内容:
            - 检查 required 字段是否齐全
            - 检查参数类型是否匹配

        Returns:
            (validated_dict, error_msg) —— 校验通过返回 (dict, None)，失败返回 (None, str)
        """
        schema = tool.parameters
        if not schema or schema.get("type") != "object":
            # 无参数定义，放行
            return arguments, None

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # 检查必填字段
        for field in required:
            if field not in arguments or arguments[field] is None:
                return None, f"缺少必填参数: '{field}'"

        # 类型转换与校验
        validated = {}
        for key, value in arguments.items():
            prop_schema = properties.get(key, {})
            expected_type = prop_schema.get("type", "string")

            try:
                if expected_type == "string":
                    validated[key] = str(value)
                elif expected_type == "number":
                    validated[key] = float(value)
                elif expected_type == "integer":
                    validated[key] = int(value)
                elif expected_type == "boolean":
                    if isinstance(value, bool):
                        validated[key] = value
                    elif str(value).lower() in ("true", "1"):
                        validated[key] = True
                    elif str(value).lower() in ("false", "0"):
                        validated[key] = False
                    else:
                        return None, f"参数 '{key}' 期望 boolean 类型，实际: {value}"
                elif expected_type == "array":
                    if isinstance(value, list):
                        validated[key] = value
                    elif isinstance(value, str):
                        # LLM 可能把数组序列化为 JSON 字符串
                        import json
                        validated[key] = json.loads(value)
                    else:
                        return None, f"参数 '{key}' 期望 array 类型"
                else:
                    validated[key] = value  # 未知类型，原样保留
            except (ValueError, TypeError, json.JSONDecodeError):
                return None, f"参数 '{key}' 类型转换失败: {value} → {expected_type}"

        # 保留 schema 中未定义但 LLM 传入的额外参数
        for key, value in arguments.items():
            if key not in validated:
                validated[key] = value

        return validated, None

    def _run_with_timeout(self, tool: ToolDef, arguments: dict) -> str:
        """带超时的执行 handler，超时抛出 TimeoutError"""
        result_holder = {}
        error_holder = {}

        def _target():
            try:
                result_holder["value"] = tool.handler(**arguments)
            except Exception as e:
                error_holder["error"] = e

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=tool.timeout)

        if thread.is_alive():
            raise TimeoutError(f"工具 '{tool.name}' 执行超时({tool.timeout}s)")

        if "error" in error_holder:
            raise error_holder["error"]

        return result_holder.get("value", "")
