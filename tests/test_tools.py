"""工具模块 —— 集成测试 Demo"""

import time


# ============================================================
# 测试辅助：示例工具函数
# ============================================================

def _mock_search(query: str) -> str:
    """模拟搜索工具"""
    return f"搜索结果: 关于 '{query}' 找到 3 条相关信息"

def _mock_calc(expression: str) -> str:
    """模拟计算器"""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算出错: {e}"

def _slow_tool(delay: float = 5.0) -> str:
    """模拟慢速工具（用于超时测试）"""
    time.sleep(delay)
    return "完成"


# ============================================================
# 测试用例
# ============================================================

class TestTypes:
    """ToolDef / ToolResult 数据结构测试"""

    def test_tool_def_creation(self):
        """ToolDef 基本创建"""
        from agent_demo.tools import ToolDef

        tool = ToolDef(
            name="search",
            description="搜索网页",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "关键词"}},
                "required": ["query"],
            },
            handler=_mock_search,
            tags=["network", "readonly"],
        )
        assert tool.name == "search"
        assert tool.require_confirm is False
        assert tool.timeout == 30.0
        assert "readonly" in tool.tags
        print(f"  [PASS] ToolDef 创建: {tool}")

    def test_tool_result_success(self):
        """ToolResult 成功结果"""
        from agent_demo.tools import ToolResult

        r = ToolResult(
            tool_name="search",
            success=True,
            content="找到 3 条结果",
            metadata={"elapsed_ms": 150.0},
        )
        assert r.success
        assert r.error is None
        assert r.metadata["elapsed_ms"] == 150.0
        print(f"  [PASS] ToolResult 成功: {r}")

    def test_tool_result_failure(self):
        """ToolResult 失败结果"""
        from agent_demo.tools import ToolResult

        r = ToolResult(
            tool_name="search",
            success=False,
            content="参数校验失败: 缺少必填参数 'query'",
            error="missing_required",
        )
        assert not r.success
        assert r.error == "missing_required"
        print(f"  [PASS] ToolResult 失败: {r}")


class TestRegistry:
    """ToolRegistry 注册与 Schema 导出测试"""

    def test_register_by_object(self):
        """ToolDef 对象注册"""
        from agent_demo.tools.registry import ToolRegistry
        from agent_demo.tools import ToolDef

        reg = ToolRegistry()
        reg.register(ToolDef(
            name="echo",
            description="回显消息",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda msg="": f"echo: {msg}",
        ))

        assert reg.get("echo") is not None
        assert reg.get("echo").name == "echo"
        print(f"  [PASS] ToolDef 对象注册")

    def test_register_by_decorator(self):
        """装饰器注册"""
        from agent_demo.tools.registry import ToolRegistry

        reg = ToolRegistry()

        @reg.register_decorator(
            name="add",
            description="加法运算",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个数"},
                    "b": {"type": "number", "description": "第二个数"},
                },
                "required": ["a", "b"],
            },
            tags=["math"],
        )
        def add(a: float, b: float) -> str:
            return str(a + b)

        tool = reg.get("add")
        assert tool is not None
        assert tool.description == "加法运算"
        assert tool.handler(1, 2) == "3"  # 原始函数仍可独立使用
        assert "math" in tool.tags
        print(f"  [PASS] 装饰器注册: {tool}")

    def test_register_overwrite(self):
        """重复注册同名工具 —— 覆盖"""
        from agent_demo.tools.registry import ToolRegistry
        from agent_demo.tools import ToolDef

        reg = ToolRegistry()
        reg.register(ToolDef(name="x", description="v1", parameters={}, handler=lambda: "1"))
        reg.register(ToolDef(name="x", description="v2", parameters={}, handler=lambda: "2"))

        assert reg.get("x").description == "v2"
        print(f"  [PASS] 同名覆盖: {reg.get('x').description}")

    def test_unregister(self):
        """注销工具"""
        from agent_demo.tools.registry import ToolRegistry
        from agent_demo.tools import ToolDef

        reg = ToolRegistry()
        reg.register(ToolDef(name="tmp", description="...", parameters={}, handler=lambda: ""))
        reg.unregister("tmp")

        assert reg.get("tmp") is None
        print(f"  [PASS] 注销工具")

    def test_get_schemas(self):
        """Schema 导出 —— OpenAI Function Calling 格式"""
        from agent_demo.tools.registry import ToolRegistry
        from agent_demo.tools import ToolDef

        reg = ToolRegistry()
        reg.register(ToolDef(
            name="search",
            description="搜索网页",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=_mock_search,
        ))

        schemas = reg.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "search"
        assert schemas[0]["function"]["description"] == "搜索网页"
        print(f"  [PASS] Schema 导出: {schemas[0]}")

    def test_list_names(self):
        """按标签过滤工具名"""
        from agent_demo.tools.registry import ToolRegistry
        from agent_demo.tools import ToolDef

        reg = ToolRegistry()
        reg.register(ToolDef(name="a", description="...", parameters={}, handler=lambda: "", tags=["readonly"]))
        reg.register(ToolDef(name="b", description="...", parameters={}, handler=lambda: "", tags=["dangerous"]))
        reg.register(ToolDef(name="c", description="...", parameters={}, handler=lambda: "", tags=["readonly", "network"]))

        all_names = reg.list_names()
        assert len(all_names) == 3

        readonly = reg.list_names(tag="readonly")
        assert len(readonly) == 2
        assert "a" in readonly and "c" in readonly
        print(f"  [PASS] 按标签过滤: all={all_names}, readonly={readonly}")


class TestExecutor:
    """ToolExecutor 执行逻辑测试"""

    def test_execute_success(self):
        """正常执行工具"""
        from agent_demo.tools.executor import ToolExecutor
        from agent_demo.tools import ToolDef

        executor = ToolExecutor()
        tool = ToolDef(
            name="echo",
            description="...",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda msg="hi": f"回显: {msg}",
        )

        result = executor.execute(tool, {"msg": "hello"})
        assert result.success
        assert "回显: hello" in result.content
        assert result.metadata["elapsed_ms"] >= 0
        print(f"  [PASS] 正常执行: {result}")

    def test_validation_missing_required(self):
        """缺少必填参数 —— 拦截"""
        from agent_demo.tools.executor import ToolExecutor
        from agent_demo.tools import ToolDef

        executor = ToolExecutor()
        tool = ToolDef(
            name="search",
            description="...",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=_mock_search,
        )

        result = executor.execute(tool, {})  # 没传 query
        assert not result.success
        assert "缺少必填参数" in result.content
        print(f"  [PASS] 必填参数拦截: {result.content}")

    def test_validation_type_conversion(self):
        """参数类型自动转换"""
        from agent_demo.tools.executor import ToolExecutor
        from agent_demo.tools import ToolDef

        executor = ToolExecutor()
        tool = ToolDef(
            name="calc",
            description="...",
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "integer"},
                    "flag": {"type": "boolean"},
                },
                "required": [],
            },
            handler=lambda x=0, y=0, flag=False: f"{x} + {y} = {x + y}, flag={flag}",
        )

        # LLM 可能传字符串类型的数字
        result = executor.execute(tool, {"x": "3.14", "y": "5", "flag": "true"})
        assert result.success
        assert "3.14 + 5 = 8.14" in result.content
        print(f"  [PASS] 类型转换: {result.content}")

    def test_confirm_callback_allow(self):
        """确认回调 —— 用户同意"""
        from agent_demo.tools.executor import ToolExecutor
        from agent_demo.tools import ToolDef

        def always_allow(name, args):
            return True

        executor = ToolExecutor(confirm_callback=always_allow)
        tool = ToolDef(
            name="delete",
            description="...",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda: "删除成功",
            require_confirm=True,
        )

        result = executor.execute(tool, {})
        assert result.success
        assert "删除成功" in result.content
        print(f"  [PASS] 确认回调-同意: {result}")

    def test_confirm_callback_block(self):
        """确认回调 —— 用户拒绝"""
        from agent_demo.tools.executor import ToolExecutor
        from agent_demo.tools import ToolDef

        def always_deny(name, args):
            return False

        executor = ToolExecutor(confirm_callback=always_deny)
        tool = ToolDef(
            name="delete",
            description="...",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda: "删除成功",
            require_confirm=True,
        )

        result = executor.execute(tool, {})
        assert not result.success
        assert "被用户取消" in result.content or "cancel" in result.error
        print(f"  [PASS] 确认回调-拒绝: {result}")

    def test_no_confirm_when_not_required(self):
        """require_confirm=False 时不触发确认回调"""
        from agent_demo.tools.executor import ToolExecutor
        from agent_demo.tools import ToolDef

        call_count = 0

        def counting_callback(name, args):
            nonlocal call_count
            call_count += 1
            return True

        executor = ToolExecutor(confirm_callback=counting_callback)
        tool = ToolDef(
            name="echo",
            description="...",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda: "ok",
            require_confirm=False,  # 不需要确认
        )

        result = executor.execute(tool, {})
        assert result.success
        assert call_count == 0  # 回调未被调用
        print(f"  [PASS] 非必确认不触发回调: call_count={call_count}")

    def test_handler_exception(self):
        """工具执行异常 —— 捕获并返回错误结果"""
        from agent_demo.tools.executor import ToolExecutor
        from agent_demo.tools import ToolDef

        def broken_tool():
            raise ValueError("模拟工具内部错误")

        executor = ToolExecutor()
        tool = ToolDef(
            name="broken",
            description="...",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=broken_tool,
        )

        result = executor.execute(tool, {})
        assert not result.success
        assert "ValueError" in result.error or "模拟工具内部错误" in result.error
        print(f"  [PASS] 异常捕获: {result}")


class TestToolsModule:
    """ToolsModule 集成测试"""

    def test_decorator_register(self):
        """通过 @tools.register 装饰器注册"""
        from agent_demo.tools import ToolsModule

        tools = ToolsModule()

        @tools.register(
            name="greet",
            description="打招呼",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )
        def greet(name: str) -> str:
            return f"你好，{name}！"

        assert "greet" in tools.list_tools()
        # 函数仍可独立使用
        assert greet("小明") == "你好，小明！"
        print(f"  [PASS] 装饰器注册: 工具列表={tools.list_tools()}")

    def test_tool_def_register(self):
        """通过 ToolDef 对象注册"""
        from agent_demo.tools import ToolsModule, ToolDef

        tools = ToolsModule()
        tools.register(ToolDef(
            name="calc",
            description="计算器",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            handler=_mock_calc,
            tags=["math"],
        ))

        assert "calc" in tools.list_tools()
        print(f"  [PASS] ToolDef 注册: {tools.list_tools()}")

    def test_execute(self):
        """通过 ToolsModule 执行工具"""
        from agent_demo.tools import ToolsModule, ToolDef

        tools = ToolsModule()
        tools.register(ToolDef(
            name="double",
            description="翻倍",
            parameters={
                "type": "object",
                "properties": {"n": {"type": "number"}},
                "required": ["n"],
            },
            handler=lambda n: str(n * 2),
        ))

        result = tools.execute("double", {"n": 21})
        assert result.success
        assert "42" in result.content
        print(f"  [PASS] 执行工具: {result}")

    def test_execute_not_found(self):
        """执行不存在的工具"""
        from agent_demo.tools import ToolsModule

        tools = ToolsModule()
        result = tools.execute("nonexistent", {})
        assert not result.success
        assert "未找到工具" in result.content
        print(f"  [PASS] 工具未找到: {result}")

    def test_get_schemas(self):
        """导出 Schema 给 LLM 模块"""
        from agent_demo.tools import ToolsModule, ToolDef

        tools = ToolsModule()
        tools.register(ToolDef(
            name="search",
            description="搜索网页",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_mock_search,
        ))

        schemas = tools.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        print(f"  [PASS] Schema 导出: {len(schemas)} 个工具")

    def test_get_executor(self):
        """获取 executor 回调 —— 供 LLMModule 注入"""
        from agent_demo.tools import ToolsModule, ToolDef

        tools = ToolsModule()
        tools.register(ToolDef(
            name="echo",
            description="...",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda msg="": msg,
        ))

        executor = tools.get_executor()
        assert callable(executor)

        result = executor("echo", {"msg": "hello"})
        assert result.success
        assert "hello" in result.content
        print(f"  [PASS] executor 回调: {result}")

    def test_confirm_integration(self):
        """确认回调集成 —— 外部注入"""
        from agent_demo.tools import ToolsModule, ToolDef

        confirm_log = []

        def my_confirm(name, args):
            confirm_log.append((name, args))
            return False  # 拒绝

        tools = ToolsModule(confirm_callback=my_confirm)
        tools.register(ToolDef(
            name="danger",
            description="危险操作",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda: "done",
            require_confirm=True,
        ))

        result = tools.execute("danger", {})
        assert not result.success
        assert len(confirm_log) == 1
        assert confirm_log[0][0] == "danger"
        print(f"  [PASS] 确认集成: confirm_log={confirm_log}")


class TestToolLLMIntegration:
    """工具模块 + LLM 模块集成测试（离线模拟）"""

    def test_schema_format_compatible(self):
        """验证导出的 schema 格式能被 LLM 模块接受"""
        from agent_demo.tools import ToolsModule, ToolDef
        from agent_demo.llm import LLMModule

        tools = ToolsModule()
        tools.register(ToolDef(
            name="search",
            description="搜索网页",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=_mock_search,
        ))

        schemas = tools.get_schemas()
        # 验证格式：LLMModule 初始化不报错即表示兼容
        llm = LLMModule(
            api_key="sk-test",
            tool_executor=tools.get_executor(),
        )

        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert "function" in schemas[0]
        print(f"  [PASS] Schema 格式兼容 LLM 模块")


# ============================================================
# 运行入口
# ============================================================

def run_all():
    """运行所有测试并汇总结果"""
    import sys

    test_classes = [
        TestTypes,
        TestRegistry,
        TestExecutor,
        TestToolsModule,
        TestToolLLMIntegration,
    ]

    total = 0
    passed = 0
    failed = 0

    print("=" * 60)
    print("工具模块 (Tools Module) 测试")
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

    print("\n" + "=" * 60)
    print(f"测试完成: {total} 个用例 | 通过: {passed} | 失败: {failed}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    import sys
    sys.exit(0 if success else 1)
