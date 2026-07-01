"""LLM 模块 —— 集成测试 Demo"""

import os
import tempfile
from pathlib import Path

# 自动加载 .env 文件（如果存在）
_ENV_PATH = Path(__file__).parent.parent / ".env"
if _ENV_PATH.exists():
    with open(_ENV_PATH, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())


# ============================================================
# 测试辅助
# ============================================================

def _make_temp_file(content: str, suffix: str = ".txt") -> str:
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix=suffix, delete=False, encoding='utf-8'
    )
    tmp.write(content)
    tmp.close()
    return tmp.name


# ============================================================
# 测试用例
# ============================================================

class TestTypes:
    """LLMResponse 数据结构测试"""

    def test_basic_response(self):
        """基本 LLMResponse 创建与打印"""
        from agent_demo.llm import LLMResponse
        r = LLMResponse(
            content="你好，有什么可以帮你的？",
            finish_reason="stop",
            token_usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        )
        assert r.content == "你好，有什么可以帮你的？"
        assert r.finish_reason == "stop"
        assert r.token_usage["total_tokens"] == 60
        assert r.tool_calls_log == []
        print(f"  [PASS] LLMResponse 创建: {r}")

    def test_response_with_tools(self):
        """带工具调用日志的 LLMResponse"""
        from agent_demo.llm import LLMResponse
        r = LLMResponse(
            content="根据搜索结果...",
            finish_reason="stop",
            token_usage={"total_tokens": 200},
            tool_calls_log=[
                {"round": 1, "name": "search", "arguments": {"q": "天气"}, "result": "今天晴天"},
            ],
        )
        assert len(r.tool_calls_log) == 1
        assert r.tool_calls_log[0]["name"] == "search"
        print(f"  [PASS] LLMResponse 含工具日志: {r}")


class TestPromptBuilder:
    """PromptBuilder 消息转换测试"""

    def test_build_with_perception_message(self):
        """将感知模块 Message 转为 OpenAI 格式"""
        from agent_demo.llm import LLMResponse
        from agent_demo.perception import Message
        from agent_demo.llm.prompt_builder import PromptBuilder

        builder = PromptBuilder(system_prompt="你是一个有用的助手")
        user_msg = Message(content="你好", role="user", source_type="text")

        result = builder.build(messages=[user_msg])

        # 应有 2 条消息：system + user
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "有用的助手" in result[0]["content"]
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "你好"
        print(f"  [PASS] Message → OpenAI 格式: {len(result)} 条消息")

    def test_build_with_dict_messages(self):
        """兼容 dict 格式输入"""
        from agent_demo.llm.prompt_builder import PromptBuilder

        builder = PromptBuilder(system_prompt="测试")
        messages = [
            {"role": "user", "content": "问题1"},
            {"role": "assistant", "content": "回答1"},
            {"role": "user", "content": "问题2"},
        ]

        result = builder.build(messages=messages)

        assert len(result) == 4  # system + 3 条对话
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        print(f"  [PASS] Dict 格式兼容: {len(result)} 条消息")

    def test_build_with_context(self):
        """上下文注入 —— 拼入 system prompt"""
        from agent_demo.llm.prompt_builder import PromptBuilder

        builder = PromptBuilder(system_prompt="你是助手")
        context = "用户偏好：喜欢简洁的回答"

        result = builder.build(
            messages=[{"role": "user", "content": "hi"}],
            context=context,
        )

        system_content = result[0]["content"]
        assert "你是助手" in system_content
        assert "用户偏好" in system_content
        assert "记忆模块" in system_content  # 上下文标签
        print(f"  [PASS] 上下文注入: {len(system_content)} 字符")

    def test_no_system_prompt(self):
        """无系统提示词时，不生成 system 消息"""
        from agent_demo.llm.prompt_builder import PromptBuilder

        builder = PromptBuilder()  # 无 system_prompt
        result = builder.build(messages=[{"role": "user", "content": "hi"}])

        # 只有 user 消息，没有 system
        assert result[0]["role"] == "user"
        print(f"  [PASS] 无 system prompt: {len(result)} 条消息")


class TestQwenClient:
    """QwenClient 初始化测试（离线）"""

    def test_no_api_key_raises(self):
        """未设置 API Key 时抛异常"""
        from agent_demo.llm.client import QwenClient

        # 临时移除环境变量
        saved = os.environ.pop("DASHSCOPE_API_KEY", None)
        try:
            QwenClient(api_key=None)  # 无参数 + 无环境变量 → 应抛 ValueError
            assert False, "应抛出 ValueError"
        except ValueError as e:
            assert "API Key" in str(e)
            print(f"  [PASS] 无 API Key 抛异常: {e}")
        finally:
            if saved:
                os.environ["DASHSCOPE_API_KEY"] = saved

    def test_explicit_api_key(self):
        """显式传入 API Key 创建客户端"""
        from agent_demo.llm.client import QwenClient

        client = QwenClient(api_key="sk-test-key-12345")
        assert client.model == "qwen3.7-plus"
        print(f"  [PASS] 显式 API Key: model={client.model}")


class TestLLMModuleInit:
    """LLMModule 初始化测试（离线）"""

    def test_init_with_api_key(self):
        """显式 API Key 初始化"""
        from agent_demo.llm import LLMModule

        llm = LLMModule(
            api_key="sk-test",
            system_prompt="你是测试助手",
        )
        assert llm.model == "qwen3.7-plus"
        assert llm._prompt_builder.system_prompt == "你是测试助手"
        print(f"  [PASS] LLMModule 初始化: model={llm.model}")

    def test_init_without_api_key(self):
        """无 API Key 初始化（环境变量也不存在时抛异常）"""
        from agent_demo.llm import LLMModule

        saved = os.environ.pop("DASHSCOPE_API_KEY", None)
        try:
            LLMModule(api_key=None)
            assert False, "应抛出 ValueError"
        except ValueError as e:
            print(f"  [PASS] 无 API Key 初始化抛异常: {e}")
        finally:
            if saved:
                os.environ["DASHSCOPE_API_KEY"] = saved

    def test_custom_model(self):
        """自定义模型名称"""
        from agent_demo.llm import LLMModule

        llm = LLMModule(api_key="sk-test", model="qwen-max")
        assert llm.model == "qwen-max"
        print(f"  [PASS] 自定义模型: {llm.model}")


class TestAPI:
    """真实 API 调用测试（需要 DASHSCOPE_API_KEY 环境变量）"""

    def test_simple_chat(self):
        """纯对话（无工具）—— 需要 API Key"""
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            print("  [SKIP] 简单对话: 未设置 DASHSCOPE_API_KEY，跳过真实 API 测试")
            return

        from agent_demo.llm import LLMModule
        from agent_demo.perception import Message

        llm = LLMModule(system_prompt="用简洁的中文回答，不超过两句话。")
        user_msg = Message(content="你好，请用一句话介绍你自己", role="user")

        response = llm.chat(messages=[user_msg])

        assert response.content, "回复内容不应为空"
        assert response.finish_reason == "stop"
        assert response.token_usage["total_tokens"] > 0
        print(f"  [PASS] 简单对话: {response}")
        print(f"    回复内容: {response.content[:150]}...")

    def test_multi_turn_chat(self):
        """多轮对话 —— 需要 API Key"""
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            print("  [SKIP] 多轮对话: 未设置 DASHSCOPE_API_KEY，跳过真实 API 测试")
            return

        from agent_demo.llm import LLMModule
        from agent_demo.perception import Message

        llm = LLMModule(system_prompt="用中文简短回答")

        # 第一轮：告诉模型名字
        r1 = llm.chat(messages=[Message(content="我叫张三", role="user")])
        assert r1.content
        print(f"  [PASS] 多轮对话-第1轮: {r1}")

        # 第二轮：验证模型记住了名字（内部 _history 自动拼接）
        r2 = llm.chat(messages=[Message(content="我叫什么名字？", role="user")])
        assert r2.content
        assert len(llm._history) >= 3  # system + user + assistant
        # 验证模型确实记住了上下文（"张三"应出现在回复中）
        assert "张三" in r2.content, f"模型应记住名字'张三'，实际回复: {r2.content[:100]}"
        print(f"  [PASS] 多轮对话-第2轮: {r2}")
        print(f"    回复内容: {r2.content[:150]}...")

    def test_context_update(self):
        """多轮对话中 context 更新应生效 —— 需要 API Key"""
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            print("  [SKIP] context 更新: 未设置 DASHSCOPE_API_KEY，跳过真实 API 测试")
            return

        from agent_demo.llm import LLMModule
        from agent_demo.perception import Message

        llm = LLMModule(system_prompt="用中文简短回答")

        # 第一轮：无上下文
        r1 = llm.chat(
            messages=[Message(content="你好", role="user")],
            context="用户偏好：喜欢用英文回答",
        )
        assert r1.content
        # system 消息应包含上下文
        assert "用户偏好" in llm._history[0]["content"]
        print(f"  [PASS] context 更新-第1轮(含上下文): {r1}")

        # 第二轮：更新上下文
        r2 = llm.chat(
            messages=[Message(content="我刚才让你用什么语言回答？", role="user")],
            context="用户偏好：喜欢用中文回答",
        )
        assert r2.content
        # system 消息应被新的上下文替换
        system_content = llm._history[0]["content"]
        assert "中文" in system_content, f"system 消息应更新为新 context，实际: {system_content}"
        print(f"  [PASS] context 更新-第2轮(上下文已切换): {r2}")
        print(f"    回复内容: {r2.content[:150]}...")

    def test_with_file_input(self):
        """感知模块 + LLM 模块串联 —— 需要 API Key"""
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            print("  [SKIP] 感知+LLM 串联: 未设置 DASHSCOPE_API_KEY，跳过真实 API 测试")
            return

        from agent_demo.llm import LLMModule
        from agent_demo.perception import PerceptionModule

        # 感知模块处理文件
        pm = PerceptionModule()
        file_path = _make_temp_file("请用一句话总结这个文件的内容。", suffix=".txt")
        try:
            msg = pm.process(file_path)
            assert not msg.is_rejected
            assert msg.source_type == "file"

            # LLM 模块处理
            llm = LLMModule(system_prompt="用中文回复，不超过两句话。")
            response = llm.chat(messages=[msg])

            assert response.content
            print(f"  [PASS] 感知+LLM 串联: {response}")
            print(f"    回复内容: {response.content[:150]}...")
        finally:
            os.unlink(file_path)


# ============================================================
# 运行入口
# ============================================================

def run_all():
    """运行所有测试并汇总结果"""
    import sys

    test_classes = [
        TestTypes,
        TestPromptBuilder,
        TestQwenClient,
        TestLLMModuleInit,
        TestAPI,
    ]

    total = 0
    passed = 0
    failed = 0

    print("=" * 60)
    print("LLM 模块 (LLM Module) 测试")
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
