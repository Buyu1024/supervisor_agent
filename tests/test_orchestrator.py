"""编排器模块 —— 单元测试

测试范围:
    - 配置与初始化
    - 感知拦截
    - 计划生成（含解析失败重试）
    - 简单对话执行
    - 多步骤计划执行
    - 重规划流程
    - 记忆读写集成
    - 错误处理
    - 完整端到端

所有 LLM 调用均通过 Mock 模拟，不实际调用 API。
"""

import json
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

sys.path.insert(0, ".")

from src.agent_demo.orchestrator import AgentOrchestrator, OrchestratorConfig, OrchestratorResult
from src.agent_demo.llm import LLMModule, LLMResponse
from src.agent_demo.planning import PlanningModule, TaskPlan, TaskStep, StepResult
from src.agent_demo.memory import MemoryModule
from src.agent_demo.perception import PerceptionModule
from src.agent_demo.tools import ToolDef

# 测试用 API Key（QwenClient 初始化需要，但实际不会调用 API）
TEST_API_KEY = "sk-test-fake-key-for-unittest"


# ── 辅助函数 ────────────────────────────────────────────────────

def make_test_config(**kwargs) -> OrchestratorConfig:
    """创建测试用配置，默认带 fake API key"""
    defaults = {"api_key": TEST_API_KEY}
    defaults.update(kwargs)
    return OrchestratorConfig(**defaults)


def make_plan_json(goal="测试任务", steps=None):
    """生成标准计划 JSON"""
    if steps is None:
        steps = [
            {
                "id": "step_1",
                "description": "分析用户请求",
                "instruction": "理解用户的请求并制定回复策略。",
                "action": "think",
                "depends_on": [],
            },
            {
                "id": "step_2",
                "description": "生成回复",
                "instruction": "基于分析结果生成最终回复。",
                "action": "respond",
                "depends_on": ["step_1"],
            },
        ]
    return json.dumps({"goal": goal, "steps": steps}, ensure_ascii=False)


def make_chat_response(content, finish_reason="stop", tool_calls_log=None):
    """生成标准 LLMResponse"""
    return LLMResponse(
        content=content,
        finish_reason=finish_reason,
        token_usage={"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
        tool_calls_log=tool_calls_log or [],
    )


def make_agent(**config_kwargs) -> AgentOrchestrator:
    """创建测试用 AgentOrchestrator"""
    return AgentOrchestrator(make_test_config(**config_kwargs))


# ── 测试类 ──────────────────────────────────────────────────────

class TestConfig(unittest.TestCase):
    """配置相关测试"""

    def test_default_config(self):
        """默认配置"""
        config = OrchestratorConfig()
        self.assertIsNone(config.api_key)
        self.assertEqual(config.model, "qwen3.7-plus")
        self.assertEqual(config.max_tool_rounds, 10)
        self.assertEqual(config.max_working_tokens, 8000)
        self.assertEqual(config.max_retries_per_step, 2)
        self.assertEqual(config.max_revisions, 3)
        self.assertEqual(config.max_input_length, 4000)
        self.assertEqual(config.embedder_provider, "dashscope")

    def test_custom_config(self):
        """自定义配置"""
        config = OrchestratorConfig(
            api_key="sk-test",
            model="qwen-turbo",
            max_working_tokens=4000,
            max_retries_per_step=1,
        )
        self.assertEqual(config.api_key, "sk-test")
        self.assertEqual(config.model, "qwen-turbo")
        self.assertEqual(config.max_working_tokens, 4000)
        self.assertEqual(config.max_retries_per_step, 1)

    def test_orchestrator_result_repr(self):
        """OrchestratorResult 字符串表示"""
        result = OrchestratorResult(
            content="测试回复",
            success=True,
            total_tool_calls=3,
            total_tokens={"total_tokens": 150},
        )
        r = repr(result)
        self.assertIn("success=True", r)
        self.assertIn("tools=3", r)
        self.assertIn("tokens=150", r)


class TestOrchestratorInit(unittest.TestCase):
    """编排器初始化测试"""

    def test_init_with_defaults(self):
        """默认初始化"""
        agent = make_agent()
        self.assertIsNotNone(agent.perception)
        self.assertIsNotNone(agent.memory)
        self.assertIsNotNone(agent.planning)
        self.assertIsNotNone(agent.llm)
        self.assertIsNotNone(agent.tools)

    def test_init_with_config(self):
        """自定义配置初始化"""
        agent = make_agent(
            model="qwen-turbo",
            max_retries_per_step=1,
            max_revisions=1,
        )
        self.assertEqual(agent.config.model, "qwen-turbo")
        self.assertEqual(agent.planning._executor.max_retries_per_step, 1)

    def test_module_access_readonly(self):
        """模块属性为只读（property）"""
        agent = make_agent()
        with self.assertRaises(AttributeError):
            agent.perception = None

    def test_repr(self):
        """字符串表示"""
        agent = make_agent()
        r = repr(agent)
        self.assertIn("AgentOrchestrator", r)
        self.assertIn("model=", r)

    def test_clear_session(self):
        """清空会话不应抛出异常"""
        agent = make_agent()
        agent.clear_session()


class TestPerceptionReject(unittest.TestCase):
    """感知拦截测试"""

    def test_rejected_input_returns_error(self):
        """被拒绝的输入应直接返回错误结果，不调用 LLM"""
        import tempfile, os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("违禁词\nbadword\n")
            tmp_path = f.name

        try:
            with patch.object(LLMModule, "chat") as mock_chat:
                agent = make_agent()
                # 替换 PerceptionModule 为带敏感词过滤的版本
                agent._perception = PerceptionModule(
                    max_length=4000,
                    sensitive_words_path=tmp_path,
                )

                result = agent.run("包含违禁词的输入")

                self.assertFalse(result.success)
                self.assertIn("违规", result.content)
                self.assertIsNotNone(result.error)
                mock_chat.assert_not_called()  # LLM 未被调用
        finally:
            os.unlink(tmp_path)

    def test_normal_input_not_rejected(self):
        """正常输入不应被拦截"""
        pm = PerceptionModule(max_length=4000)  # 无敏感词文件
        msg = pm.process("你好世界")
        self.assertFalse(msg.is_rejected)


class TestPlanGeneration(unittest.TestCase):
    """计划生成测试"""

    def test_generate_simple_plan(self):
        """生成简单计划并执行"""
        with patch.object(LLMModule, "chat") as mock_chat:
            mock_chat.side_effect = [
                make_chat_response(make_plan_json(goal="打招呼")),
                make_chat_response("用户想打个招呼，我应该友好回应。"),
                make_chat_response("你好！有什么可以帮你的吗？"),
            ]

            agent = make_agent()
            result = agent.run("你好")

            self.assertTrue(result.success)
            self.assertIsNotNone(result.plan)
            self.assertEqual(result.plan.goal, "打招呼")
            self.assertEqual(len(result.plan.steps), 2)
            self.assertEqual(mock_chat.call_count, 3)

    def test_plan_parse_retry_on_failure(self):
        """计划 JSON 解析失败时自动重试"""
        with patch.object(LLMModule, "chat") as mock_chat:
            mock_chat.side_effect = [
                make_chat_response("这不是 JSON，只是随便说说"),
                make_chat_response(make_plan_json(goal="测试", steps=[
                    {"id": "s1", "description": "做测试", "instruction": "执行测试",
                     "action": "respond", "depends_on": []}
                ])),
                make_chat_response("测试完成"),
            ]

            agent = make_agent()
            result = agent.run("做个测试")

            self.assertTrue(result.success)
            self.assertEqual(mock_chat.call_count, 3)  # 2 次计划 + 1 次执行

    def test_plan_parse_exhausts_retries(self):
        """计划解析重试耗尽后返回错误"""
        with patch.object(LLMModule, "chat") as mock_chat:
            mock_chat.side_effect = [
                make_chat_response("无效 JSON #1"),
                make_chat_response("无效 JSON #2"),
            ]

            agent = make_agent()
            result = agent.run("测试")

            self.assertFalse(result.success)
            self.assertIsNotNone(result.error)
            self.assertIn("计划生成", result.content)

    def test_memory_context_in_plan_generation(self):
        """验证记忆上下文被传入计划生成 prompt"""
        with patch.object(LLMModule, "chat") as mock_chat:
            mock_chat.side_effect = [
                make_chat_response(make_plan_json(goal="搜索任务")),
                make_chat_response("搜索完成"),
                make_chat_response("结果已整理"),
            ]

            agent = make_agent()

            # 注入带预设上下文的 mock memory
            mock_mem = MagicMock()
            mock_mem.retrieve.return_value = "用户偏好: 语言=中文"
            agent._memory = mock_mem

            agent.run("搜索一下")

            # 验证 retrieve 被调用
            mock_mem.retrieve.assert_called_once()
            # 验证 plan prompt 中包含上下文
            first_call_msg = mock_chat.call_args_list[0][1]["messages"][0]["content"]
            self.assertIn("用户偏好: 语言=中文", first_call_msg)

    def test_tool_schemas_in_plan_prompt(self):
        """验证工具 schema 被包含在计划生成 prompt 中"""
        with patch.object(LLMModule, "chat") as mock_chat:
            mock_chat.side_effect = [
                make_chat_response(make_plan_json()),
                make_chat_response("done"),
                make_chat_response("final"),
            ]

            agent = make_agent()

            # 注册一个工具
            agent.tools.register(ToolDef(
                name="test_tool",
                description="测试工具",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=lambda: "ok",
            ))

            agent.run("测试")

            # 验证 prompt 中包含工具名
            first_call_msg = mock_chat.call_args_list[0][1]["messages"][0]["content"]
            self.assertIn("test_tool", first_call_msg)


class TestSimpleExecution(unittest.TestCase):
    """简单对话执行测试"""

    def test_single_step_respond(self):
        """单步 respond 计划"""
        plan_json = make_plan_json(goal="简单问答", steps=[
            {"id": "s1", "description": "回答用户", "instruction": "直接回答用户的问题。",
             "action": "respond", "depends_on": []}
        ])

        with patch.object(LLMModule, "chat") as mock_chat:
            mock_chat.side_effect = [
                make_chat_response(plan_json),
                make_chat_response("北京今天晴，18-26°C，适合出行。"),
            ]

            agent = make_agent()
            result = agent.run("北京天气怎么样")

            self.assertTrue(result.success)
            self.assertIn("北京", result.content)
            self.assertEqual(mock_chat.call_count, 2)  # 1 plan + 1 step

    def test_step_with_tool_calls(self):
        """工具调用步骤（LLM 内部多次工具调用）"""
        plan_json = make_plan_json(goal="搜索任务", steps=[
            {"id": "s1", "description": "搜索信息", "instruction": "搜索相关信息。",
             "action": "tool_call", "depends_on": []},
            {"id": "s2", "description": "总结", "instruction": "总结结果。",
             "action": "respond", "depends_on": ["s1"]},
        ])

        with patch.object(LLMModule, "chat") as mock_chat:
            mock_chat.side_effect = [
                make_chat_response(plan_json),
                # step 1（tool_call）: LLM 内部调用了 3 次工具
                make_chat_response("搜索完成", tool_calls_log=[
                    {"round": 1, "name": "search", "arguments": {}, "result": "r1"},
                    {"round": 2, "name": "search", "arguments": {}, "result": "r2"},
                    {"round": 3, "name": "fetch", "arguments": {}, "result": "r3"},
                ]),
                # step 2（respond）
                make_chat_response("总结完毕"),
            ]

            agent = make_agent()
            result = agent.run("搜索信息")

            self.assertTrue(result.success)
            self.assertEqual(result.total_tool_calls, 3)


class TestMultiStepExecution(unittest.TestCase):
    """多步骤计划执行测试"""

    def test_sequential_steps(self):
        """顺序依赖步骤执行"""
        plan_json = make_plan_json(goal="三步任务", steps=[
            {"id": "s1", "description": "第一步", "instruction": "做第一步。",
             "action": "think", "depends_on": []},
            {"id": "s2", "description": "第二步", "instruction": "做第二步。",
             "action": "tool_call", "depends_on": ["s1"]},
            {"id": "s3", "description": "第三步", "instruction": "做第三步。",
             "action": "respond", "depends_on": ["s2"]},
        ])

        with patch.object(LLMModule, "chat") as mock_chat:
            mock_chat.side_effect = [
                make_chat_response(plan_json),               # plan gen
                make_chat_response("第一步完成"),              # s1
                make_chat_response("第二步完成"),              # s2
                make_chat_response("第三步完成，全部结束"),     # s3
            ]

            agent = make_agent()
            result = agent.run("执行三步任务")

            self.assertTrue(result.success)
            self.assertEqual(len(result.plan.get_completed_steps()), 3)
            self.assertEqual(len(result.plan.get_failed_steps()), 0)
            self.assertEqual(mock_chat.call_count, 4)

    def test_parallel_steps(self):
        """并行步骤（无依赖）"""
        plan_json = make_plan_json(goal="并行任务", steps=[
            {"id": "s1", "description": "任务 A", "instruction": "做任务 A。",
             "action": "think", "depends_on": []},
            {"id": "s2", "description": "任务 B", "instruction": "做任务 B。",
             "action": "think", "depends_on": []},
            {"id": "s3", "description": "汇总", "instruction": "汇总 A 和 B 的结果。",
             "action": "respond", "depends_on": ["s1", "s2"]},
        ])

        with patch.object(LLMModule, "chat") as mock_chat:
            mock_chat.side_effect = [
                make_chat_response(plan_json),
                make_chat_response("任务 A 完成"),
                make_chat_response("任务 B 完成"),
                make_chat_response("汇总完成"),
            ]

            agent = make_agent()
            result = agent.run("并行执行")

            self.assertTrue(result.success)
            self.assertEqual(len(result.plan.get_completed_steps()), 3)

    def test_dependency_order_respected(self):
        """验证依赖顺序被遵守 —— s3 在 s1, s2 完成后才执行"""
        plan_json = make_plan_json(goal="依赖测试", steps=[
            {"id": "s1", "description": "准备数据", "instruction": "准备。",
             "action": "think", "depends_on": []},
            {"id": "s2", "description": "分析数据", "instruction": "基于 s1 的结果分析。",
             "action": "think", "depends_on": ["s1"]},
            {"id": "s3", "description": "最终报告", "instruction": "生成报告。",
             "action": "respond", "depends_on": ["s2"]},
        ])

        with patch.object(LLMModule, "chat") as mock_chat:
            mock_chat.side_effect = [
                make_chat_response(plan_json),
                make_chat_response("s1 完成"),
                make_chat_response("s2 完成"),
                make_chat_response("s3 完成"),
            ]

            agent = make_agent()
            result = agent.run("测试依赖")

            self.assertTrue(result.success)
            # 验证步骤按 ID 顺序执行
            step_order = [r.step_id for r in result.plan.results]
            self.assertEqual(step_order, ["s1", "s2", "s3"])

    def test_token_accumulation(self):
        """验证 token 用量累加"""
        plan_json = make_plan_json(goal="t", steps=[
            {"id": "s1", "description": "d", "instruction": "i",
             "action": "respond", "depends_on": []}
        ])

        with patch.object(LLMModule, "chat") as mock_chat:
            mock_chat.side_effect = [
                make_chat_response(plan_json),
                make_chat_response("done"),
            ]

            agent = make_agent()
            result = agent.run("test")

            # 每次调用 80 tokens total，2 次调用 = 160
            self.assertEqual(result.total_tokens["total_tokens"], 160)


class TestPlanRevision(unittest.TestCase):
    """重规划测试"""

    def test_step_failure_triggers_revision(self):
        """步骤失败后触发重规划"""
        plan_json = make_plan_json(goal="容错任务", steps=[
            {"id": "s1", "description": "风险步骤", "instruction": "有可能失败的步骤。",
             "action": "tool_call", "depends_on": []},
            {"id": "s2", "description": "后续步骤", "instruction": "依赖 s1 成功才能执行。",
             "action": "respond", "depends_on": ["s1"]},
        ])

        # 重规划返回的新步骤 JSON
        revised_json = json.dumps({
            "rationale": "s1 失败，换一种方式重试",
            "revised_steps": [
                {"id": "s1_v2", "description": "改用备用方案",
                 "instruction": "用另一种方式完成任务。",
                 "action": "respond", "depends_on": []},
            ]
        }, ensure_ascii=False)

        with patch.object(LLMModule, "chat") as mock_chat:
            mock_chat.side_effect = [
                make_chat_response(plan_json),              # plan gen
                make_chat_response("s1 失败", "error"),      # s1 执行失败
                make_chat_response(revised_json),            # 重规划生成新步骤
                make_chat_response("备用方案成功"),           # s1_v2 执行成功
            ]

            agent = make_agent()
            # max_retries=0 → 第一次失败不重试，直接触发重规划
            agent._planning = PlanningModule(
                max_retries_per_step=0,
                max_revisions=3,
                skip_failed_non_critical=False,
            )
            result = agent.run("容错测试")

            self.assertTrue(result.success)
            self.assertEqual(result.plan.revision_count, 1)
            self.assertEqual(mock_chat.call_count, 4)

    def test_max_revisions_exceeded(self):
        """超过最大重规划次数后终止（skip_failed_non_critical=False）"""
        plan_json = make_plan_json(goal="反复失败任务", steps=[
            {"id": "s1", "description": "总是失败", "instruction": "总是失败。",
             "action": "tool_call", "depends_on": []},
            {"id": "s2", "description": "后续", "instruction": "后续。",
             "action": "respond", "depends_on": ["s1"]},
        ])

        # 重规划返回：只返回一个不依赖失败步骤的新步骤（替换 s1 和 s2）
        revised_alt = json.dumps({
            "rationale": "改用独立步骤",
            "revised_steps": [
                {"id": "s_alt", "description": "备用方案", "instruction": "用备选方式完成目标。",
                 "action": "respond", "depends_on": []},
            ]
        }, ensure_ascii=False)

        with patch.object(LLMModule, "chat") as mock_chat:
            mock_chat.side_effect = [
                make_chat_response(plan_json),              # plan gen
                make_chat_response("s1 失败", "error"),      # s1 失败（触发重规划 #1）
                make_chat_response(revised_alt),             # 重规划 #1: 生成 s_alt
                make_chat_response("s_alt 成功"),            # s_alt 执行成功
            ]

            agent = make_agent()
            agent._planning = PlanningModule(
                max_retries_per_step=0,
                max_revisions=1,  # 最多 1 次重规划
                skip_failed_non_critical=False,
            )
            result = agent.run("反复失败")

            # 重规划后 s_alt 成功 → 计划成功完成
            self.assertTrue(result.success)
            self.assertEqual(result.plan.revision_count, 1)
            self.assertEqual(mock_chat.call_count, 4)

    def test_max_revisions_exhausted_plan_fails(self):
        """重规划次数用尽后步骤仍失败 → 最终失败"""
        plan_json = make_plan_json(goal="耗尽重规划", steps=[
            {"id": "s1", "description": "致命步骤", "instruction": "必然失败。",
             "action": "tool_call", "depends_on": []},
            {"id": "s2", "description": "依赖步骤", "instruction": "依赖于 s1。",
             "action": "respond", "depends_on": ["s1"]},
        ])

        # 重规划也返回会失败的步骤
        revised_bad = json.dumps({
            "rationale": "重试",
            "revised_steps": [
                {"id": "s_alt", "description": "重试仍然失败", "instruction": "仍然会失败。",
                 "action": "tool_call", "depends_on": []},
            ]
        }, ensure_ascii=False)

        with patch.object(LLMModule, "chat") as mock_chat:
            mock_chat.side_effect = [
                make_chat_response(plan_json),              # plan gen
                make_chat_response("s1 失败", "error"),      # s1 失败（重规划 #1）
                make_chat_response(revised_bad),             # 重规划 prompt
                make_chat_response("s_alt 也失败", "error"), # s_alt 也失败
            ]

            agent = make_agent()
            agent._planning = PlanningModule(
                max_retries_per_step=0,
                max_revisions=1,
                skip_failed_non_critical=False,
            )
            result = agent.run("测试")

            # 重规划次数用尽 + 步骤失败 → 失败
            self.assertFalse(result.success)
            self.assertEqual(result.plan.revision_count, 1)
            self.assertEqual(mock_chat.call_count, 4)

    def test_skip_failed_non_critical(self):
        """非关键步骤失败后，依赖它的步骤自动跳过"""
        plan_json = make_plan_json(goal="可跳过任务", steps=[
            {"id": "s1", "description": "可能失败的非关键步骤", "instruction": "试一试。",
             "action": "tool_call", "depends_on": []},
            {"id": "s2", "description": "依赖 s1 的步骤", "instruction": "依赖 s1。",
             "action": "respond", "depends_on": ["s1"]},
        ])

        with patch.object(LLMModule, "chat") as mock_chat:
            mock_chat.side_effect = [
                make_chat_response(plan_json),
                make_chat_response("s1 失败", "error"),
                make_chat_response("extra1"),  # 安全余量
                make_chat_response("extra2"),
            ]

            agent = make_agent()
            agent._planning = PlanningModule(
                max_retries_per_step=0,
                skip_failed_non_critical=True,
            )
            result = agent.run("测试跳过")

            # s2 被 auto-skip，不调用 LLM → 仅 2 次 chat（plan + s1）
            self.assertEqual(mock_chat.call_count, 2)


class TestMemoryIntegration(unittest.TestCase):
    """记忆模块集成测试"""

    def test_remember_called_after_execution(self):
        """验证执行完成后 remember 被调用"""
        plan_json = make_plan_json(goal="记忆测试", steps=[
            {"id": "s1", "description": "d", "instruction": "i",
             "action": "respond", "depends_on": []}
        ])

        with patch.object(LLMModule, "chat") as mock_chat:
            mock_chat.side_effect = [
                make_chat_response(plan_json),
                make_chat_response("回复内容"),
            ]

            agent = make_agent()

            # 替换 memory 为 mock
            mock_memory = MagicMock()
            mock_memory.retrieve.return_value = "context"
            agent._memory = mock_memory

            agent.run("测试")

            mock_memory.remember.assert_called_once()
            # 验证 remember 收到了用户和助手消息
            call_args = mock_memory.remember.call_args[0][0]
            self.assertEqual(len(call_args), 2)
            self.assertEqual(call_args[0]["role"], "user")
            self.assertEqual(call_args[1]["role"], "assistant")

    def test_retrieve_called_before_planning(self):
        """验证在计划生成前调用记忆检索"""
        plan_json = make_plan_json(goal="检索测试", steps=[
            {"id": "s1", "description": "d", "instruction": "i",
             "action": "respond", "depends_on": []}
        ])

        with patch.object(LLMModule, "chat") as mock_chat:
            mock_chat.side_effect = [
                make_chat_response(plan_json),
                make_chat_response("回复"),
            ]

            agent = make_agent()

            mock_memory = MagicMock()
            mock_memory.retrieve.return_value = "检索到的上下文"
            agent._memory = mock_memory

            agent.run("用户输入")

            # 验证 retrieve 被调用
            mock_memory.retrieve.assert_called_once()
            # 验证上下文被传入了 plan prompt
            plan_call = mock_chat.call_args_list[0]
            prompt = plan_call[1]["messages"][0]["content"]
            self.assertIn("检索到的上下文", prompt)

    def test_remember_not_called_when_perception_rejects(self):
        """感知拦截时不应写入记忆"""
        import tempfile, os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("违禁词\n")
            tmp_path = f.name

        try:
            agent = make_agent()
            mock_memory = MagicMock()
            agent._memory = mock_memory
            agent._perception = PerceptionModule(
                max_length=4000, sensitive_words_path=tmp_path
            )

            agent.run("包含违禁词")

            mock_memory.remember.assert_not_called()
        finally:
            os.unlink(tmp_path)


class TestErrorHandling(unittest.TestCase):
    """错误处理测试"""

    def test_plan_validation_failure(self):
        """计划验证失败（如循环依赖）"""
        # 包含循环依赖的计划
        bad_plan = json.dumps({
            "goal": "循环依赖计划",
            "steps": [
                {"id": "a", "description": "A", "instruction": "A",
                 "action": "think", "depends_on": ["b"]},
                {"id": "b", "description": "B", "instruction": "B",
                 "action": "think", "depends_on": ["a"]},
            ]
        }, ensure_ascii=False)

        with patch.object(LLMModule, "chat") as mock_chat:
            mock_chat.side_effect = [
                make_chat_response(bad_plan),
            ]

            agent = make_agent()
            result = agent.run("测试循环")

            self.assertFalse(result.success)
            self.assertIsNotNone(result.error)
            self.assertIn("循环", result.content)

    def test_step_execution_continues_after_non_fatal_error(self):
        """非致命错误后继续执行（并行步骤间互不影响）"""
        plan_json = make_plan_json(goal="容错任务", steps=[
            {"id": "s1", "description": "可能失败", "instruction": "尝试执行。",
             "action": "tool_call", "depends_on": []},
            {"id": "s2", "description": "独立步骤", "instruction": "独立于 s1。",
             "action": "respond", "depends_on": []},
        ])

        with patch.object(LLMModule, "chat") as mock_chat:
            mock_chat.side_effect = [
                make_chat_response(plan_json),
                make_chat_response("s1 失败", "error"),
                make_chat_response("s2 成功完成"),
            ]

            agent = make_agent()
            agent._planning = PlanningModule(
                max_retries_per_step=0,
                skip_failed_non_critical=True,
            )
            result = agent.run("容错测试")

            # s2 无依赖 s1，所以即使 s1 失败，s2 仍执行成功
            self.assertTrue(result.success)
            self.assertEqual(mock_chat.call_count, 3)

    def test_revision_parse_failure(self):
        """重规划 JSON 解析失败不应崩溃"""
        plan_json = make_plan_json(goal="重规划解析失败", steps=[
            {"id": "s1", "description": "会失败", "instruction": "一定会失败。",
             "action": "tool_call", "depends_on": []},
            {"id": "s2", "description": "依赖 s1", "instruction": "依赖于 s1。",
             "action": "respond", "depends_on": ["s1"]},
        ])

        with patch.object(LLMModule, "chat") as mock_chat:
            mock_chat.side_effect = [
                make_chat_response(plan_json),
                make_chat_response("s1 失败", "error"),
                make_chat_response("无效 JSON（重规划失败）"),  # 重规划返回无效 JSON
            ]

            agent = make_agent()
            agent._planning = PlanningModule(
                max_retries_per_step=0,
                max_revisions=3,
                skip_failed_non_critical=False,
            )
            # 不应抛出异常
            result = agent.run("测试")

            # 重规划失败后终止执行
            self.assertIsNotNone(result)

    def test_empty_steps_plan(self):
        """空步骤计划应被验证拦截"""
        bad_plan = json.dumps({
            "goal": "空计划",
            "steps": []
        }, ensure_ascii=False)

        with patch.object(LLMModule, "chat") as mock_chat:
            mock_chat.side_effect = [
                make_chat_response(bad_plan),
            ]

            agent = make_agent()
            result = agent.run("测试空计划")

            self.assertFalse(result.success)
            self.assertIsNotNone(result.error)


class TestFullPipeline(unittest.TestCase):
    """完整端到端测试"""

    def test_full_pipeline_with_tools(self):
        """完整流水线：感知 → 记忆 → 规划 → 工具调用 → 响应"""
        plan_json = json.dumps({
            "goal": "在互联网上搜索 Python 和 Rust 的性能对比，生成一份对比报告",
            "steps": [
                {"id": "s1", "description": "搜索 Python 性能数据",
                 "instruction": "使用 web_search 搜索 Python web 框架性能",
                 "action": "tool_call", "depends_on": []},
                {"id": "s2", "description": "搜索 Rust 性能数据",
                 "instruction": "使用 web_search 搜索 Rust web 框架性能",
                 "action": "tool_call", "depends_on": []},
                {"id": "s3", "description": "对比分析并生成报告",
                 "instruction": "根据前两步的搜索结果，对比 Python 和 Rust 在 Web 开发中的性能差异。",
                 "action": "respond", "depends_on": ["s1", "s2"]},
            ]
        }, ensure_ascii=False)

        with patch.object(LLMModule, "chat") as mock_chat:
            mock_chat.side_effect = [
                # Plan generation
                make_chat_response(plan_json),
                # s1: search Python
                make_chat_response("Python 搜索完成：FastAPI 约 10k req/s",
                    tool_calls_log=[{"round": 1, "name": "web_search",
                                     "arguments": {"query": "Python web 性能"},
                                     "result": "FastAPI 约 10k req/s"}]),
                # s2: search Rust
                make_chat_response("Rust 搜索完成：Actix-web 约 100k req/s",
                    tool_calls_log=[{"round": 1, "name": "web_search",
                                     "arguments": {"query": "Rust web 性能"},
                                     "result": "Actix-web 约 100k req/s"}]),
                # s3: analyze and respond
                make_chat_response(
                    "## Python vs Rust 性能对比\n\n"
                    "| 维度 | Python | Rust |\n"
                    "|------|--------|------|\n"
                    "| 请求吞吐 | ~10k/s | ~100k/s |\n"
                    "| 内存占用 | 较高 | 较低 |\n\n"
                    "**总结**: Rust 在 Web 性能方面显著优于 Python。"
                ),
            ]

            agent = make_agent()
            result = agent.run("帮我对比 Python 和 Rust 的性能")

            self.assertTrue(result.success)
            self.assertEqual(result.total_tool_calls, 2)
            self.assertIsNotNone(result.plan)
            self.assertEqual(len(result.plan.get_completed_steps()), 3)
            self.assertIn("Python", result.content)
            self.assertIn("Rust", result.content)

    def test_result_contains_progress_info(self):
        """验证结果包含进度信息"""
        plan_json = make_plan_json(goal="进度测试", steps=[
            {"id": "s1", "description": "单步任务", "instruction": "完成它。",
             "action": "respond", "depends_on": []}
        ])

        with patch.object(LLMModule, "chat") as mock_chat:
            mock_chat.side_effect = [
                make_chat_response(plan_json),
                make_chat_response("任务完成！"),
            ]

            agent = make_agent()
            result = agent.run("做任务")

            # 验证结果包含步骤状态标记
            self.assertIn("✅", result.content)
            self.assertIn("单步任务", result.content)


# ── 运行 ────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
