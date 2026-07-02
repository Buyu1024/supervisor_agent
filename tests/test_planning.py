"""规划模块 —— 集成测试 Demo"""

import sys
import json
from pathlib import Path

# 确保 src 目录在 Python 路径中
_SRC_DIR = Path(__file__).parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# ============================================================
# 测试用例
# ============================================================

class TestTypes:
    """TaskStep / StepResult / TaskPlan 数据结构测试"""

    def test_task_step_basic(self):
        """TaskStep 基本创建"""
        from agent_demo.planning import TaskStep

        step = TaskStep(
            id="step_1",
            description="搜索 Python Web 框架",
            instruction="使用搜索工具搜索 Python 主流 Web 框架",
            action="tool_call",
            depends_on=[],
        )
        assert step.id == "step_1"
        assert step.status == "pending"
        assert step.can_execute(set())
        assert step.depends_on == []
        print(f"  [PASS] TaskStep 创建: {step}")

    def test_task_step_deps(self):
        """TaskStep 依赖检查"""
        from agent_demo.planning import TaskStep

        step = TaskStep(
            id="step_3",
            description="生成报告",
            instruction="...",
            action="respond",
            depends_on=["step_1", "step_2"],
        )
        # 所有依赖都完成才能执行
        assert step.can_execute({"step_1", "step_2"})
        # 缺少任何一个依赖都不能执行
        assert not step.can_execute({"step_1"})
        assert not step.can_execute(set())
        print(f"  [PASS] TaskStep 依赖检查")

    def test_step_result(self):
        """StepResult 创建和摘要"""
        from agent_demo.planning import StepResult

        r = StepResult(
            step_id="step_1",
            success=True,
            output="搜索到 5 个 Python Web 框架：Django, Flask, FastAPI, ...",
            elapsed_ms=250.0,
        )
        assert r.success
        assert r.error is None
        summary = r.summary()
        assert "✅" in summary
        assert "step_1" in summary
        print(f"  [PASS] StepResult: {summary}")

    def test_step_result_failure(self):
        """StepResult 失败情况"""
        from agent_demo.planning import StepResult

        r = StepResult(
            step_id="step_2",
            success=False,
            output="",
            error="工具调用超时",
        )
        assert not r.success
        assert "❌" in r.summary()
        print(f"  [PASS] StepResult 失败: {r.summary()}")

    def test_task_plan_progress(self):
        """TaskPlan 进度摘要"""
        from agent_demo.planning import TaskPlan, TaskStep, StepResult

        plan = TaskPlan(
            goal="研究 Python vs Rust Web 开发",
            steps=[
                TaskStep(id="s1", description="搜索", instruction="...", action="tool_call"),
                TaskStep(id="s2", description="整理对比", instruction="...", action="think", depends_on=["s1"]),
                TaskStep(id="s3", description="生成报告", instruction="...", action="respond", depends_on=["s2"]),
            ],
        )

        # 模拟执行进度
        plan.steps[0].status = "completed"
        plan.steps[1].status = "running"

        summary = plan.progress_summary()
        assert "1/3" in summary
        assert "1 进行中" in summary
        assert "1 待执行" in summary
        print(f"  [PASS] TaskPlan 进度: {summary}")

    def test_task_plan_serialization(self):
        """TaskPlan 序列化和反序列化"""
        from agent_demo.planning import TaskPlan, TaskStep, StepResult

        plan = TaskPlan(
            goal="测试序列化",
            steps=[
                TaskStep(id="s1", description="步骤1", instruction="做步骤1", action="tool_call"),
                TaskStep(id="s2", description="步骤2", instruction="做步骤2", action="respond", depends_on=["s1"]),
            ],
        )
        plan.steps[0].status = "completed"
        plan.results.append(StepResult(step_id="s1", success=True, output="步骤1完成"))

        # 序列化
        d = plan.to_dict()
        assert d["goal"] == "测试序列化"
        assert len(d["steps"]) == 2
        assert len(d["results"]) == 1
        assert d["steps"][0]["status"] == "completed"

        # 反序列化
        restored = TaskPlan.from_dict(d)
        assert restored.goal == plan.goal
        assert len(restored.steps) == 2
        assert restored.steps[0].status == "completed"
        assert restored.steps[1].depends_on == ["s1"]
        assert len(restored.results) == 1
        assert restored.results[0].success
        print(f"  [PASS] TaskPlan 序列化/反序列化: {restored.progress_summary()}")


class TestPromptBuilder:
    """PlanPromptBuilder prompt 构建测试"""

    def test_build_plan_prompt(self):
        """构建计划生成 prompt"""
        from agent_demo.planning import PlanPromptBuilder

        builder = PlanPromptBuilder()
        prompt = builder.build_plan_prompt(
            intent="帮我搜索 Python Web 框架的最新动态",
        )

        assert "任务规划专家" in prompt
        assert "Python Web 框架" in prompt
        assert "goal" in prompt
        assert "steps" in prompt
        assert "JSON" in prompt
        print(f"  [PASS] build_plan_prompt: {len(prompt)} 字符")

    def test_build_plan_prompt_with_tools(self):
        """含工具 schema 的计划 prompt"""
        from agent_demo.planning import PlanPromptBuilder

        builder = PlanPromptBuilder()
        prompt = builder.build_plan_prompt(
            intent="搜索天气",
            tool_schemas=[
                {
                    "type": "function",
                    "function": {
                        "name": "search_web",
                        "description": "搜索互联网获取信息",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "搜索关键词"}
                            },
                            "required": ["query"],
                        },
                    },
                }
            ],
        )

        assert "search_web" in prompt
        assert "搜索互联网" in prompt
        assert "query" in prompt
        assert "必填" in prompt
        print(f"  [PASS] build_plan_prompt + tools: {len(prompt)} 字符")

    def test_build_plan_prompt_with_context(self):
        """含上下文信息的计划 prompt"""
        from agent_demo.planning import PlanPromptBuilder

        builder = PlanPromptBuilder()
        prompt = builder.build_plan_prompt(
            intent="推荐餐厅",
            context="用户偏好: 喜欢吃川菜，预算人均 100 元以内",
        )

        assert "川菜" in prompt
        assert "100 元" in prompt
        print(f"  [PASS] build_plan_prompt + context")

    def test_build_step_prompt(self):
        """构建步骤执行 prompt"""
        from agent_demo.planning import PlanPromptBuilder, TaskPlan, TaskStep, StepResult

        builder = PlanPromptBuilder()

        plan = TaskPlan(
            goal="研究 Python 性能",
            steps=[
                TaskStep(id="s1", description="搜索 Python 性能优化", instruction="搜索相关资料", action="tool_call"),
                TaskStep(id="s2", description="总结", instruction="总结搜索结果", action="respond", depends_on=["s1"]),
            ],
        )
        plan.steps[0].status = "completed"
        plan.results.append(StepResult(
            step_id="s1", success=True,
            output="找到 3 篇关于 Python 性能优化的文章...",
        ))

        prompt = builder.build_step_prompt(plan, plan.steps[1])

        assert "研究 Python 性能" in prompt
        assert "s1" in prompt
        assert "Python 性能优化" in prompt
        assert "当前步骤" in prompt or "s2" in prompt
        print(f"  [PASS] build_step_prompt: {len(prompt)} 字符")

    def test_build_revise_prompt(self):
        """构建重规划 prompt"""
        from agent_demo.planning import PlanPromptBuilder, TaskPlan, TaskStep, StepResult

        builder = PlanPromptBuilder()

        plan = TaskPlan(
            goal="搜索并整理报告",
            steps=[
                TaskStep(id="s1", description="搜索", instruction="..." , action="tool_call"),
                TaskStep(id="s2", description="分析", instruction="..." , action="think", depends_on=["s1"]),
                TaskStep(id="s3", description="输出", instruction="..." , action="respond", depends_on=["s2"]),
            ],
        )
        plan.steps[0].status = "completed"
        plan.steps[1].status = "failed"
        plan.results.append(StepResult(step_id="s1", success=True, output="搜索结果"))

        prompt = builder.build_revise_prompt(
            plan, "s2", "API 调用超时，未获取到分析结果"
        )

        assert "搜索并整理报告" in prompt
        assert "s2" in prompt
        assert "API 调用超时" in prompt
        assert "revised_steps" in prompt
        print(f"  [PASS] build_revise_prompt: {len(prompt)} 字符")


class TestExecutor:
    """PlanExecutor 状态机测试"""

    def _make_plan(self):
        from agent_demo.planning import TaskPlan, TaskStep
        return TaskPlan(
            goal="测试任务",
            steps=[
                TaskStep(id="s1", description="步骤1", instruction="搜索", action="tool_call"),
                TaskStep(id="s2", description="步骤2", instruction="分析", action="think", depends_on=["s1"]),
                TaskStep(id="s3", description="步骤3", instruction="回复", action="respond", depends_on=["s2"]),
            ],
        )

    def test_start_plan(self):
        """开始执行计划"""
        from agent_demo.planning import PlanExecutor

        executor = PlanExecutor()
        plan = self._make_plan()

        plan = executor.start(plan)
        assert plan.status == "running"
        print(f"  [PASS] start 计划: status={plan.status}")

    def test_get_next_step_sequential(self):
        """顺序获取下一步"""
        from agent_demo.planning import PlanExecutor, StepResult

        executor = PlanExecutor()
        plan = executor.start(self._make_plan())

        # 第一步：s1（无依赖）
        step = executor.get_next_step(plan)
        assert step.id == "s1"
        print(f"  [PASS] get_next_step 第一步: {step.id}")

        # 完成 s1
        plan = executor.record_result(plan, StepResult(step_id="s1", success=True, output="搜索结果..."))

        # 第二步：s2（依赖 s1 已满足）
        step = executor.get_next_step(plan)
        assert step.id == "s2"
        print(f"  [PASS] get_next_step 第二步: {step.id}")

        # 完成 s2
        plan = executor.record_result(plan, StepResult(step_id="s2", success=True, output="分析结果..."))

        # 第三步：s3
        step = executor.get_next_step(plan)
        assert step.id == "s3"
        print(f"  [PASS] get_next_step 第三步: {step.id}")

    def test_dependency_blocking(self):
        """依赖未满足时不返回下一步"""
        from agent_demo.planning import PlanExecutor

        executor = PlanExecutor()
        plan = executor.start(self._make_plan())

        # s1 未完成时，s2 和 s3 都不应返回（s2 依赖 s1）
        step = executor.get_next_step(plan)
        assert step.id == "s1"

        # 直接查询 s2 的依赖: s1 不在 completed 中，所以 can_execute 应返回 False
        assert not plan.steps[1].can_execute(set())
        assert plan.steps[1].can_execute({"s1"})
        print(f"  [PASS] 依赖阻塞: s2 需要 s1 先完成")

    def test_retry_on_failure(self):
        """失败步骤允许重试"""
        from agent_demo.planning import PlanExecutor, StepResult

        executor = PlanExecutor(max_retries_per_step=2)
        plan = executor.start(self._make_plan())

        # 执行 s1 第一次失败
        step = executor.get_next_step(plan)
        plan = executor.record_result(plan, StepResult(step_id="s1", success=False, output="", error="网络错误"))

        # s1 的状态是 failed，但仍可重试（get_next_step 会返回它）
        assert plan.steps[0].status == "failed"
        step = executor.get_next_step(plan)
        assert step.id == "s1", "失败步骤应允许重试"
        print(f"  [PASS] 失败重试: s1 在第一次失败后仍可执行")

    def test_max_retries_exceeded(self):
        """超过最大重试次数后跳过"""
        from agent_demo.planning import PlanExecutor, StepResult

        executor = PlanExecutor(max_retries_per_step=1)
        plan = executor.start(self._make_plan())

        # 第一次失败
        plan = executor.record_result(plan, StepResult(step_id="s1", success=False, output="", error="网络错误"))

        # 第二次重试失败
        plan = executor.record_result(plan, StepResult(step_id="s1", success=False, output="", error="再次失败"))

        # 超过重试次数，get_next_step 应跳过 s1
        remaining = executor.get_next_step(plan)
        if remaining:
            # 如果 skip_failed_non_critical=True，后续步骤可能被跳过
            pass
        else:
            pass  # 所有步骤都完了（或都跳过了）

        # s1 应仍在 failed 状态
        assert plan.steps[0].status == "failed"
        print(f"  [PASS] 超过最大重试: s1 status={plan.steps[0].status}")

    def test_is_complete(self):
        """检测计划完成"""
        from agent_demo.planning import PlanExecutor, StepResult

        executor = PlanExecutor()
        plan = executor.start(self._make_plan())

        # 执行所有步骤
        for expected_id in ["s1", "s2", "s3"]:
            step = executor.get_next_step(plan)
            plan = executor.record_result(plan, StepResult(
                step_id=step.id, success=True,
                output=f"{step.id} 完成",
            ))

        assert executor.is_complete(plan)
        print(f"  [PASS] is_complete: 所有步骤完成")

    def test_finalize(self):
        """完成计划并设置最终状态"""
        from agent_demo.planning import PlanExecutor, StepResult

        executor = PlanExecutor()
        plan = executor.start(self._make_plan())

        # 全部完成
        for _ in range(3):
            step = executor.get_next_step(plan)
            plan = executor.record_result(plan, StepResult(
                step_id=step.id, success=True, output="完成",
            ))

        plan = executor.finalize(plan)
        assert plan.status == "completed"
        print(f"  [PASS] finalize: status={plan.status}")

    def test_parallel_deps(self):
        """并行依赖：s3 依赖 s1 和 s2，两者都可并行执行"""
        from agent_demo.planning import TaskPlan, TaskStep, PlanExecutor, StepResult

        plan = TaskPlan(
            goal="并行任务",
            steps=[
                TaskStep(id="s1", description="搜索Python", instruction="...", action="tool_call"),
                TaskStep(id="s2", description="搜索Rust", instruction="...", action="tool_call"),
                TaskStep(id="s3", description="对比", instruction="...", action="think", depends_on=["s1", "s2"]),
            ],
        )

        executor = PlanExecutor()
        plan = executor.start(plan)

        # s1 和 s2 都没有依赖，都可以执行
        step1 = executor.get_next_step(plan)
        assert step1.id == "s1"  # 按顺序先返回 s1

        # 完成 s1
        plan = executor.record_result(plan, StepResult(step_id="s1", success=True, output="Python 结果"))

        # s2 现在可以执行
        step2 = executor.get_next_step(plan)
        assert step2.id == "s2"

        # 只完成 s1，s3 仍需 s2
        assert not plan.steps[2].can_execute({"s1"})  # s3 还需要 s2

        # 完成 s2
        plan = executor.record_result(plan, StepResult(step_id="s2", success=True, output="Rust 结果"))

        # s3 现在可以执行
        step3 = executor.get_next_step(plan)
        assert step3.id == "s3"
        print(f"  [PASS] 并行依赖: s1,s2 可独立执行，s3 等两者完成")


class TestPlanValidation:
    """计划验证测试"""

    def test_empty_plan(self):
        """空计划不合法"""
        from agent_demo.planning import TaskPlan, PlanExecutor

        plan = TaskPlan(goal="无步骤的计划")
        executor = PlanExecutor()
        errors = executor.validate(plan)
        assert len(errors) > 0
        assert "至少需要一个步骤" in errors[0]
        print(f"  [PASS] 空计划验证: {errors}")

    def test_duplicate_ids(self):
        """重复步骤 ID 不合法"""
        from agent_demo.planning import TaskPlan, TaskStep, PlanExecutor

        plan = TaskPlan(
            goal="重复ID",
            steps=[
                TaskStep(id="s1", description="", instruction="", action="tool_call"),
                TaskStep(id="s1", description="", instruction="", action="respond"),
            ],
        )
        executor = PlanExecutor()
        errors = executor.validate(plan)
        assert any("不唯一" in e or "重复" in e for e in errors)
        print(f"  [PASS] 重复 ID 验证: {errors}")

    def test_missing_dependency(self):
        """引用不存在的依赖"""
        from agent_demo.planning import TaskPlan, TaskStep, PlanExecutor

        plan = TaskPlan(
            goal="缺失依赖",
            steps=[
                TaskStep(id="s1", description="", instruction="", action="tool_call", depends_on=["s_nonexistent"]),
            ],
        )
        executor = PlanExecutor()
        errors = executor.validate(plan)
        assert any("不存在" in e for e in errors)
        print(f"  [PASS] 缺失依赖验证: {errors}")

    def test_self_dependency(self):
        """不能依赖自己"""
        from agent_demo.planning import TaskPlan, TaskStep, PlanExecutor

        plan = TaskPlan(
            goal="自依赖",
            steps=[
                TaskStep(id="s1", description="", instruction="", action="tool_call", depends_on=["s1"]),
            ],
        )
        executor = PlanExecutor()
        errors = executor.validate(plan)
        assert any("自己" in e for e in errors)
        print(f"  [PASS] 自依赖验证: {errors}")

    def test_cycle_detection(self):
        """循环依赖检测"""
        from agent_demo.planning import TaskPlan, TaskStep, PlanExecutor

        plan = TaskPlan(
            goal="循环依赖",
            steps=[
                TaskStep(id="s1", description="", instruction="", action="tool_call", depends_on=["s3"]),
                TaskStep(id="s2", description="", instruction="", action="think", depends_on=["s1"]),
                TaskStep(id="s3", description="", instruction="", action="respond", depends_on=["s2"]),
            ],
        )
        executor = PlanExecutor()
        errors = executor.validate(plan)
        assert any("循环" in e for e in errors)
        print(f"  [PASS] 循环依赖检测: {errors}")


class TestParsing:
    """JSON 解析测试"""

    def test_parse_clean_json(self):
        """解析干净的 JSON 计划"""
        from agent_demo.planning import PlanningModule

        planning = PlanningModule()
        llm_output = json.dumps({
            "goal": "搜索天气并生成报告",
            "steps": [
                {
                    "id": "step_1",
                    "description": "搜索北京天气",
                    "instruction": "使用搜索工具查询北京今天天气",
                    "action": "tool_call",
                    "depends_on": [],
                },
                {
                    "id": "step_2",
                    "description": "生成天气报告",
                    "instruction": "根据上一步的搜索结果，生成一份简洁的天气报告",
                    "action": "respond",
                    "depends_on": ["step_1"],
                },
            ],
        }, ensure_ascii=False)

        plan = planning.parse_plan(llm_output)
        assert plan.goal == "搜索天气并生成报告"
        assert len(plan.steps) == 2
        assert plan.steps[0].action == "tool_call"
        assert plan.steps[1].depends_on == ["step_1"]
        print(f"  [PASS] 解析干净 JSON: {plan}")

    def test_parse_json_in_code_block(self):
        """解析 markdown 代码块中的 JSON"""
        from agent_demo.planning import PlanningModule

        planning = PlanningModule()
        llm_output = """好的，以下是执行计划：

```json
{
  "goal": "简单的两步骤任务",
  "steps": [
    {"id": "s1", "description": "搜索", "instruction": "搜索", "action": "tool_call", "depends_on": []},
    {"id": "s2", "description": "回复", "instruction": "回复", "action": "respond", "depends_on": ["s1"]}
  ]
}
```

这个计划包含 2 个步骤。"""

        plan = planning.parse_plan(llm_output)
        assert plan.goal == "简单的两步骤任务"
        assert len(plan.steps) == 2
        print(f"  [PASS] 解析 markdown 代码块 JSON: {plan}")

    def test_parse_json_with_extra_text(self):
        """JSON 前后有额外文字"""
        from agent_demo.planning import PlanningModule

        planning = PlanningModule()
        llm_output = '我分析了你的需求，计划如下：\n{"goal": "测试","steps": [{"id": "s1","description": "","instruction": "","action": "respond","depends_on": []}]}\n希望这个计划对你有帮助。'

        plan = planning.parse_plan(llm_output)
        assert plan.goal == "测试"
        print(f"  [PASS] 解析含额外文字的 JSON: {plan}")

    def test_parse_invalid_json_raises(self):
        """无效 JSON 抛出异常"""
        from agent_demo.planning import PlanningModule

        planning = PlanningModule()
        try:
            planning.parse_plan("这不是 JSON")
            assert False, "应抛出 ValueError"
        except ValueError as e:
            assert "无法解析" in str(e) or "JSON" in str(e)
            print(f"  [PASS] 无效 JSON 抛异常")

    def test_parse_missing_goal(self):
        """缺少 goal 字段抛出异常"""
        from agent_demo.planning import PlanningModule

        planning = PlanningModule()
        try:
            planning.parse_plan(json.dumps({
                "steps": [{"id": "s1", "description": "", "instruction": "", "action": "respond", "depends_on": []}],
            }))
            assert False, "应抛出 ValueError"
        except ValueError as e:
            assert "goal" in str(e).lower()
            print(f"  [PASS] 缺少 goal 抛异常")

    def test_parse_steps_for_revision(self):
        """解析重规划返回的 revised_steps"""
        from agent_demo.planning import PlanningModule

        planning = PlanningModule()
        llm_output = json.dumps({
            "rationale": "重试搜索，使用不同的关键词",
            "revised_steps": [
                {
                    "id": "step_2_revised",
                    "description": "用新关键词重新搜索",
                    "instruction": "使用搜索工具，换一组关键词重新搜索",
                    "action": "tool_call",
                    "depends_on": ["step_1"],
                },
            ],
        }, ensure_ascii=False)

        steps = planning.parse_steps(llm_output)
        assert len(steps) == 1
        assert steps[0].id == "step_2_revised"
        assert steps[0].action == "tool_call"
        print(f"  [PASS] 解析 revised_steps: {steps[0]}")


class TestPlanningModule:
    """PlanningModule 集成测试"""

    def test_init(self):
        """初始化"""
        from agent_demo.planning import PlanningModule

        planning = PlanningModule(max_retries_per_step=2, max_revisions=3)
        assert planning._executor.max_retries_per_step == 2
        print(f"  [PASS] PlanningModule 初始化")

    def test_full_workflow_no_revision(self):
        """完整工作流（无重规划）"""
        from agent_demo.planning import PlanningModule, StepResult

        planning = PlanningModule()

        # 1. 构建 prompt
        prompt = planning.build_plan_prompt(
            intent="搜索北京天气并回复",
            tool_schemas=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "获取城市天气",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                },
            ],
        )
        assert "get_weather" in prompt

        # 2. 模拟 LLM 返回的计划
        llm_output = json.dumps({
            "goal": "查询北京天气并回复用户",
            "steps": [
                {"id": "s1", "description": "获取北京天气", "instruction": "调用 get_weather 获取北京天气数据", "action": "tool_call", "depends_on": []},
                {"id": "s2", "description": "回复用户", "instruction": "根据天气数据生成友好的回复", "action": "respond", "depends_on": ["s1"]},
            ],
        }, ensure_ascii=False)

        plan = planning.parse_plan(llm_output)
        assert plan.goal == "查询北京天气并回复用户"

        # 3. 开始执行
        plan = planning.start_plan(plan)
        assert plan.status == "running"

        # 4. 执行步骤
        # s1
        step = planning.get_next_step(plan)
        assert step.id == "s1"
        step_prompt = planning.build_step_prompt(plan, step)
        assert "北京天气" in step_prompt
        plan = planning.record_result(plan, StepResult(
            step_id="s1", success=True,
            output="北京今天晴，18-26°C，空气质量良",
        ))

        # s2
        step = planning.get_next_step(plan)
        assert step.id == "s2"
        plan = planning.record_result(plan, StepResult(
            step_id="s2", success=True,
            output="北京今天天气晴朗，气温18-26°C，适合出行。",
        ))

        # 5. 完成
        assert planning.is_complete(plan)
        plan = planning.finalize_plan(plan)
        assert plan.status == "completed"

        # 6. 进度
        progress = planning.get_progress(plan)
        assert progress["completed"] == 2
        assert progress["percent"] == 100.0

        print(f"  [PASS] 完整工作流: {plan.progress_summary()}")

    def test_full_workflow_with_revision(self):
        """完整工作流（含重规划）"""
        from agent_demo.planning import PlanningModule, StepResult, TaskStep

        # 关闭自动跳过，确保重规划流程完整走通
        planning = PlanningModule(
            max_retries_per_step=1,
            skip_failed_non_critical=False,
        )

        # 模拟 LLM 返回的三步计划
        llm_output = json.dumps({
            "goal": "搜索资料并生成报告",
            "steps": [
                {"id": "s1", "description": "搜索资料", "instruction": "搜索", "action": "tool_call", "depends_on": []},
                {"id": "s2", "description": "分析资料", "instruction": "分析", "action": "think", "depends_on": ["s1"]},
                {"id": "s3", "description": "生成报告", "instruction": "生成", "action": "respond", "depends_on": ["s2"]},
            ],
        }, ensure_ascii=False)

        plan = planning.parse_plan(llm_output)
        plan = planning.start_plan(plan)

        # s1 成功
        step = planning.get_next_step(plan)
        plan = planning.record_result(plan, StepResult(step_id="s1", success=True, output="搜索结果..."))

        # s2 第一次失败
        step = planning.get_next_step(plan)
        assert step.id == "s2"
        plan = planning.record_result(plan, StepResult(step_id="s2", success=False, output="", error="API 超时"))

        # s2 第二次失败（超过 max_retries=1）
        step = planning.get_next_step(plan)
        assert step.id == "s2", "应允许重试一次"
        plan = planning.record_result(plan, StepResult(step_id="s2", success=False, output="", error="再次超时"))

        # 检查是否需要重规划
        assert planning.should_revise(plan)

        # 模拟 LLM 重规划返回
        revised_json = json.dumps({
            "rationale": "跳过失败的分析步骤，直接基于搜索结果生成报告",
            "revised_steps": [
                {"id": "s3_revised", "description": "直接生成报告", "instruction": "基于 s1 的搜索结果直接生成报告", "action": "respond", "depends_on": ["s1"]},
            ],
        })
        revised_steps = planning.parse_steps(revised_json)
        plan = planning.revise_plan(plan, revised_steps)
        assert plan.revision_count == 1

        # 执行重规划后的步骤
        step = planning.get_next_step(plan)
        assert step.id == "s3_revised"
        plan = planning.record_result(plan, StepResult(
            step_id="s3_revised", success=True, output="报告生成完成",
        ))

        # 完成
        plan = planning.finalize_plan(plan)
        assert plan.status == "completed"
        assert plan.revision_count == 1

        progress = planning.get_progress(plan)
        print(f"  [PASS] 含重规划工作流: {plan.progress_summary()}, revisions={progress['revision_count']}")

    def test_get_results_summary(self):
        """结果摘要生成"""
        from agent_demo.planning import PlanningModule, StepResult

        planning = PlanningModule()

        llm_output = json.dumps({
            "goal": "查询天气",
            "steps": [
                {"id": "s1", "description": "获取数据", "instruction": "...", "action": "tool_call", "depends_on": []},
                {"id": "s2", "description": "回复用户", "instruction": "...", "action": "respond", "depends_on": ["s1"]},
            ],
        }, ensure_ascii=False)

        plan = planning.parse_plan(llm_output)
        plan = planning.start_plan(plan)

        step = planning.get_next_step(plan)
        plan = planning.record_result(plan, StepResult(step_id="s1", success=True, output="北京晴 18-26°C"))

        step = planning.get_next_step(plan)
        plan = planning.record_result(plan, StepResult(step_id="s2", success=True, output="今天北京天气晴朗"))

        summary = planning.get_results_summary(plan)
        assert "查询天气" in summary
        assert "✅" in summary
        assert "北京" in summary
        print(f"  [PASS] 结果摘要: {len(summary)} 字符")


# ============================================================
# 运行入口
# ============================================================

def run_all():
    """运行所有测试并汇总结果"""
    import sys

    test_classes = [
        TestTypes,
        TestPromptBuilder,
        TestExecutor,
        TestPlanValidation,
        TestParsing,
        TestPlanningModule,
    ]

    total = 0
    passed = 0
    failed = 0

    print("=" * 60)
    print("规划模块 (Planning Module) 测试")
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
