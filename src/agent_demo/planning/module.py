"""PlanningModule —— 规划模块主入口

纯状态机架构 —— 不持有 LLM/Tools 引用，由 Orchestrator 驱动执行。

使用示例:
    from agent_demo.planning import PlanningModule

    planning = PlanningModule(max_retries_per_step=2, max_revisions=3)

    # Orchestrator 使用模式:
    # 1. 构建计划生成 prompt
    prompt = planning.build_plan_prompt(
        intent="帮我在网上搜索 Python 和 Rust 的性能对比，生成一份报告",
        tool_schemas=tools.get_schemas(),
        context=memory.retrieve("性能对比"),
    )
    # 2. Orchestrator 调用 LLM 生成计划
    llm_response = llm.chat(messages=[Message(content=prompt)])
    plan = planning.parse_plan(llm_response.content)

    # 3. Orchestrator 驱动执行循环
    plan = planning.start_plan(plan)
    while (step := planning.get_next_step(plan)):
        step_prompt = planning.build_step_prompt(plan, step)
        result = llm.chat(messages=[Message(content=step_prompt)], tools=...)
        plan = planning.record_result(plan, StepResult(
            step_id=step.id,
            success=True,
            output=result.content,
        ))
        if planning.should_revise(plan):
            revise_prompt = planning.build_revise_prompt(plan, failed_id, error)
            new_plan_json = llm.chat(messages=[Message(content=revise_prompt)])
            revised_steps = planning.parse_steps(new_plan_json.content)
            plan = planning.revise_plan(plan, revised_steps)

    # 4. 获取最终结果
    plan = planning.finalize_plan(plan)
"""

import json
import logging
from .types import TaskPlan, TaskStep, StepResult
from .executor import PlanExecutor
from .prompt_builder import PlanPromptBuilder

logger = logging.getLogger(__name__)


class PlanningModule:
    """
    规划模块 —— Agent 的计划生成、状态管理与执行编排

    设计决策（基于主流 Agent 调研）:
        - 纯状态机: 不持有 LLM/Tools/Memory 引用，由 Orchestrator 驱动
        - Plan-and-Execute: 所有任务先出完整计划再逐步执行
        - 借鉴 Claude Code: "计划质量取决于 prompt"
        - 借鉴 Codex: "计划是自包含活文档"
        - 借鉴 Anthropic: "一次只做一件事"
    """

    def __init__(
        self,
        max_retries_per_step: int = 2,
        max_revisions: int = 3,
        skip_failed_non_critical: bool = True,
    ):
        """
        Args:
            max_retries_per_step: 单步失败后的最大重试次数
            max_revisions: 整个计划的最大重规划次数
            skip_failed_non_critical: 非关键步骤失败时是否跳过继续
        """
        self._executor = PlanExecutor(
            max_retries_per_step=max_retries_per_step,
            max_revisions=max_revisions,
            skip_failed_non_critical=skip_failed_non_critical,
        )
        self._prompt_builder = PlanPromptBuilder()
        logger.info(
            f"PlanningModule 初始化完成，"
            f"max_retries={max_retries_per_step}，"
            f"max_revisions={max_revisions}"
        )

    # ---- Prompt 构建（给 Orchestrator 用）----

    def build_plan_prompt(
        self,
        intent: str,
        context: str | None = None,
        tool_schemas: list[dict] | None = None,
    ) -> str:
        """
        构建计划生成 prompt —— Orchestrator 将此发给 LLM 生成计划

        Args:
            intent: 用户意图描述
            context: 记忆模块检索到的上下文
            tool_schemas: 可用工具列表（OpenAI Function Calling 格式）

        Returns:
            可直接发给 LLM 的 prompt 文本
        """
        return self._prompt_builder.build_plan_prompt(
            intent=intent,
            context=context,
            tool_schemas=tool_schemas,
        )

    def build_step_prompt(self, plan: TaskPlan, step: TaskStep) -> str:
        """
        构建步骤执行 prompt —— Orchestrator 将此发给 LLM 执行步骤

        Args:
            plan: 完整任务计划
            step: 要执行的步骤

        Returns:
            步骤执行 prompt 文本
        """
        return self._prompt_builder.build_step_prompt(plan, step)

    def build_revise_prompt(
        self,
        plan: TaskPlan,
        failed_step_id: str,
        error_description: str,
    ) -> str:
        """
        构建重规划 prompt —— Orchestrator 将此发给 LLM 生成新步骤

        Args:
            plan: 当前计划
            failed_step_id: 失败的步骤 ID
            error_description: 失败原因

        Returns:
            重规划 prompt 文本
        """
        return self._prompt_builder.build_revise_prompt(
            plan, failed_step_id, error_description
        )

    # ---- 计划解析（从 LLM JSON 输出中提取）----

    def parse_plan(self, llm_output: str) -> TaskPlan:
        """
        从 LLM 的 JSON 输出中解析 TaskPlan

        能处理 LLM 输出中常见的格式问题:
            - JSON 前后有 markdown 代码块标记 (```json ... ```)
            - JSON 前后有额外文字
            - 缺少 created_at 等非必要字段

        Args:
            llm_output: LLM 的原始输出文本

        Returns:
            解析后的 TaskPlan 对象

        Raises:
            ValueError: JSON 格式无效或缺少必要字段
        """
        json_str = self._extract_json(llm_output)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"无法解析 LLM 输出中的 JSON。原始输出前 200 字符:\n"
                f"{llm_output[:200]}...\n错误: {e}"
            )

        # 验证必要字段
        if "goal" not in data:
            raise ValueError("计划 JSON 缺少 'goal' 字段")
        if "steps" not in data or not isinstance(data["steps"], list):
            raise ValueError("计划 JSON 缺少 'steps' 数组")

        # 解析步骤
        steps = self.parse_steps(llm_output)

        plan = TaskPlan(
            goal=data["goal"],
            steps=steps,
            created_at=data.get("created_at", 0),
            revision_count=data.get("revision_count", 0),
        )

        # 验证
        errors = self._executor.validate(plan)
        if errors:
            raise ValueError(
                f"计划验证失败:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        logger.info(
            f"计划已解析: goal={plan.goal[:50]}..., "
            f"steps={len(plan.steps)}"
        )
        return plan

    def parse_steps(self, llm_output: str) -> list[TaskStep]:
        """
        从 LLM JSON 输出中解析步骤列表

        用于重规划场景——LLM 只返回 revised_steps 而不是完整计划。

        Args:
            llm_output: LLM 原始输出文本

        Returns:
            TaskStep 列表
        """
        json_str = self._extract_json(llm_output)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError(f"无法解析步骤 JSON: {llm_output[:200]}...")

        # 兼容两种格式: 完整计划 {"steps": [...]} 或纯步骤数组 [...]
        if isinstance(data, list):
            step_list = data
        elif "revised_steps" in data:
            step_list = data["revised_steps"]
        elif "steps" in data:
            step_list = data["steps"]
        else:
            raise ValueError(
                "JSON 中找不到步骤列表。期望 'steps'、'revised_steps' 或顶层数组。"
            )

        steps = []
        required_fields = ["id", "description", "instruction", "action"]
        for s in step_list:
            for field in required_fields:
                if field not in s:
                    raise ValueError(
                        f"步骤缺少必要字段 '{field}': {s}"
                    )
            steps.append(TaskStep(
                id=s["id"],
                description=s["description"],
                instruction=s["instruction"],
                action=s["action"],
                depends_on=s.get("depends_on", []),
                status=s.get("status", "pending"),
            ))

        return steps

    # ---- 状态管理（给 Orchestrator 用）----

    def start_plan(self, plan: TaskPlan) -> TaskPlan:
        """标记计划开始执行"""
        return self._executor.start(plan)

    def get_next_step(self, plan: TaskPlan) -> TaskStep | None:
        """获取下一个可执行的步骤"""
        return self._executor.get_next_step(plan)

    def record_result(self, plan: TaskPlan, result: StepResult) -> TaskPlan:
        """记录步骤执行结果"""
        return self._executor.record_result(plan, result)

    def should_revise(self, plan: TaskPlan) -> bool:
        """判断是否需要重规划"""
        return self._executor.should_revise(plan)

    def revise_plan(
        self, plan: TaskPlan, revised_steps: list[TaskStep]
    ) -> TaskPlan:
        """重规划：用新步骤替换失败步骤及后续"""
        return self._executor.revise(plan, revised_steps)

    def is_complete(self, plan: TaskPlan) -> bool:
        """判断计划是否已完成"""
        return self._executor.is_complete(plan)

    def finalize_plan(self, plan: TaskPlan) -> TaskPlan:
        """完成计划，设置最终状态"""
        return self._executor.finalize(plan)

    # ---- 进度查询 ----

    def get_progress(self, plan: TaskPlan) -> dict:
        """获取执行进度摘要"""
        completed = len(plan.get_completed_steps())
        total = len(plan.steps)
        failed = len(plan.get_failed_steps())

        return {
            "goal": plan.goal,
            "status": plan.status,
            "progress": f"{completed}/{total}",
            "percent": round(completed / total * 100, 1) if total else 0,
            "completed": completed,
            "total": total,
            "failed": failed,
            "current_step": (
                plan.steps[plan.current_step_index].id
                if plan.current_step_index < len(plan.steps)
                else None
            ),
            "revision_count": plan.revision_count,
            "summary": plan.progress_summary(),
        }

    def get_results_summary(self, plan: TaskPlan) -> str:
        """获取所有步骤结果的文本摘要 —— 适合作为最终回复的一部分"""
        parts = [f"## 任务完成情况\n目标: {plan.goal}\n"]

        for step in plan.steps:
            result = plan.get_result(step.id)
            status_icon = {
                "completed": "✅", "failed": "❌",
                "skipped": "⏭️", "pending": "⬜",
            }.get(step.status, "❓")

            parts.append(f"{status_icon} **{step.description}**")
            if result:
                output_preview = result.output[:150].replace("\n", " ")
                parts.append(f"   {output_preview}")
                if result.error:
                    parts.append(f"   错误: {result.error}")

        return "\n".join(parts)

    # ---- 内部辅助 ----

    def _extract_json(self, text: str) -> str:
        """
        从 LLM 原始输出中提取 JSON 字符串

        处理:
            - ```json ... ``` 代码块
            - ``` ... ``` 代码块
            - 前后有额外文字的情况
        """
        text = text.strip()

        # 尝试从 markdown 代码块中提取
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.rindex("```")
            if end > start:
                return text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.rindex("```")
            if end > start:
                return text[start:end].strip()

        # 尝试找到第一个 { 和最后一个 }
        brace_start = text.find("{")
        bracket_start = text.find("[")
        if brace_start == -1 and bracket_start == -1:
            return text  # 原样返回，让 json.loads 报错

        if brace_start != -1 and (bracket_start == -1 or brace_start < bracket_start):
            # JSON 对象
            brace_end = text.rfind("}")
            if brace_end > brace_start:
                return text[brace_start:brace_end + 1]
        elif bracket_start != -1:
            # JSON 数组
            bracket_end = text.rfind("]")
            if bracket_end > bracket_start:
                return text[bracket_start:bracket_end + 1]

        return text
