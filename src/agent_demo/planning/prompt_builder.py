"""PlanPromptBuilder —— 构建让 LLM 产出结构化计划的 prompt

借鉴 Claude Code TodoWrite 和 Codex ExecPlan 的设计:
    - 计划质量取决于 prompt 写得好不好
    - 明确约束输出格式（JSON）
    - 提供上下文（工具 schema + 前置结果 + 记忆）

三类 prompt:
    1. build_plan_prompt()   → 生成初始计划
    2. build_step_prompt()    → 执行单个步骤
    3. build_revise_prompt()  → 重规划后续步骤
"""

import json
from .types import TaskPlan, TaskStep, StepResult


class PlanPromptBuilder:
    """
    计划 Prompt 构建器 —— 负责生成三种关键 prompt

    设计原则:
        - 每个 prompt 都包含明确的 JSON 输出格式约束
        - 借鉴 Claude Code: "计划是 prompt 工程，不是代码逻辑"
        - 借鉴 Codex: "计划是自包含的，只看 prompt 就能理解任务"
    """

    # ---- 1. 计划生成 Prompt ----

    def build_plan_prompt(
        self,
        intent: str,
        context: str | None = None,
        tool_schemas: list[dict] | None = None,
    ) -> str:
        """
        构建生成初始计划的 prompt

        Args:
            intent: 用户意图描述
            context: 记忆模块检索到的上下文（偏好、历史对话等）
            tool_schemas: 可用工具的 OpenAI Function Calling 格式列表

        Returns:
            可直接发给 LLM 的 prompt 文本
        """
        parts = []

        # 角色设定
        parts.append(
            "你是一个任务规划专家。你的职责是将用户的请求分解为清晰、可执行的步骤序列。\n"
            "\n"
            "## 规划原则\n"
            "- 每个步骤必须是单一、明确的操作（one thing at a time）\n"
            "- 步骤之间有依赖关系时必须声明 depends_on\n"
            "- 优先考虑可并行执行的步骤（减少 depends_on）\n"
            "- 简单任务 1-3 步，中等任务 3-7 步，复杂任务 7-15 步\n"
            "- 步骤描述要具体，包含做什么、用什么工具、期望产出什么\n"
        )

        # 用户意图
        parts.append(f"## 用户请求\n{intent}")

        # 记忆上下文
        if context:
            parts.append(f"## 相关上下文\n{context}")

        # 可用工具
        if tool_schemas:
            tools_desc = self._format_tools_summary(tool_schemas)
            parts.append(f"## 可用工具\n{tools_desc}")

        # 输出格式约束
        parts.append(
            "## 输出格式\n"
            "请严格按以下 JSON 格式返回计划，不要输出任何其他内容：\n"
            "```json\n"
            "{\n"
            '  "goal": "任务目标的一句话描述",\n'
            '  "steps": [\n'
            "    {\n"
            '      "id": "step_1",\n'
            '      "description": "人类可读的步骤描述（给用户看的）",\n'
            '      "instruction": "给 AI 的具体执行指令（包含足够的上下文让 AI 独立完成）",\n'
            '      "action": "tool_call|think|respond|ask_user",\n'
            '      "depends_on": []\n'
            "    },\n"
            "    {\n"
            '      "id": "step_2",\n'
            '      "description": "整理搜索结果，生成对比报告",\n'
            '      "instruction": "根据上一步的搜索结果，将 Python 和 Rust 在 Web 开发方面的特点进行对比，列出优势和劣势，生成一份结构化的对比摘要。",\n'
            '      "action": "think",\n'
            '      "depends_on": ["step_1"]\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n"
            "\n"
            "## Action 类型说明\n"
            "- **tool_call**: 需要调用工具完成的操作（搜索、文件读写、计算等）\n"
            "- **think**: 纯推理/分析/总结，不需要调用工具\n"
            "- **respond**: 生成给用户的最终回复\n"
            "- **ask_user**: 需要向用户询问更多信息\n"
        )

        return "\n\n".join(parts)

    # ---- 2. 步骤执行 Prompt ----

    def build_step_prompt(
        self,
        plan: TaskPlan,
        step: TaskStep,
    ) -> str:
        """
        构建执行单个步骤的 prompt

        每个步骤的 prompt 是自包含的——包含目标、前置结果和当前指令，
        借鉴 Codex ExecPlan 的"自包含文档"理念。

        Args:
            plan: 完整的任务计划（用于获取上下文）
            step: 当前要执行的步骤

        Returns:
            可直接发给 LLM 的步骤执行 prompt
        """
        parts = []

        # 任务目标
        parts.append(f"## 任务目标\n{plan.goal}")

        # 整体计划概览（让 LLM 知道自己在计划中的位置）
        parts.append("## 整体计划进度")
        for s in plan.steps:
            marker = "← 当前步骤" if s.id == step.id else ""
            status_icon = {
                "completed": "✅", "failed": "❌", "running": "🔄",
                "skipped": "⏭️", "pending": "⬜",
            }.get(s.status, "⬜")
            parts.append(f"  {status_icon} {s.description} {marker}")

        # 前置步骤的结果（关键！）
        if step.depends_on:
            parts.append("## 前置步骤的结果")
            for dep_id in step.depends_on:
                result = plan.get_result(dep_id)
                if result:
                    status = "成功" if result.success else "失败"
                    parts.append(
                        f"### {dep_id} ({status})\n{result.output}"
                    )
                else:
                    parts.append(f"### {dep_id}\n（结果不可用）")

        # 当前步骤指令
        parts.append(
            f"## 当前任务\n{step.instruction}\n"
            "\n请完成上述任务。如果需要调用工具，请直接调用。"
            "完成后用简洁的语言总结你的执行结果。"
        )

        return "\n\n".join(parts)

    # ---- 3. 重规划 Prompt ----

    def build_revise_prompt(
        self,
        plan: TaskPlan,
        failed_step_id: str,
        error_description: str,
    ) -> str:
        """
        构建重规划 prompt —— 某步骤失败后，让 LLM 重新规划后续步骤

        Args:
            plan: 当前的 TaskPlan（包含已完成的步骤和结果）
            failed_step_id: 失败的步骤 ID
            error_description: 失败原因描述

        Returns:
            重规划 prompt（LLM 应返回新的后续步骤 JSON）
        """
        parts = []

        parts.append(
            "你是一个任务规划专家。执行计划中的某个步骤失败了，"
            "你需要重新规划后续步骤来完成任务目标。\n"
        )

        # 任务目标
        parts.append(f"## 原始任务目标\n{plan.goal}")

        # 已完成步骤
        completed = plan.get_completed_steps()
        if completed:
            parts.append("## 已成功完成的步骤")
            for s in completed:
                r = plan.get_result(s.id)
                if r:
                    parts.append(
                        f"- {s.id}: {s.description}\n"
                        f"  结果: {r.output[:200]}..."
                    )

        # 失败步骤
        parts.append(f"## 失败的步骤\n- {failed_step_id}: {error_description}")

        # 待执行的步骤
        remaining = [
            s for s in plan.get_remaining_steps()
            if s.id != failed_step_id
        ]
        if remaining:
            parts.append("## 原计划的后续步骤（可参考或调整）")
            for s in remaining:
                parts.append(f"- {s.id}: {s.description} (depends_on={s.depends_on})")

        # 输出格式
        parts.append(
            "## 输出格式\n"
            "请根据失败情况，重新规划后续步骤。可以：\n"
            "- 重试失败步骤（换个方式）\n"
            "- 跳过失败步骤（如果不影响最终目标）\n"
            "- 调整后续步骤的顺序和指令\n"
            "\n"
            "返回严格的 JSON 格式（仅包含需要替换/新增的步骤）：\n"
            "```json\n"
            "{\n"
            '  "rationale": "重规划理由（一句话）",\n'
            '  "revised_steps": [\n'
            "    {\n"
            '      "id": "step_3_revised",\n'
            '      "description": "...",\n'
            '      "instruction": "...",\n'
            '      "action": "tool_call",\n'
            '      "depends_on": ["step_1"]\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n"
            "\n"
            "注意: revised_steps 中只包含需要新增或替换的步骤。已完成步骤不需要列出。"
        )

        return "\n\n".join(parts)

    # ---- 辅助 ----

    def _format_tools_summary(self, tool_schemas: list[dict]) -> str:
        """将工具 schema 列表格式化为人类可读的摘要（用于计划生成 prompt）"""
        lines = []
        for schema in tool_schemas:
            func = schema.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "无描述")
            params = func.get("parameters", {})
            required = params.get("required", [])
            properties = params.get("properties", {})

            # 格式化参数
            param_list = []
            for pname, pinfo in properties.items():
                req_mark = " *必填*" if pname in required else ""
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                param_list.append(f"    - {pname} ({ptype}){req_mark}: {pdesc}")

            param_str = "\n".join(param_list) if param_list else "    无参数"
            lines.append(f"- **{name}**: {desc}\n{param_str}")

        return "\n".join(lines)
