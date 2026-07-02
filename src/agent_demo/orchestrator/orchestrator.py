"""AgentOrchestrator —— 五大模块的中央编排器

职责:
    1. 按正确顺序初始化五大模块并注入依赖
    2. 驱动 Plan-and-Execute 主循环
    3. 管理感知 → 记忆 → 规划 → 执行 → 收尾的完整数据流
    4. 统一错误处理和结果汇总

编排器自身不实现任何 AI 能力 —— 所有能力来自五大模块的协作。

使用示例:
    from agent_demo.orchestrator import AgentOrchestrator, OrchestratorConfig

    agent = AgentOrchestrator()
    result = agent.run("帮我搜索 Python 和 Rust 的性能对比")
    print(result.content)
"""

import logging
from ..perception import PerceptionModule
from ..memory import MemoryModule
from ..planning import PlanningModule, TaskPlan, TaskStep, StepResult
from ..llm import LLMModule, LLMResponse
from ..tools import ToolsModule

from .config import OrchestratorConfig, OrchestratorResult

logger = logging.getLogger(__name__)

# ── 默认系统提示词 ─────────────────────────────────────────────
DEFAULT_SYSTEM_PROMPT = """\
你是一个智能 AI 助手，具备任务规划和工具调用能力。

## 工作方式

对于**简单对话**（问候、常识问答、闲聊），直接回复用户。
对于**复杂任务**（搜索、数据分析、文件操作、多步推理），你需要：
1. 先分析用户意图，生成详细的执行计划
2. 逐步执行每个步骤，必要时调用工具获取信息
3. 最终生成完整、有条理的回复

## 行为准则
- 始终以中文回复用户
- 调用工具前先确认参数是否正确
- 步骤失败时尝试替代方案
- 回复要简洁但信息完整"""


class AgentOrchestrator:
    """Agent 核心编排器 —— 五大模块的中央协调者

    架构:
        AgentOrchestrator
        ├── PerceptionModule   # 输入感知（文本/文件 → Message）
        ├── MemoryModule       # 三层记忆（检索上下文 / 保存对话）
        ├── PlanningModule     # 任务规划（纯状态机）
        ├── LLMModule          # LLM 调用（含工具调用闭环）
        └── ToolsModule        # 工具注册与执行

    主流程:
        原始输入 → 感知 → 记忆检索 → 生成计划 → 逐步执行 → 记忆写入 → 返回结果
    """

    def __init__(self, config: OrchestratorConfig | None = None):
        """
        Args:
            config: 编排器配置。None 则使用全默认配置（API Key 从环境变量读取）。
        """
        self.config = config or OrchestratorConfig()

        # ── 1. ToolsModule（最先初始化，LLM 依赖它的 executor）──
        self._tools = ToolsModule()

        # ── 2. LLMModule（注入 tool_executor 回调，解耦）──
        self._llm = LLMModule(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            model=self.config.model,
            system_prompt=self.config.system_prompt or DEFAULT_SYSTEM_PROMPT,
            tool_executor=self._tools.get_executor(),
            max_tool_rounds=self.config.max_tool_rounds,
        )

        # ── 3. MemoryModule ──
        self._memory = MemoryModule(
            embedder_provider=self.config.embedder_provider,
            persist_dir=self.config.memory_persist_dir,
            max_working_tokens=self.config.max_working_tokens,
            system_prompt=self.config.system_prompt,
            api_key=self.config.api_key,
        )

        # ── 4. PlanningModule（纯状态机，不持有 LLM/Tools）──
        self._planning = PlanningModule(
            max_retries_per_step=self.config.max_retries_per_step,
            max_revisions=self.config.max_revisions,
        )

        # ── 5. PerceptionModule ──
        self._perception = PerceptionModule(
            max_length=self.config.max_input_length,
        )

        # 单次运行的统计（在 run() 开始时重置）
        self._run_tokens: dict = {}
        self._run_tool_calls: int = 0

        logger.info(
            f"AgentOrchestrator 初始化完成，"
            f"model={self.config.model}, "
            f"embedder={self.config.embedder_provider}"
        )

    # ── 主入口 ─────────────────────────────────────────────────

    def run(self, raw_input) -> OrchestratorResult:
        """处理一次用户输入，返回最终回复。

        这是编排器唯一需要调用的方法。内部自动完成:
        感知 → 记忆检索 → 计划生成 → 逐步执行 → 收尾。

        Args:
            raw_input: 用户原始输入（str / 文件路径）

        Returns:
            OrchestratorResult（包含最终回复和执行元信息）
        """
        # 重置本轮统计
        self._run_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._run_tool_calls = 0

        # ── Phase 1: 感知 ──
        msg = self._perception.process(raw_input)
        if msg.is_rejected:
            logger.warning(f"输入被拒绝: {msg.reject_reason}")
            return OrchestratorResult(
                content=msg.content,
                success=False,
                error=f"输入被拒绝: {msg.reject_reason}",
            )

        logger.info(f"感知完成: source={msg.source_type}, len={len(msg.content)}")

        # ── Phase 2: 记忆检索 ──
        context = self._memory.retrieve(msg.content, top_k=5)
        logger.debug(f"记忆检索完成: context_len={len(context)}")

        # ── Phase 3: 生成计划 ──
        try:
            plan = self._generate_plan(msg.content, context)
        except ValueError as e:
            logger.error(f"计划生成失败: {e}")
            return OrchestratorResult(
                content=f"抱歉，我无法理解您的请求。请尝试更具体地描述您的需求。\n\n（调试信息：计划生成失败 — {e}）",
                success=False,
                error=str(e),
            )

        # ── Phase 4: 执行循环 ──
        try:
            plan = self._planning.start_plan(plan)
        except ValueError as e:
            logger.error(f"计划验证失败: {e}")
            return OrchestratorResult(
                content=f"任务计划存在逻辑问题，无法执行。\n\n（调试信息：{e}）",
                success=False,
                plan=plan,
                error=str(e),
            )

        plan = self._execute_loop(plan)

        # ── Phase 5: 收尾 ──
        plan = self._planning.finalize_plan(plan)

        # 写入记忆
        self._memory.remember(self._build_memory_messages(msg, plan))

        # 构建结果
        result_content = self._planning.get_results_summary(plan)
        return OrchestratorResult(
            content=result_content,
            success=plan.status == "completed",
            plan=plan,
            total_tool_calls=self._run_tool_calls,
            total_tokens=self._run_tokens.copy(),
        )

    # ── 计划生成 ───────────────────────────────────────────────

    def _generate_plan(self, intent: str, context: str, max_retries: int = 2) -> TaskPlan:
        """生成任务计划 —— 调用 LLM 获取 JSON 计划并解析

        JSON 格式错误时自动重试（将解析错误反馈给 LLM）。
        计划逻辑错误（循环依赖、空步骤等）直接上抛，不重试。
        """
        base_prompt = self._planning.build_plan_prompt(
            intent=intent,
            context=context if context else None,
            tool_schemas=self._tools.get_schemas() if self._tools.list_tools() else None,
        )

        prompt = base_prompt
        for attempt in range(max_retries):
            self._llm.clear_history()
            response = self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=None,  # 计划生成阶段不调用工具
            )
            self._accumulate_tokens(response)

            try:
                plan = self._planning.parse_plan(response.content)
                logger.info(
                    f"计划生成成功: goal={plan.goal[:50]}..., "
                    f"steps={len(plan.steps)}"
                )
                return plan
            except ValueError as e:
                error_msg = str(e)
                # 计划逻辑错误（循环依赖、空步骤等）→ 不重试，直接上抛
                if "计划验证失败" in error_msg or "缺少 'goal'" in error_msg or "缺少 'steps'" in error_msg:
                    raise
                # JSON 格式错误 → 重试
                logger.warning(f"计划 JSON 解析失败 (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise ValueError(f"计划生成失败：JSON 解析错误 — {e}")
                # 重试：将错误反馈给 LLM
                prompt = (
                    f"{base_prompt}\n\n"
                    f"【重要提示】上次输出的 JSON 解析失败，错误信息：\n"
                    f"{e}\n\n"
                    f"请确保：\n"
                    f"1. 输出严格的 JSON 格式\n"
                    f"2. 不要添加 ```json 代码块标记之外的额外文字\n"
                    f"3. 每个步骤必须包含 id, description, instruction, action 四个必要字段\n"
                    f"4. depends_on 必须是有效的步骤 ID 列表"
                )

        # 不应该到达这里
        raise ValueError("计划生成失败：超过最大重试次数")

    # ── 执行循环 ───────────────────────────────────────────────

    def _execute_loop(self, plan: TaskPlan) -> TaskPlan:
        """Plan-and-Execute 主循环

        由 PlanningModule 的状态机驱动，编排器负责:
        - 为每一步构建 prompt 并调用 LLM
        - 检测失败并触发重规划
        """
        while (step := self._planning.get_next_step(plan)):
            logger.info(
                f"执行步骤: {step.id} action={step.action} "
                f"progress={self._planning.get_progress(plan)['progress']}"
            )

            step_result = self._execute_step(plan, step)
            plan = self._planning.record_result(plan, step_result)

            if self._planning.should_revise(plan):
                logger.info(f"触发重规划 (第 {plan.revision_count + 1} 次)")
                try:
                    plan = self._revise_plan(plan)
                except ValueError as e:
                    logger.error(f"重规划失败: {e}，终止执行")
                    break

        return plan

    def _execute_step(self, plan: TaskPlan, step: TaskStep) -> StepResult:
        """执行单个步骤 —— 构建 prompt → LLM 推理（含工具调用）

        步骤之间完全独立：每次执行前清空 LLM 对话历史，
        步骤的上下文由 PlanPromptBuilder 自包含地提供。
        """
        prompt = self._planning.build_step_prompt(plan, step)

        # 根据步骤类型决定是否传入工具
        tools = self._tools.get_schemas() if step.action == "tool_call" else None

        self._llm.clear_history()
        response = self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
        )
        self._accumulate_tokens(response)
        self._run_tool_calls += len(response.tool_calls_log)

        return StepResult(
            step_id=step.id,
            success=response.finish_reason != "error",
            output=response.content,
            error=response.content if response.finish_reason == "error" else None,
            metadata={
                "finish_reason": response.finish_reason,
                "tool_calls_log": response.tool_calls_log,
            },
        )

    # ── 重规划 ──────────────────────────────────────────────────

    def _revise_plan(self, plan: TaskPlan) -> TaskPlan:
        """执行一次重规划 —— 让 LLM 重新设计后续步骤"""
        # 找到最终失败的步骤
        failed_step = None
        for s in plan.steps:
            if s.status == "failed":
                failed_step = s
                break

        if failed_step is None:
            logger.warning("重规划被调用但没有找到失败步骤")
            return plan

        # 获取失败原因
        result = plan.get_result(failed_step.id)
        error_desc = "未知错误"
        if result:
            if result.error:
                error_desc = result.error
            elif result.output:
                error_desc = result.output[:500]

        # 构建重规划 prompt → LLM 生成新步骤
        prompt = self._planning.build_revise_prompt(plan, failed_step.id, error_desc)

        self._llm.clear_history()
        response = self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
        )
        self._accumulate_tokens(response)

        revised_steps = self._planning.parse_steps(response.content)
        return self._planning.revise_plan(plan, revised_steps)

    # ── 记忆 ────────────────────────────────────────────────────

    def _build_memory_messages(
        self, user_msg, plan: TaskPlan
    ) -> list[dict]:
        """构建写入记忆的消息列表"""
        summary = self._planning.get_results_summary(plan)
        return [
            {"role": "user", "content": user_msg.content},
            {"role": "assistant", "content": summary},
        ]

    # ── 内部辅助 ────────────────────────────────────────────────

    def _accumulate_tokens(self, response: LLMResponse) -> None:
        """累加 token 用量"""
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self._run_tokens[key] = (
                self._run_tokens.get(key, 0) + response.token_usage.get(key, 0)
            )

    # ── 模块访问（只读，用于调试和自定义）────────────────────

    @property
    def perception(self) -> PerceptionModule:
        """感知模块"""
        return self._perception

    @property
    def memory(self) -> MemoryModule:
        """记忆模块"""
        return self._memory

    @property
    def planning(self) -> PlanningModule:
        """规划模块"""
        return self._planning

    @property
    def llm(self) -> LLMModule:
        """LLM 模块"""
        return self._llm

    @property
    def tools(self) -> ToolsModule:
        """工具模块"""
        return self._tools

    # ── 会话管理 ────────────────────────────────────────────────

    def clear_session(self) -> None:
        """清空所有会话状态（对话历史 + 工作记忆 + 会话变量）

        注意：长期记忆保留。
        """
        self._llm.clear_history()
        self._memory.clear_session()
        logger.info("会话状态已清空")

    def __repr__(self) -> str:
        return (
            f"AgentOrchestrator(model={self.config.model}, "
            f"tools={len(self._tools.list_tools())}, "
            f"embedder={self.config.embedder_provider})"
        )
