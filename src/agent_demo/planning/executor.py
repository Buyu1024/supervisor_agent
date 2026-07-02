"""PlanExecutor —— 计划执行器（纯状态机）

职责:
    1. 管理计划执行状态（生命周期管理）
    2. 依赖解析和拓扑排序（确定下一步执行哪个步骤）
    3. 步骤结果记录
    4. 重规划决策和失败处理

设计决策（借鉴主流 Agent）:
    - 借鉴 Claude Code "一次一步"原则: 每步执行完明确记录结果
    - 借鉴 Codex ExecPlan "活文档"理念: 计划状态在整个执行中持续更新
    - 借鉴 Anthropic 指导: 步骤失败不立即中止，先判断能否重规划
    - 纯 Python 逻辑，不调 API，不依赖外部模块
"""

import logging
from .types import TaskPlan, TaskStep, StepResult

logger = logging.getLogger(__name__)


class PlanExecutor:
    """
    计划执行器 —— 纯状态机，管理 TaskPlan 的执行生命周期

    使用示例:
        plan = TaskPlan(goal="...", steps=[...])
        executor = PlanExecutor(max_retries_per_step=2)

        # Orchestrator 驱动循环:
        plan = executor.start(plan)
        while (step := executor.get_next_step(plan)):
            # Orchestrator 调用 LLM 执行 step
            result = orchestrator.run_step(step)
            plan = executor.record_result(plan, result)

            if executor.should_revise(plan):
                # Orchestrator 调用 LLM 生成新步骤
                new_steps = orchestrator.revise(plan)
                plan = executor.revise(plan, new_steps)

        status = executor.finalize(plan)
    """

    def __init__(
        self,
        max_retries_per_step: int = 2,
        max_revisions: int = 3,
        skip_failed_non_critical: bool = True,
    ):
        """
        Args:
            max_retries_per_step: 单个步骤失败后的最大重试次数
            max_revisions: 整个计划的最大重规划次数
            skip_failed_non_critical: 非关键步骤失败时是否跳过继续
        """
        self.max_retries_per_step = max_retries_per_step
        self.max_revisions = max_revisions
        self.skip_failed_non_critical = skip_failed_non_critical

        # 内部重试计数: step_id → retry count
        self._retry_count: dict[str, int] = {}

    # ---- 生命周期 ----

    def start(self, plan: TaskPlan) -> TaskPlan:
        """
        标记计划开始执行

        验证计划合法性，将状态从 pending 变为 running。

        Args:
            plan: 待执行的计划

        Returns:
            更新后的计划（status="running"）

        Raises:
            ValueError: 计划不合法（无步骤、循环依赖等）
        """
        # 验证
        errors = self.validate(plan)
        if errors:
            raise ValueError(f"计划不合法:\n" + "\n".join(f"  - {e}" for e in errors))

        plan.status = "running"
        plan.current_step_index = 0
        logger.info(f"计划开始执行: {plan.goal[:60]}... ({len(plan.steps)} 步)")
        return plan

    def get_next_step(self, plan: TaskPlan) -> TaskStep | None:
        """
        获取下一个可执行的步骤

        选择逻辑:
            1. 只考虑 status="pending" 或 status="failed"（重试）的步骤
            2. 检查依赖：所有 depends_on 中的步骤必须已完成
            3. 按步骤在 steps 列表中的顺序返回第一个满足条件的
            4. 不跳过依赖未满足的步骤

        Args:
            plan: 当前计划

        Returns:
            可执行的下一个步骤，无则返回 None
        """
        completed_ids = {
            s.id for s in plan.steps if s.status == "completed"
        }

        for i, step in enumerate(plan.steps):
            # 只选 pending 或 failed（允许重试）
            if step.status not in ("pending", "failed"):
                continue

            # 检查重试上限（> 而非 >=，因为首次失败不是重试，count=1 意味着还有空余）
            if step.status == "failed":
                retries = self._retry_count.get(step.id, 0)
                if retries > self.max_retries_per_step:
                    continue  # 超过重试次数，跳过

            # 依赖检查
            if not step.can_execute(completed_ids):
                # 如果某依赖步骤的状态是 failed/skipped，此步也可能需要跳过
                blocking = [
                    dep for dep in step.depends_on
                    if dep not in completed_ids
                ]
                if any(
                    self._get_step(plan, dep).status in ("failed", "skipped")
                    for dep in blocking
                ):
                    if self.skip_failed_non_critical:
                        step.status = "skipped"
                        logger.warning(f"跳过步骤 {step.id}（依赖 {blocking} 未成功）")
                        continue
                continue  # 依赖未满足，等下一轮

            plan.current_step_index = i
            return step

        return None

    def record_result(
        self,
        plan: TaskPlan,
        result: StepResult,
    ) -> TaskPlan:
        """
        记录步骤执行结果，更新计划状态

        Args:
            plan: 当前计划
            result: 步骤执行结果

        Returns:
            更新后的计划
        """
        step = self._get_step(plan, result.step_id)
        if step is None:
            logger.warning(f"步骤 {result.step_id} 不在计划中，结果被忽略")
            return plan

        # 更新步骤状态
        if result.success:
            step.status = "completed"
        else:
            retries = self._retry_count.get(step.id, 0)
            if retries < self.max_retries_per_step:
                step.status = "failed"  # 允许重试
                self._retry_count[step.id] = retries + 1
                logger.warning(
                    f"步骤 {step.id} 失败 ({retries + 1}/{self.max_retries_per_step} 次重试)"
                )
            else:
                step.status = "failed"  # 超过重试上限
                logger.error(f"步骤 {step.id} 失败，已达最大重试次数")

        # 记录结果
        plan.results.append(result)

        logger.debug(f"步骤结果已记录: {result.summary()}")
        return plan

    # ---- 重规划 ----

    def should_revise(self, plan: TaskPlan) -> bool:
        """
        判断是否需要重规划

        条件:
            1. 有步骤最终失败（超过重试次数）
            2. 该步骤有后续步骤依赖它
            3. 尚未超过最大重规划次数

        Args:
            plan: 当前计划

        Returns:
            是否需要触发重规划
        """
        if plan.revision_count >= self.max_revisions:
            return False

        # 找到最终失败的步骤
        for step in plan.steps:
            if step.status != "failed":
                continue
            retries = self._retry_count.get(step.id, 0)
            if retries < self.max_retries_per_step:
                continue  # 还有重试机会

            # 检查是否有后续步骤依赖它
            dependents = [
                s for s in plan.steps
                if step.id in s.depends_on and s.status == "pending"
            ]
            if dependents:
                logger.info(
                    f"触发重规划: {step.id} 失败，"
                    f"影响 {len(dependents)} 个后续步骤"
                )
                return True

        return False

    def revise(
        self,
        plan: TaskPlan,
        revised_steps: list[TaskStep],
    ) -> TaskPlan:
        """
        用新步骤替换计划中的失败步骤及其后续步骤

        重规划逻辑:
            1. 保留所有已完成的步骤
            2. 找到第一个失败的步骤
            3. 删除该步骤及其所有后续步骤（status=pending/skipped/failed）
            4. 插入 revised_steps

        Args:
            plan: 当前计划
            revised_steps: 新的后续步骤列表

        Returns:
            重规划后的计划
        """
        # 找到第一个失败步骤的索引
        first_failed_idx = None
        for i, step in enumerate(plan.steps):
            if step.status == "failed":
                first_failed_idx = i
                break

        if first_failed_idx is None:
            logger.warning("重规划被调用但没有找到失败步骤")
            return plan

        # 保留失败步骤之前的步骤，移除失败步骤及其后的步骤
        kept_steps = plan.steps[:first_failed_idx]

        # 为 revised_steps 设置初始状态和依赖
        for step in revised_steps:
            step.status = "pending"
            # 清除对已删除步骤的依赖
            valid_deps = [d for d in step.depends_on if self._get_step(plan, d)]
            step.depends_on = valid_deps

        # 构建新步骤列表
        plan.steps = kept_steps + revised_steps
        plan.revision_count += 1
        plan.status = "running"

        logger.info(
            f"计划已重规划: 保留 {len(kept_steps)} 步，"
            f"新增 {len(revised_steps)} 步 (第 {plan.revision_count} 次重规划)"
        )
        return plan

    # ---- 结束 ----

    def is_complete(self, plan: TaskPlan) -> bool:
        """
        判断计划是否已完成（包括成功、失败、取消）

        Returns:
            True 表示无需继续执行了
        """
        # 所有步骤都处理完了
        all_done = all(
            s.status in ("completed", "skipped", "failed")
            for s in plan.steps
        )
        if all_done:
            return True

        # 检查是否有步骤在等待，但无法执行（依赖永远无法满足）
        return self.get_next_step(plan) is None

    def finalize(self, plan: TaskPlan) -> TaskPlan:
        """
        完成计划执行，设置最终状态

        Returns:
            最终状态的计划
        """
        if plan.status in ("completed", "failed", "cancelled"):
            return plan  # 已经结束了

        completed = len(plan.get_completed_steps())
        total = len(plan.steps)
        failed = len(plan.get_failed_steps())

        if completed == total:
            plan.status = "completed"
            logger.info(f"计划执行完成: {completed}/{total} 步全部成功")
        elif failed > 0 and not self.should_revise(plan):
            plan.status = "failed"
            logger.warning(
                f"计划执行失败: {completed}/{total} 完成，{failed} 失败"
            )
        else:
            plan.status = "completed"  # 部分步骤被跳过但仍可达终点
            logger.info(f"计划执行完成（含跳过步骤）: {completed}/{total}")

        return plan

    # ---- 验证 ----

    def validate(self, plan: TaskPlan) -> list[str]:
        """
        验证计划的合法性

        检查项:
            1. 是否有步骤
            2. 步骤 ID 是否唯一
            3. depends_on 引用的步骤是否存在
            4. 是否有循环依赖
            5. 是否有步骤依赖自己

        Returns:
            错误信息列表，空列表表示合法
        """
        errors = []

        if not plan.steps:
            errors.append("计划至少需要一个步骤")
            return errors

        step_ids = {s.id for s in plan.steps}

        # ID 唯一性
        if len(step_ids) != len(plan.steps):
            errors.append("步骤 ID 不唯一（存在重复 ID）")

        for step in plan.steps:
            # 依赖引用检查
            for dep in step.depends_on:
                if dep not in step_ids:
                    errors.append(f"步骤 '{step.id}' 依赖不存在的步骤 '{dep}'")
                if dep == step.id:
                    errors.append(f"步骤 '{step.id}' 不能依赖自己")

        # 循环依赖检测（DFS）
        cycle = self._detect_cycle(plan)
        if cycle:
            errors.append(f"检测到循环依赖: {' → '.join(cycle)}")

        return errors

    # ---- 内部辅助 ----

    def _get_step(self, plan: TaskPlan, step_id: str) -> TaskStep | None:
        """按 ID 获取步骤"""
        for s in plan.steps:
            if s.id == step_id:
                return s
        return None

    def _detect_cycle(self, plan: TaskPlan) -> list[str] | None:
        """
        DFS 检测循环依赖

        Returns:
            循环路径（步骤 ID 列表），无循环则返回 None
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {s.id: WHITE for s in plan.steps}
        parent = {}

        def _dfs(node_id, path):
            color[node_id] = GRAY
            step = self._get_step(plan, node_id)
            if step is None:
                return None
            for dep in step.depends_on:
                if color.get(dep) == GRAY:
                    # 找到循环
                    cycle_start = path.index(dep)
                    return path[cycle_start:] + [dep]
                if color.get(dep) == WHITE:
                    result = _dfs(dep, path + [dep])
                    if result:
                        return result
            color[node_id] = BLACK
            return None

        for step in plan.steps:
            if color[step.id] == WHITE:
                result = _dfs(step.id, [step.id])
                if result:
                    return result

        return None
