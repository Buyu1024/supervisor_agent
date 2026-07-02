"""规划模块 —— 数据类型定义

TaskStep, StepResult, TaskPlan 构成规划模块的核心数据结构。
所有类型都是纯数据容器，不包含业务逻辑。
"""

import time
from dataclasses import dataclass, field


@dataclass
class TaskStep:
    """任务计划中的一个步骤

    注意与 LLM Function Calling 中的 "tool_call" 区分:
        - TaskStep 是计划层面的步骤定义（"应该做什么"）
        - Function Calling tool_call 是 LLM 执行层面的工具调用（"具体怎么做"）
        一个 TaskStep 可能触发零次或多次 tool_call。
    """

    id: str                                    # 步骤唯一 ID（如 "step_1"）
    description: str                           # 给用户看的人类可读描述
    instruction: str                           # 给 LLM 的执行指令（会作为 prompt 发送）
    action: str                                # "think" | "tool_call" | "respond" | "ask_user"
    depends_on: list[str] = field(default_factory=list)  # 前置依赖步骤 ID 列表
    status: str = "pending"                    # pending | running | completed | failed | skipped

    def can_execute(self, completed_step_ids: set[str]) -> bool:
        """检查此步骤是否满足执行条件（所有依赖步骤已完成）"""
        return all(dep in completed_step_ids for dep in self.depends_on)

    def __repr__(self) -> str:
        deps = f" depends_on={self.depends_on}" if self.depends_on else ""
        return f"TaskStep(id={self.id}, action={self.action}, status={self.status}{deps})"


@dataclass
class StepResult:
    """单步执行结果"""

    step_id: str                               # 对应的步骤 ID
    success: bool                              # 是否成功
    output: str                                # 步骤输出文本（供后续步骤参考）
    error: str | None = None                   # 失败原因
    elapsed_ms: float = 0.0                    # 执行耗时
    metadata: dict = field(default_factory=dict)  # 扩展字段

    def summary(self) -> str:
        """单行摘要，用于日志和进度展示"""
        status = "✅" if self.success else "❌"
        preview = self.output[:80].replace("\n", " ") if self.output else ""
        return f"{status} [{self.step_id}] {preview}{'...' if len(self.output) > 80 else ''}"


@dataclass
class TaskPlan:
    """完整的任务执行计划

    生命周期:
        pending  → 计划已生成，尚未开始执行
        running  → 正在逐步执行中
        revising → 某步失败，正在重规划后续步骤
        completed → 所有步骤执行成功
        failed   → 无法继续执行（重规划失败或关键步骤失败）
        cancelled → 用户取消了执行
    """

    goal: str                                  # 任务目标（给用户看的）
    steps: list[TaskStep] = field(default_factory=list)
    status: str = "pending"                    # pending | running | completing | completed | failed | cancelled
    current_step_index: int = 0                # 当前正在执行/即将执行的步骤索引
    results: list[StepResult] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    revision_count: int = 0                    # 重规划次数

    # ---- 状态查询 ----

    def get_remaining_steps(self) -> list[TaskStep]:
        """获取尚未完成的步骤（按索引顺序）"""
        return [s for s in self.steps if s.status not in ("completed", "skipped")]

    def get_completed_steps(self) -> list[TaskStep]:
        """获取已完成的步骤"""
        return [s for s in self.steps if s.status == "completed"]

    def get_failed_steps(self) -> list[TaskStep]:
        """获取失败的步骤"""
        return [s for s in self.steps if s.status == "failed"]

    def get_result(self, step_id: str) -> StepResult | None:
        """获取指定步骤的执行结果"""
        for r in self.results:
            if r.step_id == step_id:
                return r
        return None

    def progress_summary(self) -> str:
        """人类可读的进度摘要"""
        total = len(self.steps)
        completed = sum(1 for s in self.steps if s.status == "completed")
        failed = sum(1 for s in self.steps if s.status == "failed")
        pending = sum(1 for s in self.steps if s.status == "pending")
        running = sum(1 for s in self.steps if s.status == "running")
        skipped = sum(1 for s in self.steps if s.status == "skipped")

        bar_len = 10
        done_bars = int(bar_len * completed / total) if total else 0
        bar = "█" * done_bars + "░" * (bar_len - done_bars)

        return (
            f"[{bar}] {completed}/{total} 完成"
            + (f", {running} 进行中" if running else "")
            + (f", {failed} 失败" if failed else "")
            + (f", {skipped} 跳过" if skipped else "")
            + (f", {pending} 待执行" if pending else "")
        )

    # ---- 序列化 ----

    def to_dict(self) -> dict:
        """序列化为 dict —— 用于持久化和跨会话恢复"""
        return {
            "goal": self.goal,
            "steps": [
                {
                    "id": s.id,
                    "description": s.description,
                    "instruction": s.instruction,
                    "action": s.action,
                    "depends_on": s.depends_on,
                    "status": s.status,
                }
                for s in self.steps
            ],
            "status": self.status,
            "current_step_index": self.current_step_index,
            "results": [
                {
                    "step_id": r.step_id,
                    "success": r.success,
                    "output": r.output,
                    "error": r.error,
                    "elapsed_ms": r.elapsed_ms,
                    "metadata": r.metadata,
                }
                for r in self.results
            ],
            "created_at": self.created_at,
            "revision_count": self.revision_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskPlan":
        """从 dict 反序列化 —— 用于跨会话恢复"""
        plan = cls(
            goal=data["goal"],
            steps=[
                TaskStep(
                    id=s["id"],
                    description=s["description"],
                    instruction=s["instruction"],
                    action=s["action"],
                    depends_on=s.get("depends_on", []),
                    status=s.get("status", "pending"),
                )
                for s in data["steps"]
            ],
            status=data.get("status", "pending"),
            current_step_index=data.get("current_step_index", 0),
            results=[
                StepResult(
                    step_id=r["step_id"],
                    success=r["success"],
                    output=r["output"],
                    error=r.get("error"),
                    elapsed_ms=r.get("elapsed_ms", 0.0),
                    metadata=r.get("metadata", {}),
                )
                for r in data.get("results", [])
            ],
            created_at=data.get("created_at", time.time()),
            revision_count=data.get("revision_count", 0),
        )
        return plan

    def __repr__(self) -> str:
        return f"TaskPlan(goal={self.goal[:50]}..., steps={len(self.steps)}, status={self.status})"
