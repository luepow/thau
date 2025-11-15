"""Planning and task decomposition."""

from typing import List, Dict
from loguru import logger


class Planner:
    """Task planner for breaking down complex tasks."""

    def __init__(self, reasoning_engine=None):
        """Initialize planner.

        Args:
            reasoning_engine: ReasoningEngine instance
        """
        from reasoning.chain_of_thought import ReasoningEngine
        self.reasoning_engine = reasoning_engine or ReasoningEngine()

        logger.info("Planner initialized")

    def create_plan(self, goal: str, constraints: List[str] = None) -> Dict:
        """Create a plan to achieve a goal.

        Args:
            goal: Goal to achieve
            constraints: Optional constraints

        Returns:
            Plan with steps
        """
        prompt = f"Goal: {goal}\n\n"

        if constraints:
            prompt += "Constraints:\n"
            for i, constraint in enumerate(constraints, 1):
                prompt += f"{i}. {constraint}\n"
            prompt += "\n"

        prompt += "Create a step-by-step plan to achieve this goal:"

        result = self.reasoning_engine.reason(prompt)

        return {
            "goal": goal,
            "constraints": constraints or [],
            "steps": result["steps"],
            "plan": result["reasoning"],
        }
