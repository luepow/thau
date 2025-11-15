"""Chain of Thought reasoning implementation."""

from typing import List, Dict, Optional
from loguru import logger

from core.inference.generator import TextGenerator


class ReasoningEngine:
    """Chain of Thought reasoning engine.

    Implements step-by-step reasoning using prompting techniques.
    """

    def __init__(self, generator: Optional[TextGenerator] = None):
        """Initialize reasoning engine.

        Args:
            generator: TextGenerator instance
        """
        self.generator = generator or TextGenerator()

        self.cot_template = """Let's solve this step by step:

Question: {question}

Step-by-step reasoning:
1."""

        self.reflection_template = """Given this reasoning:
{reasoning}

Let's reflect on this and identify any potential issues or improvements."""

        logger.info("ReasoningEngine initialized")

    def reason(
        self,
        question: str,
        context: Optional[str] = None,
        max_steps: int = 5,
    ) -> Dict[str, any]:
        """Perform step-by-step reasoning.

        Args:
            question: Question to reason about
            context: Optional context
            max_steps: Maximum reasoning steps

        Returns:
            Dict with reasoning steps and final answer
        """
        # Build prompt
        if context:
            prompt = f"Context: {context}\n\n{self.cot_template.format(question=question)}"
        else:
            prompt = self.cot_template.format(question=question)

        # Generate reasoning
        reasoning = self.generator.generate(
            prompt=prompt,
            max_new_tokens=512,
            temperature=0.7,
        )[0]

        # Parse steps
        steps = self._parse_steps(reasoning)

        logger.info(f"Generated {len(steps)} reasoning steps")

        return {
            "question": question,
            "steps": steps,
            "reasoning": reasoning,
            "final_answer": steps[-1] if steps else reasoning,
        }

    def _parse_steps(self, reasoning: str) -> List[str]:
        """Parse reasoning into steps.

        Args:
            reasoning: Reasoning text

        Returns:
            List of steps
        """
        steps = []
        lines = reasoning.split("\n")

        current_step = ""
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                if current_step:
                    steps.append(current_step.strip())
                current_step = line
            elif current_step:
                current_step += " " + line

        if current_step:
            steps.append(current_step.strip())

        return steps

    def reflect(self, reasoning: str) -> str:
        """Reflect on reasoning to identify improvements.

        Args:
            reasoning: Previous reasoning

        Returns:
            Reflection and improvements
        """
        prompt = self.reflection_template.format(reasoning=reasoning)

        reflection = self.generator.generate(
            prompt=prompt,
            max_new_tokens=256,
            temperature=0.8,
        )[0]

        return reflection

    def think_step_by_step(self, problem: str) -> Dict[str, any]:
        """Think through a problem step by step.

        Args:
            problem: Problem to solve

        Returns:
            Reasoning result
        """
        return self.reason(problem)
