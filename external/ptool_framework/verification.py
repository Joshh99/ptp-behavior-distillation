"""
Chain-of-Verification: Verify claims independently, then revise to reduce hallucination.

Paper: Dhuliawala et al., "Chain-of-Verification Reduces Hallucination" (Meta AI, 2023)

Pipeline:
1. Generate response (or accept existing response)
2. Extract claims and generate verification questions
3. Answer each question independently (WITHOUT access to original response)
4. Revise response given verification answers

Key insight: The independent verification step is critical — verification questions
are answered in isolation, preventing the original (potentially hallucinated)
response from biasing the verification. Python orchestrates this isolation.

Example:
    >>> from ptool_framework.verification import ChainOfVerification
    >>> cove = ChainOfVerification()
    >>> result = cove.verify("BMI = 70 * 1.75 = 122.5", "Calculate BMI")
    >>> print(result.was_revised)  # True
    >>> print(result.correction_count)  # 1
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .llm_backend import call_llm, execute_ptool, LLMResponse
from .ptool import PToolSpec
from .traces import WorkflowTrace, TraceStep, StepStatus


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class VerificationQuestion:
    """A question generated to verify a specific claim."""
    question: str
    original_claim: str
    index: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "original_claim": self.original_claim,
            "index": self.index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VerificationQuestion:
        return cls(
            question=data["question"],
            original_claim=data["original_claim"],
            index=data["index"],
        )


@dataclass
class VerificationAnswer:
    """Answer to a verification question, obtained independently."""
    question: str
    answer: str
    supports_claim: bool
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "supports_claim": self.supports_claim,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VerificationAnswer:
        return cls(
            question=data["question"],
            answer=data["answer"],
            supports_claim=data["supports_claim"],
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class VerificationTrace:
    """Complete verification session."""
    original_response: str
    revised_response: str
    questions: List[VerificationQuestion] = field(default_factory=list)
    answers: List[VerificationAnswer] = field(default_factory=list)
    claims_verified: int = 0
    claims_corrected: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_response": self.original_response,
            "revised_response": self.revised_response,
            "questions": [q.to_dict() for q in self.questions],
            "answers": [a.to_dict() for a in self.answers],
            "claims_verified": self.claims_verified,
            "claims_corrected": self.claims_corrected,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VerificationTrace:
        return cls(
            original_response=data["original_response"],
            revised_response=data["revised_response"],
            questions=[VerificationQuestion.from_dict(q) for q in data.get("questions", [])],
            answers=[VerificationAnswer.from_dict(a) for a in data.get("answers", [])],
            claims_verified=data.get("claims_verified", 0),
            claims_corrected=data.get("claims_corrected", 0),
        )

    def to_workflow_trace(self) -> WorkflowTrace:
        """Convert to WorkflowTrace for standard trace storage."""
        trace = WorkflowTrace(goal="Chain-of-Verification")

        # Step 1: Generate verification questions
        q_step = TraceStep(
            ptool_name="cove_generate_questions",
            args={"n_questions": len(self.questions)},
            goal="Generate verification questions",
        )
        q_step.status = StepStatus.COMPLETED
        q_step.result = [q.question for q in self.questions]
        trace.steps.append(q_step)

        # Step 2: Independent verification for each question
        for answer in self.answers:
            v_step = TraceStep(
                ptool_name="cove_verify",
                args={"question": answer.question},
                goal=f"Verify: {answer.question[:80]}",
            )
            v_step.status = StepStatus.COMPLETED
            v_step.result = {
                "answer": answer.answer,
                "supports_claim": answer.supports_claim,
            }
            trace.steps.append(v_step)

        # Step 3: Revise
        r_step = TraceStep(
            ptool_name="cove_revise",
            args={"corrections": self.claims_corrected},
            goal="Revise based on verification",
        )
        r_step.status = StepStatus.COMPLETED
        r_step.result = self.revised_response
        trace.steps.append(r_step)

        return trace


@dataclass
class VerificationResult:
    """Final output from Chain-of-Verification."""
    output: str
    was_revised: bool
    trace: VerificationTrace
    correction_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output": self.output,
            "was_revised": self.was_revised,
            "trace": self.trace.to_dict(),
            "correction_count": self.correction_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VerificationResult:
        return cls(
            output=data["output"],
            was_revised=data["was_revised"],
            trace=VerificationTrace.from_dict(data["trace"]),
            correction_count=data.get("correction_count", 0),
        )


# ============================================================================
# Chain-of-Verification Agent
# ============================================================================

class ChainOfVerification:
    """
    Verify and revise LLM outputs to reduce hallucination.

    Pipeline: Extract claims -> Generate verification questions ->
    Answer independently -> Revise if needed.

    Args:
        model: LLM model for verification
        llm_backend: Custom LLM backend (for testing)
        echo: Whether to print progress
    """

    def __init__(
        self,
        model: str = "deepseek-v3-0324",
        llm_backend: Optional[Callable] = None,
        echo: bool = False,
    ):
        self.model = model
        self.llm_backend = llm_backend
        self.echo = echo

    def _call_llm(self, prompt: str) -> str:
        """Call LLM via backend or default call_llm."""
        if self.llm_backend:
            result = self.llm_backend(prompt, self.model)
            return result.content if isinstance(result, LLMResponse) else result
        return call_llm(prompt, self.model).content

    def verify(self, response: str, query: str) -> VerificationResult:
        """
        Verify an existing response and revise if needed.

        Args:
            response: The response to verify
            query: The original query that produced the response

        Returns:
            VerificationResult with potentially revised output
        """
        # Step 1: Generate verification questions
        if self.echo:
            print("  [CoVe] Generating verification questions...")
        questions = self._generate_questions(response, query)

        if not questions:
            # No claims to verify
            trace = VerificationTrace(
                original_response=response,
                revised_response=response,
            )
            return VerificationResult(
                output=response, was_revised=False,
                trace=trace, correction_count=0,
            )

        # Step 2: Answer each question independently (no access to original response)
        if self.echo:
            print(f"  [CoVe] Verifying {len(questions)} claims independently...")
        answers = []
        for q in questions:
            answer = self._verify_independently(q, query)
            answers.append(answer)

        # Count corrections
        corrections = [a for a in answers if not a.supports_claim]
        correction_count = len(corrections)

        # Step 3: Revise if needed
        if correction_count > 0:
            if self.echo:
                print(f"  [CoVe] {correction_count} correction(s) needed, revising...")
            revised = self._revise(response, query, answers)
            was_revised = True
        else:
            if self.echo:
                print("  [CoVe] All claims verified, no revision needed.")
            revised = response
            was_revised = False

        trace = VerificationTrace(
            original_response=response,
            revised_response=revised,
            questions=questions,
            answers=answers,
            claims_verified=len(questions),
            claims_corrected=correction_count,
        )

        return VerificationResult(
            output=revised,
            was_revised=was_revised,
            trace=trace,
            correction_count=correction_count,
        )

    def generate_and_verify(self, query: str) -> VerificationResult:
        """
        Generate a response to the query, then verify and revise it.

        Args:
            query: The query to answer

        Returns:
            VerificationResult with verified output
        """
        if self.echo:
            print("  [CoVe] Generating initial response...")
        response = self._call_llm(query)
        return self.verify(response.strip(), query)

    def verify_ptool(self, spec: PToolSpec, inputs: Dict[str, Any]) -> VerificationResult:
        """
        Execute a ptool and verify its output.

        Args:
            spec: The ptool specification
            inputs: Dictionary of input arguments

        Returns:
            VerificationResult with verified ptool output
        """
        prompt = spec.format_prompt(**inputs)
        response = self._call_llm(prompt)
        task_description = spec.docstring or spec.name
        return self.verify(response.strip(), task_description)

    def _generate_questions(self, response: str, query: str) -> List[VerificationQuestion]:
        """Generate verification questions from claims in the response."""
        prompt = (
            f"Given this response to a query, extract the key factual claims and "
            f"generate a verification question for each.\n\n"
            f"Query: {query}\n"
            f"Response: {response}\n\n"
            f"Respond with ONLY a JSON array:\n"
            f'[{{"claim": "the claim", "question": "verification question"}}]\n\n'
            f"Only include verifiable factual claims, not opinions."
        )

        raw = self._call_llm(prompt)

        # Parse questions
        try:
            array_match = re.search(r'\[[\s\S]*\]', raw)
            if array_match:
                items = json.loads(array_match.group())
            else:
                items = json.loads(raw)
        except json.JSONDecodeError:
            return []

        questions = []
        for i, item in enumerate(items):
            if isinstance(item, dict):
                questions.append(VerificationQuestion(
                    question=item.get("question", ""),
                    original_claim=item.get("claim", ""),
                    index=i,
                ))

        return questions

    def _verify_independently(
        self, question: VerificationQuestion, query: str
    ) -> VerificationAnswer:
        """
        Answer a verification question independently.

        CRITICAL: The prompt does NOT include the original response.
        This prevents the original (potentially hallucinated) claims
        from biasing the verification.
        """
        prompt = (
            f"Answer the following question factually. "
            f"Context: This relates to the query '{query}'.\n\n"
            f"Question: {question.question}\n\n"
            f"Respond with ONLY a JSON object:\n"
            f'{{"answer": "your factual answer", "supports_original": true/false}}\n\n'
            f"Set supports_original to true if the answer is consistent with: "
            f'"{question.original_claim}"'
        )

        raw = self._call_llm(prompt)

        # Parse answer
        try:
            json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                answer_text = data.get("answer", raw)
                supports = data.get("supports_original", True)
            else:
                answer_text = raw.strip()
                supports = True  # Assume supports if can't parse
        except (json.JSONDecodeError, ValueError):
            answer_text = raw.strip()
            supports = True

        return VerificationAnswer(
            question=question.question,
            answer=answer_text,
            supports_claim=supports,
        )

    def _revise(self, response: str, query: str, answers: List[VerificationAnswer]) -> str:
        """Revise the response given verification answers."""
        corrections = []
        for a in answers:
            if not a.supports_claim:
                corrections.append(f"- Claim was wrong. Correct answer: {a.answer}")

        corrections_text = "\n".join(corrections) if corrections else "No corrections needed."

        prompt = (
            f"Revise the following response based on verification results.\n\n"
            f"Original query: {query}\n"
            f"Original response: {response}\n\n"
            f"Corrections needed:\n{corrections_text}\n\n"
            f"Provide the corrected response. Output ONLY the corrected text."
        )

        return self._call_llm(prompt).strip()


# ============================================================================
# Convenience Functions
# ============================================================================

def verify_and_revise(
    response: str,
    query: str,
    model: str = "deepseek-v3-0324",
    llm_backend: Optional[Callable] = None,
    echo: bool = False,
) -> VerificationResult:
    """
    Quick chain-of-verification on an existing response.

    Args:
        response: The response to verify
        query: The original query
        model: LLM model
        llm_backend: Custom backend (for testing)
        echo: Print progress

    Returns:
        VerificationResult with potentially revised output
    """
    cove = ChainOfVerification(model=model, llm_backend=llm_backend, echo=echo)
    return cove.verify(response, query)
