import asyncio
from typing import Any

import instructor
import weave
from litellm import acompletion
from pydantic import BaseModel, Field
from weave.scorers import Scorer


class StructuredLLMScorer(Scorer):
    client: instructor.client = instructor.from_litellm(acompletion)
    model: str = "gpt-4o-mini"
    temperature: float = 0.1

    @weave.op
    async def score(self, *, output: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class ResponseCorrectnessScorer(StructuredLLMScorer):
    system_prompt: str = open("prompts/response_correctness.txt").read()

    @weave.op
    async def score(self, input: str, output: str, context: str, **kwargs: Any) -> Any:
        class ResponseCorrectness(BaseModel):
            """An annotation representing the correctness of a response to a question w.r.t a reference"""

            reason: str = Field(
                description="A concise explanation describing the score of the response."
            )
            correct: bool = Field(description="Whether the response is correct")
            score: int = Field(
                description="The score of the response in the likert scale (1-3) where 1 is incorrect, 2 is partially correct, and 3 is correct",
                gt=0,
                lt=4,
            )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": f"## Question:\n{input}\n## Reference Answer:\n{context}\n## Generated Answer:\n{output}",
                },
            ],
            temperature=self.temperature,
            response_model=ResponseCorrectness,
        )
        return response.model_dump(mode="json")


class ResponseRelevanceScorer(StructuredLLMScorer):
    system_prompt: str = open("prompts/response_relevance.txt").read()

    @weave.op
    async def score(
        self,
        input: str | None = None,
        output: str | None = None,
        context: str | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        class ResponseRelevance(BaseModel):
            """An annotation representing the relevance of a response to a question w.r.t a reference"""

            reason: str = Field(
                description="A concise explanation describing the relevance of the response."
            )
            relevant: bool = Field(description="Whether the response is relevant")
            score: int = Field(
                description="The score of the response in the likert scale (1-3) where 1 is irrelevant, 2 is partially relevant, and 3 is highly relevant",
                gt=0,
                lt=4,
            )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": f"## Question:\n{input}\n## Reference Answer:\n{context}\n## Generated Answer:\n{output}",
                },
            ],
            temperature=self.temperature,
            response_model=ResponseRelevance,
        )
        return response.model_dump(mode="json")


class DocumentRelevanceScorer(StructuredLLMScorer):
    system_prompt: str = open("prompts/document_relevance.txt").read()

    @weave.op
    async def score_single(
        self, input: str, output: str, context: str, **kwargs: Any
    ) -> Any:
        class DocumentRelevance(BaseModel):
            """An annotation representing the relevance of a document to a question w.r.t a response"""

            reason: str = Field(
                description="A concise explanation describing the relevance of the document."
            )
            relevant: bool = Field(description="Whether the document is relevant")
            score: int = Field(
                description="The score of the document in the likert scale (1-3) where 1 is irrelevant, 2 is partially relevant, and 3 is highly relevant",
                gt=0,
                lt=4,
            )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": f"## Question:\n{input}\n## Document Content:\n{output}",
                },
            ],
            temperature=self.temperature,
            response_model=DocumentRelevance,
        )
        return response.model_dump(mode="json")

    @weave.op
    async def score(
        self,
        input: str | None = None,
        output: list[str] | None = None,
        context: str | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        relevance_outputs = []
        relevance_scores = []
        tasks = []
        for response in output:
            tasks.append(
                self.score_single(
                    input=input,
                    output=response,
                    context=context,
                    chat_history=chat_history,
                )
            )
        results = await asyncio.gather(*tasks)
        for result in results:
            relevance_outputs.append(result["relevant"])
            relevance_scores.append(result["score"])
        result = {
            "relevance": round(sum(relevance_outputs) / len(relevance_outputs), 4),
            "relevance_score": round(sum(relevance_scores) / len(relevance_scores), 4),
        }

        return result
