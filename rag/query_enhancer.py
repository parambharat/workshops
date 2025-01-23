from enum import Enum
from typing import Any, List

import instructor
import litellm
import weave
from litellm import acompletion
from litellm.caching.caching import Cache
from pydantic import BaseModel, Field

litellm.cache = Cache(disk_cache_dir="data/cache")


class SearchQueries(BaseModel):
    """A list of search queries to gather relevant information from the web to answer the following question."""

    search_queries: list[str] = Field(
        description="A list distinct search queries to gather relevant information from the web",
        min_length=1,
        max_length=5,
    )


class Labels(str, Enum):
    """Enum representing different intent labels."""

    UNRELATED = "unrelated"
    CODE_TROUBLESHOOTING = "code_troubleshooting"
    INTEGRATIONS = "integrations"
    PRODUCT_FEATURES = "product_features"
    SALES_AND_GTM_RELATED = "sales_and_gtm_related"
    BEST_PRACTICES = "best_practices"
    COURSE_RELATED = "course_related"
    NEEDS_MORE_INFO = "needs_more_info"
    OPINION_REQUEST = "opinion_intent_promptrequest"
    NEFARIOUS_QUERY = "nefarious_query"
    OTHER = "other"


class Intent(BaseModel):
    """An intent with the associated label and a reasoning for predicting the intent."""

    intent: Labels
    reason: str


class IntentPrediction(BaseModel):
    """A list of intents associated with the question"""

    intents: List[Intent] = Field(
        description="A list of intents associated with the question",
        min_length=1,
        max_length=5,
    )


class QueryEnhancer(weave.Model):
    """A class for enhancing user queries using the Cohere API."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.5

    @weave.op
    async def generate_search_queries(self, query: str) -> list[str]:
        """
        Generate search queries using the Cohere API.

        Args:
            query (str): The input query for which to generate search queries.

        Returns:
            List[str]: A list of generated search queries.
        """
        client = instructor.from_litellm(acompletion)
        prompt = open("prompts/search_query.txt", "r").read()
        prompt += f"\n<question>\n{query}\n</question>\n"
        search_queries = await client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "user", "content": prompt},
            ],
            response_model=SearchQueries,
        )
        return search_queries.search_queries

    @weave.op
    async def get_intent_prediction(
        self,
        question: str,
    ) -> list[dict[str, Any]]:
        """
        Get intent prediction for a given question using the Cohere API.

        Args:
            question (str): The question for which to get the intent prediction.
            prompt_file (str, optional): The file path to the prompt JSON. Defaults to "prompts/intent_prompt.json".

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the intent predictions.
        """
        client = instructor.from_litellm(acompletion)
        prompt = open("prompts/intent_prediction.txt", "r").read()
        prompt += f"\n<question>\n{question}\n</question>"
        if "command" in self.model.lower():
            intents = await client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
                response_model=IntentPrediction,
                force_single_step=True,
            )
        else:
            intents = await client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
                response_model=IntentPrediction,
            )

        return intents.model_dump(mode="json")["intents"]

    @weave.op
    async def invoke(self, query: str) -> dict[str, Any]:
        """
        Predict the language, generate search queries, and get intent predictions for a given query.

        Args:
            query (str): The input query to process.

        Returns:
            Dict[str, Any]: A dictionary containing the original query, detected language, generated search queries, and intent predictions.
        """
        search_queries = await self.generate_search_queries(query)
        intents = await self.get_intent_prediction(query)
        return {
            "query": query,
            "search_queries": search_queries,
            "intents": intents,
        }
