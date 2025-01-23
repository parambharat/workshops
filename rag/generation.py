from typing import Any

import litellm
import weave
from litellm import acompletion
from litellm.caching.caching import Cache

litellm.cache = Cache(disk_cache_dir="data/cache")


class ResponseGenerator(weave.Model):
    """Generates responses based on a query and provided documents.

    Attributes:
       model (str): The model to use for generating responses.
       temperature (float): The temperature setting for the model.
       max_tokens (int): The maximum number of tokens for the response.
       system_prompt (str): The system prompt to use for the model.
    """

    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: str

    @weave.op
    async def invoke(
        self, query: str, documents: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Invokes the model to generate a response based on the query and documents.

        Args:
            query (str): The user query to process.
            documents (str): The list of documents to use for generating the response.

        Returns:
            dict: The response from the model in JSON format.
        """
        assert isinstance(documents, list), "Documents must be a list of dictionaries "
        assert all(
            isinstance(doc, dict) for doc in documents
        ), "Documents must be a list of dictionaries."
        if "command" in self.model.lower():
            response = await acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                documents=documents,
            )
        else:
            docs = ("\n\n".join([doc["document"] for doc in documents]),)
            response = await acompletion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt.format(documents=docs),
                    },
                    {"role": "user", "content": query},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        return response.model_dump(mode="json")


class SimpleResponseGenerator(ResponseGenerator):
    system_prompt: str = open("prompts/simple_system.txt", "r").read()


class QueryEnhancedResponseGenerator(ResponseGenerator):
    system_prompt: str = open("prompts/qe_system_prompt.txt", "r").read()

    @weave.op
    async def invoke(
        self, query: str, documents: list[dict[str, Any]], intents: str
    ) -> dict[str, Any]:
        """
        Invokes the model to generate a response based on the query, documents, and intents.

        Args:
            query (str): The user query to process.
            documents (list[dict[str, Any]]): The list of documents to use for generating the response.
            intents (str): The intents to format into the system prompt.

        Returns:
            dict: The response from the model in JSON format.
        """
        if "command" in self.model.lower():
            response = await acompletion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt.format(intents=intents),
                    },
                    {"role": "user", "content": query},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                documents=documents,
            )
        else:
            docs = ("\n\n".join([doc["document"] for doc in documents]),)
            response = await acompletion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt.format(
                            intents=intents, documents=docs
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        return response.model_dump(mode="json")
