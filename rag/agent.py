import json
from typing import Any

import litellm
import weave
from litellm import acompletion
from litellm.caching.caching import Cache
from litellm.types.utils import ModelResponse

from tools import FUNCTION_MAP, TOOL_SCHEMAS

litellm.cache = Cache(disk_cache_dir="data/cache")


class WandbAgent(weave.Model):
    """An agent that processes user queries and generates responses using various tools."""

    # Load the system prompt from a file
    system_prompt: str = open("prompts/agent.txt", "r").read()
    model: str = "gpt-4o"
    temperature: float = 0.1
    tools: list[dict[str, Any]] = TOOL_SCHEMAS

    @weave.op
    async def call_llm(self, messages=None, tools=None) -> ModelResponse:
        """
        Calls the language model with the provided messages and tools.

        Args:
            messages (list, optional): List of messages to send to the model.
            tools (list, optional): List of tools to use with the model.

        Returns:
            ModelResponse: The response from the language model.
        """
        response = await acompletion(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            tools=tools,
        )
        return response

    @weave.op
    async def run_tool_calls(self, messages=None, tool_calls=None):
        """
        Executes tool calls and appends the results to the messages.

        Args:
            messages (list, optional): List of messages to update with tool results.
            tool_calls (list, optional): List of tool calls to execute.

        Returns:
            list: Updated list of messages with tool results.
        """
        for idx, tc in enumerate(tool_calls):
            tool_result = await FUNCTION_MAP[tc.function.name](
                **json.loads(tc.function.arguments)
            ).run()
            tool_result = json.dumps(tool_result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": tool_result,
                }
            )
        return messages

    @weave.op
    async def run_agent(self, query, messages=None):
        """
        Runs the agent to process a query and generate a response.

        Args:
            query (str): The user query to process.
            messages (list, optional): List of messages to send to the model.

        Returns:
            list: Updated list of messages with the agent's response.
        """
        if messages is None:
            messages = []

        if "system" not in {m.get("role") for m in messages}:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.append({"role": "user", "content": query})

        response = await self.call_llm(messages=messages, tools=self.tools)
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            messages.append({"role": "assistant", "tool_calls": tool_calls})
            messages = await self.run_tool_calls(messages, tool_calls)

            response = await self.call_llm(messages=messages, tools=self.tools)
            response_message = response.choices[0].message

        messages.append({"role": "assistant", "content": response_message.content})

        return messages

    @weave.op
    async def invoke(self, query: str) -> dict[str, str]:
        """
        Invokes the agent to process a query and return the final response.

        Args:
            query (str): The user query to process.

        Returns:
            dict: A dictionary containing the final response.
        """
        results = await self.run_agent(query)
        return {"answer": results[-1]["content"]}
