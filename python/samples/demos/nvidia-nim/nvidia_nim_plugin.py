# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
from typing import Annotated

from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.kernel import Kernel
from openai import AsyncOpenAI, OpenAI

# 
# Please Replace the url with the real NIM endpoint.
#
nim_url = os.getenv("NIM_BASE_URL", "http://localhost:8000/v1")
nim_api_key = os.getenv("NIM_API_KEY", "<replace_with_NIM_API_key>")

class NLlama3Plugin:
    """A sample plugin that provides response from NIM."""
 
    @kernel_function(name="get_nllama3_opinion", description="Get the opinion of nllama3")
    def get_nllama3_opinion(self, question: Annotated[str, "The input question"]) -> Annotated[str, "The output is a string"]:
       
        prompt = question.replace("nllama3", "you")
        
        client = OpenAI(base_url=nim_url, api_key=nim_api_key or "nim-local")
        messages = [
            {"content": prompt, "role": "user"}
        ]        
        response = client.chat.completions.create(
            model="meta/llama3-8b-instruct",
            messages=messages,
            max_tokens=64,
            stream=False
        )
        completion = response.choices[0].message
        return completion.content or ""


async def main():
    kernel = Kernel()

    use_azure_openai = os.getenv("USE_AZURE_OPENAI", "false").strip().lower() in {"1", "true", "yes"}
    service_id = "function_calling"
    if use_azure_openai:
        # Please make sure your AzureOpenAI Deployment allows for function calling
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not azure_endpoint or not azure_deployment or not azure_api_key:
            raise ValueError(
                "Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_CHAT_DEPLOYMENT_NAME, and AZURE_OPENAI_API_KEY "
                "or set USE_AZURE_OPENAI=false to use OpenAI."
            )
        ai_service = AzureChatCompletion(
            service_id=service_id,
            endpoint=azure_endpoint,
            deployment_name=azure_deployment,
            api_key=azure_api_key,
        )
    else:
        nim_async_client = AsyncOpenAI(base_url=nim_url, api_key=nim_api_key or "nim-local")
        ai_service = OpenAIChatCompletion(
            service_id=service_id,
            ai_model_id=os.getenv("OPENAI_CHAT_MODEL_ID", "meta/llama3-8b-instruct"),
            api_key=nim_api_key or "nim-local",
            async_client=nim_async_client,
        )
    kernel.add_service(ai_service)

    kernel.add_plugin(NLlama3Plugin(), plugin_name="nllama3")

    # Example 1: Use automated function calling with a non-streaming prompt
    print("========== Example 1: Use automated function calling with a non-streaming prompt ==========")
    settings: OpenAIChatPromptExecutionSettings = kernel.get_prompt_execution_settings_from_service_id(
        service_id=service_id
    )
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto(
        auto_invoke=True, filters={"included_plugins": ["nllama3"]}
    )

    print(
        await kernel.invoke_prompt(
            function_name="get_nllama3_opinion",
            plugin_name="nllama3",
            prompt="What does nllama3 knows about NVIDIA H100?",
            settings=settings,
        )
    )

    # Example 2: Use automated function calling with a streaming prompt
    print("========== Example 2: Use automated function calling with a streaming prompt ==========")
    settings: OpenAIChatPromptExecutionSettings = kernel.get_prompt_execution_settings_from_service_id(
        service_id=service_id
    )
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto(
        auto_invoke=True, filters={"included_plugins": ["nllama3"]}
    )

    result = kernel.invoke_prompt_stream(
        function_name="get_nllama3_opinion",
        plugin_name="nllama3",
        prompt="What does nllama3 knows about NVIDIA H100?",
        settings=settings,
    )

    async for message in result:
        print(str(message[0]), end="")
    print("")

    # Example 3: Use manual function calling with a non-streaming prompt
    print("========== Example 3: Use manual function calling with a non-streaming prompt ==========")

    chat: OpenAIChatCompletion | AzureChatCompletion = kernel.get_service(service_id)
    chat_history = ChatHistory()
    settings: OpenAIChatPromptExecutionSettings = kernel.get_prompt_execution_settings_from_service_id(
        service_id=service_id
    )
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto(
        auto_invoke=False, filters={"included_plugins": ["nllama3"]}
    )
    chat_history.add_user_message(
        "What does nllama3 knows about NVIDIA H100?"
    )

    while True:
        # The result is a list of ChatMessageContent objects, grab the first one
        result = await chat.get_chat_message_contents(chat_history=chat_history, settings=settings, kernel=kernel)
        result = result[0]

        if result.content:
            print(result.content)

        if not result.items or not any(isinstance(item, FunctionCallContent) for item in result.items):
            break

        chat_history.add_message(result)
        for item in result.items:
            await kernel.invoke_function_call(
                function_call=item,
                chat_history=chat_history,
                arguments=KernelArguments(),
                function_call_count=1,
                request_index=0,
            )


if __name__ == "__main__":
    asyncio.run(main())
