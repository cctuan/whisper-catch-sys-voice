import os
from collections.abc import Iterable

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


def create(messages: Iterable[ChatCompletionMessageParam]) -> str:
    """
    Creates a chat completion using the OpenAI API.

    Args:
        messages (Iterable[ChatCompletionMessageParam]): The messages to be sent to the OpenAI API.

    Returns:
        str: The content of the first completion choice.

    Raises:
        ValueError: If no completion choices are returned or if the completion message content is empty.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model = "gpt-4o-mini"
    temperature = 0.0

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    if not completion.choices:
        raise ValueError("No completion choices returned")

    content = completion.choices[0].message.content
    if not content:
        raise ValueError("No completion message content")

    return content
