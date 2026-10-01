import os
import re
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic

from generation.prompt_builder import PromptMessages

load_dotenv()


@dataclass
class Answer:
    text: str
    citations: List[str]   # extracted citation strings from the answer text
    query: str
    chunks_used: int


def generateUsingOpenAI(prompt: PromptMessages, query: str, chunks_used: int) -> Answer:
    """
    Call the OpenAI API with the prompt messages.
    Parse citation strings from the response (lines matching '[Source: ..., Section: ...]').
    Return an Answer dataclass.
    Model: gpt-4o-mini
    Max tokens: 1024
    Read OPENAI_API_KEY from environment variables using python-dotenv.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ],
    )

    text = response.choices[0].message.content or ""
    citations = re.findall(r'\[Source: .+?, Section: .+?\]', text)

    return Answer(
        text=text,
        citations=citations,
        query=query,
        chunks_used=chunks_used,
    )

def generate(prompt: PromptMessages, query: str, chunks_used: int) -> Answer:
    """
    Call the Anthropic API with the prompt messages.
    Parse citation strings from the response (lines matching '[Source: ..., Section: ...]').
    Return an Answer dataclass.
    Model: claude-haiku-4-5
    Max tokens: 1024
    Read ANTHROPIC_API_KEY from environment variables using python-dotenv.
    """
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=prompt.system,
        messages=[
            {"role": "user", "content": prompt.user},
        ],
    )

    text = "".join(block.text for block in response.content if block.type == "text")
    citations = re.findall(r'\[Source: .+?, Section: .+?\]', text)

    return Answer(
        text=text,
        citations=citations,
        query=query,
        chunks_used=chunks_used,
    )