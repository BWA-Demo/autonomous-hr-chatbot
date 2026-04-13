"""Minimal runnable example using LiteLLM directly.

Install dependencies::

    pip install -r requirements.txt

Set your OpenAI key, then run::

    export OPENAI_API_KEY="sk-..."
    python litellm_example.py
"""

from __future__ import annotations

import os

import litellm


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "Set OPENAI_API_KEY to call the OpenAI-backed model through LiteLLM."
        )

    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Reply with exactly: OK"},
        ],
        temperature=0.0,
    )
    text = response.choices[0].message.content
    print(text)


if __name__ == "__main__":
    main()
