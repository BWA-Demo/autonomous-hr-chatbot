"""Thin reference to the LiteLLM package (https://github.com/BerriAI/litellm).

LiteLLM is listed in ``requirements.txt``. The HR backend uses it via
``litellm_integrations`` (see ``hr_agent_backend_local.py``).

Example::

    from litellm_reference import completion

    resp = completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.0,
    )
"""

from __future__ import annotations

from litellm import acompletion, completion

__all__ = ["completion", "acompletion"]
