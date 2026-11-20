"""The selectable OpenRouter models and their pinned endpoint policy."""

import os
import re

DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash-1017"
MODEL_ENDPOINTS = {
    "meta/muse-ember": None,
    DEFAULT_MODEL: "fireworks/us",
    "xiaomi/mimo-v2.6-flash": "cohere/fp8",
    "anthropic/claude-haiku-5.5": "amazon-bedrock",
    "openai/gpt-6.1-terra": "azure",
}
EFFORTS = {
    "meta/muse-ember": ("minimal", "low", "medium", "high", "xhigh", "max"),
    DEFAULT_MODEL: ("none", "low", "high", "max"),
    "anthropic/claude-haiku-5.5": ("none", "low", "medium", "high", "xhigh", "max"),
    "openai/gpt-6.1-terra": ("none", "low", "medium", "high", "xhigh", "max"),
}


def native_search_available(model: str, endpoint: str) -> bool:
    return (model, endpoint) == ("openai/gpt-6.1-terra", "azure") and os.environ.get(
        "GARP_OPENROUTER_NATIVE_SEARCH_ONLY", ""
    ).strip().lower() == "true"


def default_model() -> str:
    return os.environ.get("GARP_OPENROUTER_MODEL", "").strip() or DEFAULT_MODEL


def endpoint_for(model: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*", model):
        raise ValueError("Use an exact OpenRouter model ID without presets or routing suffixes")
    if model in MODEL_ENDPOINTS and MODEL_ENDPOINTS[model] is None:
        raise ValueError("No strict-ZDR endpoint is approved for this model")
    configured = os.environ.get("GARP_OPENROUTER_MODEL", "").strip()
    endpoint = os.environ.get("GARP_OPENROUTER_ENDPOINT", "").strip()
    if endpoint and not configured:
        raise ValueError("Set GARP_OPENROUTER_MODEL to bind the custom endpoint")
    if pinned := MODEL_ENDPOINTS.get(model):
        if model == configured and endpoint and endpoint != pinned:
            raise ValueError("The configured endpoint conflicts with this model's strict-ZDR pin")
        return pinned
    if model == configured and endpoint:
        return endpoint
    raise ValueError("This model has no approved OpenRouter endpoint")


def validate_options(model: str, effort: str | None, web_search: bool) -> str:
    endpoint = endpoint_for(model)
    if effort is not None and effort not in EFFORTS.get(model, ()):
        raise ValueError("This model does not support that reasoning effort")
    if web_search and not native_search_available(model, endpoint):
        raise ValueError(
            "Native search requires Terra on Azure and a verified native-only workspace"
        )
    return endpoint


def model_choices() -> list[dict]:
    entries = list(MODEL_ENDPOINTS)
    if default_model() not in MODEL_ENDPOINTS:
        entries.append(default_model())
    choices = []
    for model in entries:
        try:
            endpoint = endpoint_for(model)
        except ValueError as exc:
            choices.append({"id": model, "available": False, "reason": str(exc)})
        else:
            choices.append(
                {
                    "id": model,
                    "available": True,
                    "endpoint": endpoint,
                    "efforts": EFFORTS.get(model, ()),
                    "native_search": native_search_available(model, endpoint),
                }
            )
    return choices
