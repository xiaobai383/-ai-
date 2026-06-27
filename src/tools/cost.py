"""Token counting and cost estimation utilities."""

# Pricing in CNY per 1M tokens (input, output)
# USD→CNY rate ~7.2
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "deepseek-v4-flash": (1.0, 2.0),       # $0.14 / $0.28 per 1M
    "deepseek-v4-pro": (3.1, 6.3),          # $0.435 / $0.87 per 1M
    # Legacy aliases (deprecated 2026/07/24)
    "deepseek-chat": (1.0, 2.0),            # → deepseek-v4-flash non-thinking
    "deepseek-reasoner": (1.0, 2.0),        # → deepseek-v4-flash thinking
    "gpt-4o": (18.0, 54.0),
    "gpt-4o-mini": (0.9, 3.6),
    "qwen-turbo": (2.0, 6.0),
    "qwen-plus": (2.8, 11.2),
}

# Default pricing when model not in table (conservative estimate)
_DEFAULT_PRICING: tuple[float, float] = (1.0, 2.0)


def estimate_tokens(text: str, model: str) -> int:
    """Estimate token count for a text string.

    Uses tiktoken with cl100k_base encoding as a reasonable approximation
    for most modern models. Falls back to character-based estimate if
    tiktoken encoding is unavailable.

    Args:
        text: The input text.
        model: Model name (unused currently, kept for future model-specific logic).

    Returns:
        Estimated number of tokens.
    """
    if not text:
        return 0

    try:
        import tiktoken
        # cl100k_base is the most common modern encoding (GPT-4, DeepSeek, etc.)
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback: rough character-based estimate
        # Chinese chars ~1.5 tokens, English ~0.25 tokens per char
        return max(1, len(text) // 2)


def estimate_cost(
    tokens_in: int, tokens_out: int, model: str
) -> float:
    """Estimate cost in CNY for a given token usage.

    Args:
        tokens_in: Number of input (prompt) tokens.
        tokens_out: Number of output (completion) tokens.
        model: Model name to look up pricing.

    Returns:
        Estimated cost in Chinese Yuan (CNY).
    """
    price_in, price_out = MODEL_PRICING.get(model, _DEFAULT_PRICING)
    cost = (tokens_in / 1_000_000) * price_in + (
        tokens_out / 1_000_000
    ) * price_out
    return cost


def check_limits(
    current_tokens_in: int,
    current_cost_yuan: float,
    limits: dict,
) -> tuple[bool, str]:
    """Check if current usage exceeds configured limits.

    Args:
        current_tokens_in: Cumulative input tokens so far.
        current_cost_yuan: Cumulative cost so far.
        limits: Dict with 'max_tokens_per_request' and 'max_cost_per_request_yuan'.

    Returns:
        Tuple of (blocked: bool, reason: str).
    """
    max_tokens = limits.get("max_tokens_per_request")
    max_cost = limits.get("max_cost_per_request_yuan")

    reasons = []

    if max_tokens is not None and current_tokens_in > max_tokens:
        reasons.append(
            f"Token limit exceeded: {current_tokens_in} > {max_tokens}"
        )

    if max_cost is not None and current_cost_yuan > max_cost:
        reasons.append(
            f"费用超限: {current_cost_yuan:.4f} > {max_cost} 元"
        )

    if reasons:
        return True, "; ".join(reasons)

    return False, ""
