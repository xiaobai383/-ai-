"""Token 计数与成本估算工具。"""
import logging

import httpx

logger = logging.getLogger(__name__)

# 每百万 token 的定价（输入，输出），单位：人民币
# 美元→人民币汇率 ~7.2
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "deepseek-v4-flash": (1.0, 2.0),       # $0.14 / $0.28 per 1M
    "deepseek-v4-pro": (3.1, 6.3),          # $0.435 / $0.87 per 1M
    # 旧版别名（已于 2026/07/24 弃用）
    "deepseek-chat": (1.0, 2.0),            # → deepseek-v4-flash 非思考模式
    "deepseek-reasoner": (1.0, 2.0),        # → deepseek-v4-flash 思考模式
    "gpt-4o": (18.0, 54.0),
    "gpt-4o-mini": (0.9, 3.6),
    "qwen-turbo": (2.0, 6.0),
    "qwen-plus": (2.8, 11.2),
}

# 模型不在价格表中的默认定价（保守估计）
_DEFAULT_PRICING: tuple[float, float] = (1.0, 2.0)


def estimate_tokens(text: str, model: str) -> int:
    """估算文本字符串的 token 数量。

    使用 tiktoken 的 cl100k_base 编码作为大多数现代模型的合理近似。
    如果 tiktoken 编码不可用，则回退到基于字符的估算。

    Args:
        text: 输入文本。
        model: 模型名称（当前未使用，保留用于未来的模型特定逻辑）。

    Returns:
        估算的 token 数量。
    """
    if not text:
        return 0

    try:
        import tiktoken
        # cl100k_base 是最常见的现代编码（GPT-4、DeepSeek 等）
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # 回退方案：基于字符的粗略估算
        # 中文字符约 1.5 个 token，英文字符约 0.25 个 token
        return max(1, len(text) // 2)


def estimate_cost(
    tokens_in: int, tokens_out: int, model: str
) -> float:
    """根据给定的 token 用量估算人民币成本。

    Args:
        tokens_in: 输入（提示词）token 数量。
        tokens_out: 输出（补全）token 数量。
        model: 用于查找定价的模型名称。

    Returns:
        估算的人民币（CNY）成本。
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
    """检查当前用量是否超出配置的限制。

    Args:
        current_tokens_in: 累计输入 token 数。
        current_cost_yuan: 累计人民币成本。
        limits: 包含 'max_tokens_per_request' 和 'max_cost_per_request_yuan' 的字典。

    Returns:
        (是否被阻止: bool, 原因: str) 元组。
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


def fetch_real_balance(
    api_key: str, base_url: str = "https://api.deepseek.com/v1"
) -> float | None:
    """从 DeepSeek API 获取真实账户余额（CNY）。

    调用 GET https://api.deepseek.com/user/balance 接口（注意：不在 /v1 路径下）。
    失败时返回 None，由调用方决定回退策略。
    """
    if not api_key or "sk-fake" in api_key:
        logger.debug("跳过余额查询：api_key 为空或为假 key")
        return None

    # DeepSeek 余额接口不在 /v1 下，需要去掉 base_url 中的 /v1 后缀
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    url = base + "/user/balance"
    try:
        logger.info("正在查询 DeepSeek 余额: %s", url)
        resp = httpx.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5.0,
        )
        logger.info("DeepSeek 余额响应: status=%d, body=%s", resp.status_code, resp.text[:200])
        resp.raise_for_status()
        data = resp.json()
        # DeepSeek 返回格式: {"balance_infos": [{"total_balance": "6.04", ...}]}
        infos = data.get("balance_infos") or []
        if infos:
            return float(infos[0]["total_balance"])
        return None
    except Exception as e:
        logger.warning("DeepSeek 余额查询失败: %s", e)
        return None
