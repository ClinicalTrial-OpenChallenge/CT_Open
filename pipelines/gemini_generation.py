# pip install google-genai
import os, pickle, time, random
from typing import Dict, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types
from tqdm import tqdm

# ---------- One-time client ----------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Please set GEMINI_API_KEY in your environment.")
CLIENT = genai.Client(api_key=GEMINI_API_KEY)

# ---------- Pricing ----------
# Tokens (per-call tiers)
PRICING = {
    "gemini-2.5-pro": {
        "threshold": 200_000,
        "input":  {"lte": 1.25, "gt": 2.50},
        "output": {"lte": 10.00, "gt": 15.00},  # output includes thinking tokens
    }
}
# Grounding with Google Search: $35 / 1000 grounded prompts = $0.035 per call
GROUNDING_COST_PER_CALL_USD = 0.035

def _pick_rates(model: str, input_tokens: int, output_tokens: int):
    tb = PRICING.get(model) or PRICING.get(model.split("@")[0])
    if not tb:
        return {"input_per_mtok": 0.0, "output_per_mtok": 0.0,
                "threshold": 200_000, "input_tier": "unknown", "output_tier": "unknown"}
    thr = tb["threshold"]
    in_rate  = tb["input"]["lte"]  if input_tokens  <= thr else tb["input"]["gt"]
    out_rate = tb["output"]["lte"] if output_tokens <= thr else tb["output"]["gt"]
    return {
        "input_per_mtok": in_rate,
        "output_per_mtok": out_rate,
        "threshold": thr,
        "input_tier":  "<=200k" if input_tokens  <= thr else ">200k",
        "output_tier": "<=200k" if output_tokens <= thr else ">200k",
    }

# ---------- Single call ----------
def generate_once(
    prompt: str,
    *,
    model: str = "gemini-3-pro-preview",
    thinking_budget: int = 60000,
    use_web_tools: bool = True,          # if True, includes google_search -> grounding fee applies
) -> Tuple[str, Dict[str, Any]]:
    tools = []
    if use_web_tools:
        tools = [
            types.Tool(url_context=types.UrlContext()),
            types.Tool(googleSearch=types.GoogleSearch()),
        ]

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        tools=tools,
    )

    resp = CLIENT.models.generate_content(model=model, contents=contents, config=config)
    text = getattr(resp, "text", "") or ""
    usage = getattr(resp, "usage_metadata", None)

    def _get(u, *names, default=0):
        for n in names:
            if hasattr(u, n):
                v = getattr(u, n)
                if v is not None:
                    return v
            if isinstance(u, dict) and n in u:
                return u[n]
        return default

    usage_dict = {}
    if usage is not None:
        input_tokens  = int(_get(usage, "input_tokens", "prompt_token_count", "inputTokenCount", default=0))
        output_tokens = int(_get(usage, "output_tokens", "candidates_token_count", "outputTokenCount", default=0))
        total_tokens  = int(_get(usage, "total_tokens", "total_token_count", "totalTokenCount",
                                 default=input_tokens + output_tokens))
        usage_dict = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}

    meta = {
        "model": getattr(resp, "model_version", model),
        "usage": usage_dict,
        "grounding_used": use_web_tools,   # flag so batch can charge per-call grounding fee
    }
    return text, meta

# ---------- Retry wrapper ----------
def call_with_retries(prompt: str, *, model: str, max_retries: int = 2, use_web_tools: bool = True):
    attempt = 0
    while True:
        try:
            return generate_once(prompt, model=model, use_web_tools=use_web_tools)
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                return f"[ERROR] {type(e).__name__}: {e}", {
                    "model": model,
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "error": str(e),
                    "grounding_used": use_web_tools,
                }
            time.sleep((2 ** (attempt - 1)) + random.uniform(0, 0.25))

# ---------- Batch runner ----------
def run_batch(
    prompts: Dict[str, str],
    *,
    model: str = "gemini-3-pro-preview",
    max_workers: int = 8,
    max_retries: int = 2,
    use_web_tools: bool = True,   # turn off if you don't want to pay the grounding fee
    show_progress: bool = True,   # NEW: show a tqdm progress bar
    progress_desc: str = "Generating",  # NEW: bar label
):
    results: Dict[str, str] = {}
    per_key_usage: Dict[str, Dict[str, Any]] = {}

    totals_in = totals_out = totals_total = 0
    totals_cost = totals_grounding = 0.0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_key = {
            ex.submit(call_with_retries, prompt, model=model, max_retries=max_retries, use_web_tools=use_web_tools): key
            for key, prompt in prompts.items()
        }

        iterator = as_completed(fut_to_key)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(fut_to_key),
                desc=progress_desc,
                unit="item",
                smoothing=0.1,
                miniters=1,
                leave=False,
            )

        for fut in iterator:
            key = fut_to_key[fut]
            text, meta = fut.result()
            results[key] = text

            usage = meta.get("usage", {}) or {}
            in_tok  = int(usage.get("input_tokens", 0) or 0)
            out_tok = int(usage.get("output_tokens", 0) or 0)
            tot_tok = int(usage.get("total_tokens", in_tok + out_tok) or 0)

            rates = _pick_rates(model, in_tok, out_tok)
            token_cost = (in_tok / 1_000_000.0) * rates["input_per_mtok"] + \
                         (out_tok / 1_000_000.0) * rates["output_per_mtok"]

            grounding_used = bool(meta.get("grounding_used"))
            grounding_cost = GROUNDING_COST_PER_CALL_USD if grounding_used else 0.0

            totals_in += in_tok
            totals_out += out_tok
            totals_total += tot_tok
            totals_cost += token_cost + grounding_cost
            totals_grounding += grounding_cost

            per_key_usage[key] = {
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_tokens": tot_tok,
                "token_cost_usd": token_cost,
                "grounding_used": grounding_used,
                "grounding_cost_usd": grounding_cost,
                "total_cost_usd": token_cost + grounding_cost,
                "pricing_applied": {
                    "threshold_tokens": rates["threshold"],
                    "input_tier": rates["input_tier"],
                    "output_tier": rates["output_tier"],
                    "input_usd_per_mtok": rates["input_per_mtok"],
                    "output_usd_per_mtok": rates["output_per_mtok"],
                    "grounding_usd_per_call": GROUNDING_COST_PER_CALL_USD,
                },
                **({"error": meta.get("error")} if meta.get("error") else {}),
            }

    summary = {
        "model": model,
        "per_key_usage": per_key_usage,
        "totals": {
            "input_tokens": totals_in,
            "output_tokens": totals_out,
            "total_tokens": totals_total,
            "token_cost_usd": totals_cost - totals_grounding,
            "grounding_cost_usd": totals_grounding,
            "cost_usd": totals_cost,
        },
        "pricing_policy": {
            "token_tiers": {"threshold_tokens": PRICING.get(model, {}).get("threshold", 200_000),
                            "input_usd_per_mtok": {"<=200k": 1.25, ">200k": 2.50},
                            "output_usd_per_mtok": {"<=200k": 10.00, ">200k": 15.00}},
            "grounding_with_google_search_usd_per_call": GROUNDING_COST_PER_CALL_USD,
        },
    }
    return results, summary


# ---------- CLI (pickle in/out) ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-pickle",  required=True,  help="Path to input pickle: dict {key: prompt}")
    parser.add_argument("--out-pickle", required=True,  help="Path to output pickle: dict {key: output}")
    parser.add_argument("--summary-pickle", required=False, help="(Optional) path to save summary dict")
    parser.add_argument("--max-workers", type=int, default=30)
    parser.add_argument("--model", default="gemini-3-pro-preview")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--no-web-tools", action="store_true", help="Disable google_search grounding to avoid per-call fee")
    args = parser.parse_args()

    with open(args.in_pickle, "rb") as f:
        prompts = pickle.load(f)
    if not isinstance(prompts, dict):
        raise ValueError("Input pickle must be a dict {key: prompt}")

    results, summary = run_batch(
        prompts,
        model=args.model,
        max_workers=args.max_workers,
        max_retries=args.retries,
        use_web_tools=not args.no_web_tools,
    )

    with open(args.out_pickle, "wb") as f:
        pickle.dump(results, f)
    if args.summary_pickle:
        with open(args.summary_pickle, "wb") as f:
            pickle.dump(summary, f)

    print(f"Completed {len(results)} items | "
          f"in={summary['totals']['input_tokens']} out={summary['totals']['output_tokens']} | "
          f"token_cost=${summary['totals']['token_cost_usd']:.4f} + "
          f"grounding=${summary['totals']['grounding_cost_usd']:.4f} "
          f"= total=${summary['totals']['cost_usd']:.4f}")
