from openai import OpenAI
import time
import traceback
import argparse
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import concurrent.futures
import os
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

os.environ["DEBUG"] = "False"
DEBUG = False

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set")

openai_client = OpenAI(
    api_key=openai_api_key,
    base_url="https://api.openai.com/v1",
)


def count_web_search_calls(response) -> int:
    count = 0
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "web_search_call":
            count += 1
    return count


def get_model_pricing(model: str):
    if "gpt-4o-mini" in model:
        return 0.15, 0.6
    elif "gpt-4o" in model:
        return 2.5, 10
    elif "gpt-4.1-mini-2025-04-14" in model:
        return 0.4, 1.6
    elif "o3-mini-2025-01-31" in model or "o4-mini-2025-04-16" in model:
        return 1.1, 4.4
    elif "o3-2025-04-16" in model:
        return 2, 8
    elif "gpt-5-2025-08-07" in model:
        return 1.25, 10
    elif "o1-2024-12-17" in model:
        return 15, 60
    elif "gpt-5.2-2025-12-11" in model:
        return 1.75, 14
    elif "gpt-5-mini-2025-08-07" in model:
        return 0.25, 2
    elif "gpt-5.4-2026-03-05" in model:
        return 2.5, 15
    else:
        raise ValueError(f"Unknown model for pricing: {model}")


WEB_SEARCH_COST_PER_CALL = 10.0 / 1000.0  # $10 per 1000 calls = $0.01 per call


def use_o3_mini(prompt, model, max_completion_tokens, reasoning, timeout=240):
    start_time = time.time()

    def openai_call():
        return openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning,
            stream=False,
        )

    with ThreadPoolExecutor(max_workers=1) as local_executor:
        future = local_executor.submit(openai_call)
        try:
            response = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print("Timeout!")
            return "TIMEOUT_ERROR", {
                "input_tokens": 0,
                "reasoning_tokens": 0,
                "output_tokens": 0,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "web_search_calls": 0,
                "web_search_cost": 0.0,
                "total_cost": 0.0,
                "time_secs": int(time.time() - start_time),
            }
        except Exception as e:
            print(f"Error processing prompt: {e}")
            traceback.print_exc()
            return None, {
                "input_tokens": 0,
                "reasoning_tokens": 0,
                "output_tokens": 0,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "web_search_calls": 0,
                "web_search_cost": 0.0,
                "total_cost": 0.0,
                "time_secs": int(time.time() - start_time),
            }

    input_cost_per_m, output_cost_per_m = get_model_pricing(model)
    content = response.choices[0].message.content
    reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens or 0

    metadata = {}
    metadata["input_tokens"] = response.usage.prompt_tokens
    metadata["reasoning_tokens"] = reasoning_tokens
    metadata["output_tokens"] = response.usage.completion_tokens - reasoning_tokens
    metadata["input_cost"] = input_cost_per_m / 1e6 * metadata["input_tokens"]
    metadata["output_cost"] = output_cost_per_m / 1e6 * response.usage.completion_tokens
    metadata["web_search_calls"] = 0
    metadata["web_search_cost"] = 0.0
    metadata["total_cost"] = metadata["input_cost"] + metadata["output_cost"]
    metadata["time_secs"] = int(time.time() - start_time)

    return content, metadata


def use_4o(prompt, model, max_completion_tokens, reasoning, timeout=240):
    start_time = time.time()
    input_cost_per_m, output_cost_per_m = get_model_pricing(model)

    def openai_call():
        return openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=max_completion_tokens,
            temperature=0,
            top_p=1,
            store=True,
            stream=False,
        )

    with ThreadPoolExecutor(max_workers=1) as local_executor:
        future = local_executor.submit(openai_call)
        try:
            response = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"Timeout for prompt: {prompt[:50]}...")
            return "TIMEOUT_ERROR", {
                "input_tokens": 0,
                "reasoning_tokens": 0,
                "output_tokens": 0,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "web_search_calls": 0,
                "web_search_cost": 0.0,
                "total_cost": 0.0,
                "time_secs": int(time.time() - start_time),
            }
        except Exception as e:
            print(f"Error processing prompt: {e}")
            traceback.print_exc()
            return None, {
                "input_tokens": 0,
                "reasoning_tokens": 0,
                "output_tokens": 0,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "web_search_calls": 0,
                "web_search_cost": 0.0,
                "total_cost": 0.0,
                "time_secs": int(time.time() - start_time),
            }

    content = response.choices[0].message.content
    reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens or 0

    metadata = {}
    metadata["input_tokens"] = response.usage.prompt_tokens
    metadata["reasoning_tokens"] = reasoning_tokens
    metadata["output_tokens"] = response.usage.completion_tokens - reasoning_tokens
    metadata["input_cost"] = input_cost_per_m / 1e6 * metadata["input_tokens"]
    metadata["output_cost"] = output_cost_per_m / 1e6 * response.usage.completion_tokens
    metadata["web_search_calls"] = 0
    metadata["web_search_cost"] = 0.0
    metadata["total_cost"] = metadata["input_cost"] + metadata["output_cost"]
    metadata["time_secs"] = int(time.time() - start_time)

    return content, metadata


def use_gpt5(prompt, model, max_output_tokens, reasoning, verbosity="low", timeout=240, web_search=False):
    """
    model currently supports:
    ['gpt-5-2025-08-07', 'gpt-5.2-2025-12-11', 'gpt-5-mini-2025-08-07', 'gpt-5.4-2026-03-05']
    reasoning currently supports:
    ['minimal', 'low', 'medium', 'high']
    """
    start_time = time.time()
    input_cost_per_m, output_cost_per_m = get_model_pricing(model)

    if web_search:
        tools = [{"type": "web_search"}]
    else:
        tools = []

    def openai_call():
        return openai_client.responses.create(
            model=model,
            tools=tools,
            input=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            reasoning={
                "effort": reasoning,
            },
            text={
                "verbosity": verbosity,
            },
            max_output_tokens=max_output_tokens,
            stream=False,
        )

    with ThreadPoolExecutor(max_workers=1) as local_executor:
        future = local_executor.submit(openai_call)
        try:
            response = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print("Timeout!")
            return "TIMEOUT_ERROR", {
                "input_tokens": 0,
                "reasoning_tokens": 0,
                "output_tokens": 0,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "web_search_calls": 0,
                "web_search_cost": 0.0,
                "total_cost": 0.0,
                "time_secs": int(time.time() - start_time),
            }
        except Exception as e:
            print(f"Error processing prompt: {e}")
            traceback.print_exc()
            return None, {
                "input_tokens": 0,
                "reasoning_tokens": 0,
                "output_tokens": 0,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "web_search_calls": 0,
                "web_search_cost": 0.0,
                "total_cost": 0.0,
                "time_secs": int(time.time() - start_time),
            }

    content = response.output_text
    reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens or 0
    web_search_calls = count_web_search_calls(response)

    metadata = {}
    metadata["input_tokens"] = response.usage.input_tokens
    metadata["reasoning_tokens"] = reasoning_tokens
    metadata["output_tokens"] = response.usage.output_tokens - reasoning_tokens
    metadata["input_cost"] = input_cost_per_m / 1e6 * metadata["input_tokens"]
    metadata["output_cost"] = output_cost_per_m / 1e6 * response.usage.output_tokens
    metadata["web_search_calls"] = web_search_calls
    metadata["web_search_cost"] = WEB_SEARCH_COST_PER_CALL * web_search_calls
    metadata["total_cost"] = (
        metadata["input_cost"]
        + metadata["output_cost"]
        + metadata["web_search_cost"]
    )
    metadata["time_secs"] = int(time.time() - start_time)

    return content, metadata


def generate(prompt_dict, model, max_completion_tokens, reasoning="", verbosity="", timeout=240, web_search=False):
    output_dict = {}
    input_tokens, reasoning_tokens, output_tokens = 0, 0, 0
    total_input_cost, total_output_cost, total_web_search_cost, total_cost = 0.0, 0.0, 0.0, 0.0
    total_web_search_calls = 0
    timed_out_prompts = {}

    if model in ['gpt-4o-2024-08-06', 'gpt-4o-2024-11-20', "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14"]:
        genfunc = use_4o
    elif model in ['o3-mini-2025-01-31', 'o3-2025-04-16', 'o4-mini-2025-04-16', 'o1-2024-12-17']:
        assert reasoning != ""
        assert reasoning in ['medium', 'high', 'low']
        genfunc = use_o3_mini
    elif model in ['gpt-5-2025-08-07', 'gpt-5.2-2025-12-11', 'gpt-5-mini-2025-08-07', 'gpt-5.4-2026-03-05']:
        print(f'Going to use {model}')
        assert reasoning != ""
        assert verbosity != ""
        assert reasoning in ['minimal', 'low', 'medium', 'high']
        assert verbosity in ['low', 'medium', 'high']
        genfunc = use_gpt5
    else:
        raise ValueError(f"Unknown model: {model}")

    with ThreadPoolExecutor(max_workers=200) as executor:
        if model not in ['gpt-5-2025-08-07', 'gpt-5.2-2025-12-11', 'gpt-5-mini-2025-08-07', 'gpt-5.4-2026-03-05']:
            future_to_key = {
                executor.submit(genfunc, prompt, model, max_completion_tokens, reasoning, timeout): key
                for key, prompt in prompt_dict.items()
            }
        else:
            future_to_key = {
                executor.submit(genfunc, prompt, model, max_completion_tokens, reasoning, verbosity, timeout, web_search): key
                for key, prompt in prompt_dict.items()
            }

        for future in tqdm(as_completed(future_to_key), total=len(future_to_key), desc="Processing Prompts"):
            key = future_to_key[future]
            try:
                result, metadata = future.result()

                if result == "TIMEOUT_ERROR":
                    timed_out_prompts[key] = prompt_dict[key]
                else:
                    output_dict[key] = result

                input_tokens += metadata["input_tokens"]
                reasoning_tokens += metadata["reasoning_tokens"]
                output_tokens += metadata["output_tokens"]

                total_input_cost += metadata["input_cost"]
                total_output_cost += metadata["output_cost"]
                total_web_search_cost += metadata["web_search_cost"]
                total_web_search_calls += metadata["web_search_calls"]
                total_cost += metadata["total_cost"]

            except Exception as exc:
                print(f"Prompt with key {key} generated an exception: {exc}")
                output_dict[key] = None

    print('=' * 30)
    print('Generation Summary')
    print('=' * 30)
    print(f"Input tokens(M): {input_tokens / 1e6}")
    print(f"Reasoning tokens(M): {reasoning_tokens / 1e6}")
    print(f"Visible output tokens(M): {output_tokens / 1e6}")
    print(f"Web search calls: {total_web_search_calls}")
    print('-' * 30)
    print(f"Input cost: {total_input_cost}")
    print(f"Output cost: {total_output_cost}")
    print(f"Web search cost: {total_web_search_cost}")
    print('-' * 30)
    print(f"Total cost: {total_cost}")
    print("Time out dict length: ", len(timed_out_prompts))
    print('=' * 30)

    return output_dict, timed_out_prompts, total_cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pickle', type=str, required=True, help='Path to input pickle file containing prompts')
    parser.add_argument('--output_pickle', type=str, required=True, help='Path to output pickle file to save results')
    parser.add_argument('--model', type=str, required=True, help='Model to use for generation')
    parser.add_argument('--max_completion_tokens', type=int, default=10000, help='Maximum completion tokens')
    parser.add_argument('--reasoning', type=str, default='medium', help='Reasoning effort level')
    parser.add_argument('--verbosity', type=str, default='low', help='Verbosity level (for GPT-5)')
    parser.add_argument('--timeout', type=int, default=360, help='Timeout in seconds for each prompt')
    parser.add_argument('--web_search', type=bool, default=False, help='Whether to use web search tool (for GPT-5)')
    args = parser.parse_args()

    with open(args.input_pickle, 'rb') as f:
        prompt_dict = pickle.load(f)

    output_dict, timed_out_prompts, total_cost = generate(
        prompt_dict=prompt_dict,
        model=args.model,
        max_completion_tokens=args.max_completion_tokens,
        reasoning=args.reasoning,
        verbosity=args.verbosity,
        timeout=args.timeout,
        web_search=args.web_search
    )

    with open(args.output_pickle, 'wb') as f:
        pickle.dump(output_dict, f)

    if timed_out_prompts:
        timeout_pickle = args.output_pickle.replace('.pickle', '_timeouts.pickle')
        with open(timeout_pickle, 'wb') as f:
            pickle.dump(timed_out_prompts, f)
        print(f"Timed out prompts saved to {timeout_pickle}")