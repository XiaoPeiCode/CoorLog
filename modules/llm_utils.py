import asyncio
import platform
import json
from modules import llm_chat
import pandas as pd
from tqdm.asyncio import tqdm_asyncio  # 新增导入


def get_json_results_simple(prompts, llm_config):
    """
        Given prompts, return the parsed and unparsed content.
    """
    if not prompts or len(prompts) <= 0:
        return
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    common_params = {
        "model": "qwen-plus",
        "top_p": 0.9,
        "temperature": 0.7,
        "response_format": {"type": "json_object"},
        "presence_penalty": 0.1,
    }

    llm_ouputs = asyncio.run(llm_chat.qwen_chat_async(prompts, llm_config), debug=False)
    llm_ouputs = [json.loads(i) for i in llm_ouputs]
    results = pd.DataFrame(llm_ouputs)
    results['predicted_label'] = results['system_state'].apply(lambda x: int(x == "Abnormal"))

    return results


import asyncio
import platform
import json
import pandas as pd
from openai import RateLimitError


async def process_prompts_in_batches(prompts, llm_config, batch_size=10, delay=1):
    """
    Batch process prompts and add delays between each batch
    """
    results = []
    for i in tqdm_asyncio(range(0, len(prompts), batch_size), desc="Processing batches"):  # 使用 tqdm_asyncio 显示进度条
        batch = prompts[i:i + batch_size]
        try:
            batch_results = await llm_chat.qwen_chat_async(batch, llm_config)
            results.extend(batch_results)
        except RateLimitError:
            print("Rate limit exceeded. Waiting for 10 seconds...")
            await asyncio.sleep(10)  # 等待 10 秒后重试
            batch_results = await llm_chat.qwen_chat_async(batch, llm_config)
            results.extend(batch_results)
        await asyncio.sleep(delay)  # 每批次之间延迟
    return results


def completion_usage_to_dict(completion_usage):
    return {
        "completion_tokens": completion_usage.completion_tokens,
        "prompt_tokens": completion_usage.prompt_tokens,
        "total_tokens": completion_usage.total_tokens,
        "completion_tokens_details": completion_usage.completion_tokens_details,
        # "prompt_tokens_details": {
        #     "audio_tokens": completion_usage.prompt_tokens_details.audio_tokens,
        #     "cached_tokens": completion_usage.prompt_tokens_details.cached_tokens
        # }
    }


def llm_json_results(prompts, llm_config):
    """
    Given prompts, return the content before and after parsing
    """
    if not prompts or len(prompts) <= 0:
        return None

    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    common_params = {
        "model": "qwen-plus",
        "top_p": 0.9,
        "temperature": 0.7,
        "response_format": {"type": "json_object"},
        "presence_penalty": 0.1,
    }

    llm_outputs = asyncio.run(process_prompts_in_batches(prompts, llm_config), debug=False)
    llm_answer = [json.loads(i.choices[0].message.content) for i in llm_outputs]
    llm_usage = [completion_usage_to_dict(i.usage) for i in llm_outputs]

    # results = pd.DataFrame(llm_answer)
    # results['predicted_label'] = results['system_state'].apply(lambda x: int(x == "Abnormal"))

    # # llm_usage_df = pd.DataFrame(llm_usage)

    return llm_answer, llm_usage


def get_json_results(prompts, llm_config):
    """
Given prompts, return the content before and after parsing
    """
    if not prompts or len(prompts) <= 0:
        return None

    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Batch processing prompts

    common_params = {
        "model": "qwen-plus",
        "top_p": 0.9,
        "temperature": 0.7,
        "response_format": {"type": "json_object"},
        "presence_penalty": 0.1,
    }

    llm_outputs = asyncio.run(process_prompts_in_batches(prompts, llm_config), debug=False)
    llm_answer = [json.loads(i.choices[0].message.content) for i in llm_outputs]
    llm_usage = [completion_usage_to_dict(i.usage) for i in llm_outputs]

    results = pd.DataFrame(llm_answer)
    results['predicted_label'] = results['system_state'].apply(lambda x: int(x == "Abnormal"))

    # llm_usage_df = pd.DataFrame(llm_usage)

    return results, llm_usage, llm_answer
