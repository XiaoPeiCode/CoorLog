
import os
import asyncio
from openai import AsyncOpenAI
import platform
import os
import asyncio
from openai import AsyncOpenAI
import platform

QWEN_API_KEY=  
# QWEN_API_KEY =  


qwen_client = AsyncOpenAI(
    api_key= QWEN_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def get_response():
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"), # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务的base_url
    )
    completion = qwen_client.chat.completions.create(
        model="qwen-turbo",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': '你是谁？'}]
        )
    print(completion.model_dump_json())





async def qwen_chat_async_task(messages,common_params):
    # print(f"Sending question: {messages}")
    response = await qwen_client.chat.completions.create(
        messages=messages,
        **common_params,
    )
    # print(f"Received answer: {response.choices[0].message.content}")
    return response


async def qwen_chat_async(messages,common_params):
    tasks = [qwen_chat_async_task(message,common_params) for message in messages]
    results = await asyncio.gather(*tasks)
    return results

if __name__ == '__main__':
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    common_params = {
    "model" : "qwen-plus",
    "top_p": 0.9,
    "temperature": 0.7,
    "response_format": {"type": "json_object"},
    "presence_penalty": 0.1,
    }
    messages=[
        [{"role": "user", "content": '说json格式的 {\"result\":1}'}],
        # [{"role": "user", "content": '说2'}],
    ]
    asyncio.run(qwen_chat_async_task(messages[0],common_params))
    results = asyncio.run(qwen_chat_async(messages,common_params), debug=False)
    print(results.choices[0].message.content)
