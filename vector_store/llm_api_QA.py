import requests

from .config import DEFAULT_LLM_MODEL, DEFAULT_URL


def reshape_query(query: str, model_name: str = DEFAULT_LLM_MODEL) -> str:
    """使用 LLM 对用户查询进行改写和优化，以提升检索效果。"""
    # 这里可以调用上面的 LLM API 来实现查询改写
    # 目前是一个占位符实现，直接返回原始查询

    headers = {
        "Authorization": "Bearer sk-wjrtizmtakyahakiovtuqynxrvzaafpcbrxddfdlutaglfhj",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEFAULT_LLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": "What opportunities and challenges will the Chinese large model industry face in 2025?",
            }
        ],
    }

    prompt_str = f"你是菜谱RAG检索的输入优化助手，需将用户提问改写为高检索友好型文本，核心目标是提升菜谱知识库的召回率和精准度。\
    改写规则如下：\
    1. 如果输入不是中文，需要先翻译成中文。\
    2. 抽取输入中的关键词，关键词包括但不限于菜名、食材、调料、做法、感受、意图等。\
    直接给我改写后的文本即可。现在将{query}进行改写。并给我结果。"

    new_query_json = (
        requests.post(
            DEFAULT_URL,
            json={
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": prompt_str,
                    }
                ],
            },
            headers=headers,
        )
        .json()
    )

    new_query = new_query_json['choices'][0]['message']['content'].strip()
    return new_query
