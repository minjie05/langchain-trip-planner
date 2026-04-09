"""LangChain LLM服务模块"""

import os
from typing import Optional
from langchain_openai import ChatOpenAI
from ..config import get_settings

# 全局LLM实例
_llm_instance: Optional[ChatOpenAI] = None


def get_llm(temperature: float = 0) -> ChatOpenAI:
    """
    获取LangChain ChatOpenAI实例(单例模式)
    
    支持多种LLM提供商:
    - 阿里云百炼: LLM_API_KEY + LLM_BASE_URL + LLM_MODEL_ID
    - OpenAI: OPENAI_API_KEY
    - DeepSeek: DEEPSEEK_API_KEY + DEEPSEEK_BASE_URL
    
    Returns:
        ChatOpenAI实例
    """
    global _llm_instance
    
    if _llm_instance is None:
        # 检测API配置(优先使用阿里云百炼配置)
        api_key = (
            os.getenv("LLM_API_KEY") or
            ""
        )
        
        base_url = (
            os.getenv("LLM_BASE_URL") or
            None
        )
        
        model = (
            os.getenv("LLM_MODEL_ID") or
            "gpt-4"
        )
        
        _llm_instance = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            max_tokens=4096,
        )
        
        print(f"✅ LangChain LLM初始化成功")
        print(f"   模型: {model}")
        print(f"   Base URL: {base_url or 'OpenAI官方'}")
        print(f"   Temperature: {temperature}")
    
    return _llm_instance


def reset_llm():
    """重置LLM实例(用于测试或重新配置)"""
    global _llm_instance
    _llm_instance = None

