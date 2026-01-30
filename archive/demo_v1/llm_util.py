"""
llm_util.py - LLM interface for secretagent
Supports GitHub Models API (OpenAI-compatible)
"""

import os
from openai import OpenAI

def llm(prompt: str, service: str = None, model: str = None, echo_service: bool = False):
    """
    Call LLM via GitHub Models API (OpenAI-compatible endpoint).
    
    Requires:
        os.environ['OPENAI_API_KEY'] = os.environ.get('GITHUB_TOKEN')
        os.environ['OPENAI_BASE_URL'] = 'https://models.github.ai/inference'
    """
    client = OpenAI()
    
    if echo_service:
        print(f"\n=== LLM Call [{service}:{model}] ===")
        print(f"Prompt length: {len(prompt)} chars")
    
    response = client.chat.completions.create(
        model=model or "gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    result = response.choices[0].message.content
    
    if echo_service:
        print(f"Response length: {len(result)} chars")
        print("=" * 50)
    
    return result