import re

from pydantic import BaseModel
from pydantic_ai import Agent 
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key="TOGETHER_API_KEY",
    base_url="https://api.together.xyz/v1",
)

model = OpenAIModel("deepseek-ai/DeepSeek-V3", provider=OpenAIProvider(openai_client=client))

class BaggageFee(BaseModel):
    total_fee_usd: float
    reasoning: str

agent = Agent(
    model=model,
    result_type=BaggageFee,
    system_prompt="You compute airline baggage fees.",
)

result = agent.run_sync("Passenger has 2 checked bags on Delta economy.")
print(result.data)
print(result.reasoning)