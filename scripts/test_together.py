from together import Together
from usage_tracker import UsageTracker

api_key = "tgp_v1_ltHhNSf3rT-bfZKM6wJpoDIZxEK9KfByeRuuHNnu2GA"
client = Together(api_key=api_key)
tracker = UsageTracker()

# Update these prices (per 1M tokens)
PRICE_PER_M_INPUT = 0.88  # Llama-3.1-70B
PRICE_PER_M_OUTPUT = 0.88

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-72B-Instruct-Turbo",  # Note the "Meta-" prefix
    messages=[{"role": "user", "content": "What is 2+2?"}],
    max_tokens=50
)

# Calculate cost
cost = (response.usage.prompt_tokens * PRICE_PER_M_INPUT / 1_000_000 + 
        response.usage.completion_tokens * PRICE_PER_M_OUTPUT / 1_000_000)

# Log it
tracker.log_call(
    model="Qwen-2.5-72B",
    prompt_tokens=response.usage.prompt_tokens,
    completion_tokens=response.usage.completion_tokens,
    cost=cost
)

tracker.report()