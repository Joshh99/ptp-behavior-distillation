from together import Together

api_key = "tgp_v1_ltHhNSf3rT-bfZKM6wJpoDIZxEK9KfByeRuuHNnu2GA"
client = Together(api_key=api_key)

# List available models
models = client.models.list()

print("Available Chat Models (Serverless):")
print("=" * 60)

for model in models:
    # Filter for chat/instruct models that are serverless
    if 'instruct' in model.id.lower() or 'chat' in model.id.lower():
        print(f"\nModel: {model.id}")
        if hasattr(model, 'pricing'):
            print(f"  Pricing: {model.pricing}")