import json
from datetime import datetime

class UsageTracker:
    def __init__(self, filename="usage_log.json"):
        self.filename = filename
        self.load()
    
    def load(self):
        try:
            with open(self.filename, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = {"calls": [], "total_cost": 0.0}
    
    def log_call(self, model, prompt_tokens, completion_tokens, cost):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost
        }
        # Change this line:
        self.data["calls"].insert(0, entry)  # Insert at beginning instead of append
        self.data["total_cost"] += cost
        self.save()
    
    def save(self):
        ordered_data = {
            "total_cost": self.data["total_cost"],
            "calls": self.data["calls"]
        }
        with open(self.filename, "w") as f:  # Changed from self.budget_file
            json.dump(ordered_data, f, indent=2)
    
    def report(self):
        print(f"Total calls: {len(self.data['calls'])}")
        print(f"Total cost: ${self.data['total_cost']:.4f}")
        print(f"Budget remaining: ${200 - self.data['total_cost']:.2f}")