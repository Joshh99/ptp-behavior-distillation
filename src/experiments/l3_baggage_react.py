"""
L3 Baggage Allowance - ReAct/Agentic Pattern (Together.ai)

Autonomous agent approach - flexible but unreliable.
Target reliability: 60-75%
Use for trace collection and distillation research.
"""

import sys
import os
import re
import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.rule_arena_loader import RuleArenaDataset, load_baggage_rules
from experiments.config import call_llm, DEFAULT_MODEL


@dataclass
class ReActTrace:
    query: str
    steps: List[Dict] = field(default_factory=list)
    final_answer: Optional[Any] = None
    success: bool = False
    total_time: float = 0.0
    llm_calls: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "steps": self.steps,
            "final_answer": self.final_answer,
            "success": self.success,
            "total_time": self.total_time,
            "llm_calls": self.llm_calls,
        }


class BaggageTools:
    def __init__(self):
        self.rules = load_baggage_rules()
    
    def lookup_rule(self, airline: str, travel_class: str, route_type: str) -> Dict:
        for rule in self.rules:
            if (rule.conditions.get("airline", "").lower() == airline.lower() and
                rule.conditions.get("class", "").lower() == travel_class.lower() and
                rule.conditions.get("route", "").lower() == route_type.lower()):
                return {"fees": rule.fees, "thresholds": rule.thresholds}
        return {"error": "No rule found"}
    
    def calculate_overweight(self, weight_kg: float) -> Dict:
        if weight_kg <= 23:
            return {"fee": 0}
        elif weight_kg <= 32:
            return {"fee": 100}
        elif weight_kg <= 45:
            return {"fee": 200}
        return {"fee": 400}


class ReActAgent:
    def __init__(self, model: str = DEFAULT_MODEL, max_steps: int = 8, verbose: bool = True):
        self.model = model
        self.max_steps = max_steps
        self.verbose = verbose
        self.tools = BaggageTools()
    
    def run(self, query: str, context: Dict = None) -> ReActTrace:
        trace = ReActTrace(query=query)
        start = time.time()
        
        if self.verbose:
            print(f"[L3 ReAct] {query[:60]}...")
        
        # Simple prompt for ReAct
        prompt = f'''Answer this baggage query step by step.
Query: {query}
Context: {context}

Think through the problem and provide a numeric answer.
Format your final answer as: ANSWER: [number]'''
        
        try:
            response = call_llm(prompt, model=self.model, max_tokens=512)
            trace.llm_calls += 1
            
            # Extract answer
            match = re.search(r'ANSWER:\s*(\d+)', response)
            if match:
                trace.final_answer = int(match.group(1))
                trace.success = True
            else:
                # Try to find any number
                numbers = re.findall(r'\True(\d+)', response)
                if numbers:
                    trace.final_answer = int(numbers[-1])
                    trace.success = True
            
            trace.steps.append({"response": response[:200]})
            
        except Exception as e:
            if self.verbose:
                print(f"  Error: {e}")
            trace.steps.append({"error": str(e)})
        
        trace.total_time = time.time() - start
        return trace


def baggage_allowance_l3(query: str, context: Dict = None, model: str = DEFAULT_MODEL, verbose: bool = True) -> Dict:
    agent = ReActAgent(model=model, verbose=verbose)
    trace = agent.run(query, context)
    return {
        "answer": trace.final_answer,
        "success": trace.success,
        "trace": trace.to_dict(),
        "metrics": {"total_time": trace.total_time, "llm_calls": trace.llm_calls}
    }


if __name__ == "__main__":
    print("L3 Baggage - ReAct Agent (Together.ai)")
    dataset = RuleArenaDataset()
    instance = dataset.load("test")[0]
    print(f"Query: {instance.query}")
    result = baggage_allowance_l3(instance.query, instance.passenger_context, model="llama-8b")
    print(f"Result: {result['answer']}")
