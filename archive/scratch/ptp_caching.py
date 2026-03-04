import time
import os
import datetime
from cachier import cachier

# --- CONFIGURATION ---
# In a real project, these would load from a yaml file or env variables
CONFIG = {
    "cache_enabled": True,           # Master switch
    "cache_dir": "./.experiment_cache", # Where to save the files
    "stale_after_days": 7            # How long until we force a refresh?
}

# --- THE INFRASTRUCTURE ---

def get_research_cache():
    """
    Returns a decorator that either caches to disk OR does nothing,
    depending on CONFIG.
    """
    if CONFIG["cache_enabled"]:
        # Ensure directory exists
        os.makedirs(CONFIG["cache_dir"], exist_ok=True)
        
        # Return the cachier decorator
        # separate_files=True is safer for parallel experiments
        return cachier(
            stale_after=datetime.timedelta(days=CONFIG["stale_after_days"]),
            cache_dir=CONFIG["cache_dir"],
            separate_files=True 
        )
    else:
        # Return a "dummy" decorator that does nothing (Pass-through)
        return lambda func: func

# Initialize our custom decorator
ptp_cache = get_research_cache()

# --- THE SIMULATION ---

@ptp_cache
def extract_parameters(problem_text, model="gpt-4"):
    print(f"  [API CALL] Hit the 'LLM' for: {problem_text[:15]}...")
    time.sleep(2.0) # Simulate a slow API call
    return {"status": "extracted", "value": 42}

def run_pipeline():
    print("\n--- Starting Experiment ---")
    t0 = time.time()
    
    # Call 1
    extract_parameters("Problem A: Tax Calculation")
    
    # Call 2 (Should hit cache if enabled)
    extract_parameters("Problem A: Tax Calculation")
    
    # Call 3 (Different input)
    extract_parameters("Problem B: Fee Calculation")
    
    print(f"--- Total Time: {time.time() - t0:.2f}s ---")

# --- EXECUTION ---
if __name__ == "__main__":
    run_pipeline()