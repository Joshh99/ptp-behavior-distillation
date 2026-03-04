import functools
import time

def my_cache(func):
    store = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs)))

        if key in store:
            print(f"  [CACHE HIT] Returns instantly.")
            return store[key]
        
        print(f"  [CACHE MISS] Calling expensive API...")
        result = func(*args, **kwargs)
        store[key] = result
        return result
    
    return wrapper 

# --- THE SIMULATION ---

# 1. The "Expensive" LLM Step
# We decorate ONLY this one.
@my_cache
def extract_parameters(problem_text, model="gpt-4"):
    # Simulate API latency
    time.sleep(2.0) 
    # Simulate parsing an answer
    return {"amount": 50, "days": 3, "problem": problem_text[:5]}

# Because it's fast (0.0s) and we might change the formula while debugging.
def compute_answer(params):
    return params["amount"] * params["days"] * 1.5


# 3. The Full Pipeline
def run_experiment(problem: str):
    print(f"\nRunning PTP on: '{problem}'")
    
    # Step A: Expensive Parameter Extraction
    t0 = time.time()
    params = extract_parameters(problem, model="gpt-4")
    t1 = time.time()
    
    # Step B: Cheap Calculation
    final_score = compute_answer(params)
    
    print(f"  -> Result: {final_score}")
    print(f"  -> Time Taken: {t1 - t0:.2f}s")

#-----------Transparent Cache----------------
def my_transparent_cache(func):
    # 1. Closure
    store = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"\n--- Calling {func.__name__} ---")

        # A. Visualize RAW inputs
        print(f"\n Raw inputs: args={args}, kwargs={kwargs}")
        
        # B. Visualize key creation
        # sort kwargs to ensure (b=1, a=2) is same as (a=2, b=1)
        sorted_kwargs = tuple(sorted(kwargs.items()))
        key = (args, sorted_kwargs)
        print(f"2. Generated Key: {key}")

        # C. Visualize store state before lookup
        print(f"Store before lookup: {list(store.keys())}")

        if key in store:
            print("   -> STATUS: HIT! Returning saved value.")
            return store[key]
        
        print("   -> STATUS: MISS. Computing...")
        result = func(*args, **kwargs)

        # D. Updating the Closure
        store[key] = result
        print(f"4. Store AFTER: {list(store.keys())}")
        return result

    return wrapper


@my_transparent_cache
def heavy_math2(x, y, mode="fast"):
    return x + y


if __name__ == "__main__":
    print(">>> ROUND 1 (Cold Cache) <<<")
    run_experiment('Problem A: Calculate fees') # Should take ~1.0s
    run_experiment('Problem B: Calculate tax')  # Should take ~1.0s

    print("\n>>> ROUND 2 (Warm Cache) <<<")
    run_experiment('Problem A: Calculate fees') # Should be 0.0s!
    run_experiment('Problem A: Calculate fees') # Should be 0.0s!

    print("\n>>> ROUND 3 (Modification) <<<")
    # If we change the input, the cache MUST miss.
    run_experiment('Problem A: Calculate fees (modified)')

    # print("\n>>> Run 1: heavy_math(10, 20)")
    # heavy_math2(10, 20)

    # print("\n>>> Run 2: heavy_math(10, 20) <-- EXACT SAME INPUT")
    # heavy_math2(10, 20)

    # print("\n>>> Run 3: heavy_math(10, 20, mode='slow') <-- CHANGED KWARG")
    # heavy_math2(10, 20, mode="slow")

    # # First call: Should MISS
    # print(f"Result: {heavy_math(10, 20)}") 

    # # Second call: Should HIT
    # print(f"Result: {heavy_math(10, 20)}")

    # # Third call (different args): Should MISS
    # print(f"Result: {heavy_math(10, 30)}")

    # # 4th call (different args): Should MISS
    # print(f"Result: {heavy_math(a=10, b=30)}")

    # # 5th call (different args): Should HIT
    # print(f"Result: {heavy_math(a=10, b=30)}")