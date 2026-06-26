import os
import sys
import time
import subprocess

# Configure encoding for Windows console output
sys.stdout.reconfigure(encoding='utf-8')

def run_module(module_name):
    print(f"\n==================================================")
    print(f"   RUNNING MODULE: {module_name}")
    print(f"==================================================")
    t0 = time.time()
    
    # Run the module as a script with python path set correctly
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", module_name],
        capture_output=False,
        text=True
    )
    
    duration = time.time() - t0
    if result.returncode != 0:
        print(f"\n[Error] Module {module_name} failed with return code {result.returncode}.")
        sys.exit(result.returncode)
        
    print(f"\n[Success] Module {module_name} finished successfully in {duration:.2f} seconds.")

def main():
    print("==================================================")
    print("   TRENDYOL SEARCH RELEVANCE DATATHON PIPELINE")
    print("==================================================")
    t_start = time.time()
    
    # Phase 1 & 2: High-Performance Data Processing & Negative Sampling
    run_module("src.data_processing")
    
    # Phase 3: Dense Cosine Embedding Generation
    run_module("src.generate_embeddings")
    
    # Phase 4: Multi-Modal Feature Engineering
    run_module("src.feature_engineering")
    
    # Phase 5: GBDT Model Training with CV & Threshold Sweep
    run_module("src.train")
    
    # Phase 6: Ensemble Inference & Sanity Checks
    run_module("src.predict")
    
    print("\n==================================================")
    print(f"   PIPELINE FINISHED IN {time.time() - t_start:.2f} SECONDS")
    print("==================================================")

if __name__ == "__main__":
    main()
