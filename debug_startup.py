import time
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

log("Starting debug script...")

try:
    log("Importing lawgorithm.analytics...")
    from lawgorithm.analytics import AnalyticsEngine
    
    log("Initializing AnalyticsEngine...")
    start = time.time()
    analytics = AnalyticsEngine()
    log(f"AnalyticsEngine initialized in {time.time() - start:.2f}s")
    
    log("Importing lawgorithm.retriever...")
    from lawgorithm.retriever import HybridRetriever
    
    log("Initializing HybridRetriever...")
    start = time.time()
    retriever = HybridRetriever()
    log(f"HybridRetriever initialized in {time.time() - start:.2f}s")

except Exception as e:
    log(f"Error: {e}")
    import traceback
    traceback.print_exc()

log("Debug script finished.")
