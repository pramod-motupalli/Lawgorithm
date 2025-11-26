import os
import sys
from lawgorithm.agents import RouterAgent, CaseParsingAgent, RetrievalAgent, VerdictAgent
from lawgorithm import config

def test_pipeline():
    print("=== Testing Lawgorithm Pipeline ===")
    
    # Check API Key
    if not config.GOOGLE_API_KEY:
        print("[ERROR] GOOGLE_API_KEY not found in environment variables.")
        return

    # 1. Test Router
    print("\n[1] Testing Router Agent...")
    router = RouterAgent()
    description = "The accused was caught stealing a motorcycle from the parking lot. He was apprehended by the security guard."
    category = router.route(description)
    print(f"Input: {description}")
    print(f"Output Category: {category}")
    
    if category not in ["Criminal", "Civil", "Traffic"]:
        print("[FAIL] Router returned invalid category.")
    else:
        print("[PASS] Router works.")

    # 2. Test Parser
    print("\n[2] Testing Case Parsing Agent...")
    parser = CaseParsingAgent()
    structure = parser.parse(description, "State", "John Doe")
    print(f"Structured Keys: {list(structure.keys())}")
    if "summary" in structure and "legal_issues" in structure:
        print("[PASS] Parser works.")
    else:
        print("[FAIL] Parser output missing keys.")

    # 3. Test Retrieval (Mocking if data not present, but we expect data)
    print("\n[3] Testing Retrieval Agent...")
    try:
        retriever = RetrievalAgent()
        precedents = retriever.retrieve(structure['summary'], category, k=2)
        print(f"Retrieved {len(precedents)} cases.")
        if len(precedents) > 0:
            print(f"Top case: {precedents[0].get('title')}")
            print("[PASS] Retrieval works.")
        else:
            print("[WARN] No cases retrieved (might be empty dataset).")
    except Exception as e:
        print(f"[FAIL] Retrieval failed: {e}")

    # 4. Test Verdict
    print("\n[4] Testing Verdict Agent...")
    if 'precedents' in locals() and len(precedents) > 0:
        verdict_agent = VerdictAgent()
        verdict = verdict_agent.predict(structure, precedents, category)
        print("Verdict Preview:")
        print(verdict[:200] + "...")
        if "academic" in verdict.lower() or "disclaimer" in verdict.lower():
             print("[PASS] Verdict generated with disclaimer.")
        else:
             print("[WARN] Verdict generated but check for disclaimer.")
    else:
        print("[SKIP] Skipping verdict test due to no precedents.")

if __name__ == "__main__":
    test_pipeline()
