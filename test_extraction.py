import build_3_legal_dataset as b3

test_cases = [
    {
        "text": "The accused was charged under Section 302 of IPC and Section 34 of IPC. The court convicted him.",
        "expected_ipc": ["302", "34"],
        "expected_verdict": "Conviction",
        "expected_crime": "Murder, Common Intention"
    },
    {
        "text": "FIR registered u/s 420, 468, 471 IPC. The appeal was allowed and he was acquitted.",
        "expected_ipc": ["420", "468", "471"],
        "expected_verdict": "Acquittal, Appeal Allowed",
        "expected_crime": "Cheating, Forgery"
    },
    {
        "text": "Punishable u/ss 376(2)(n) and 506 IPC. Sentenced to rigorous imprisonment for 10 years.",
        "expected_ipc": ["376", "506"],
        "expected_verdict": "Imprisonment",
        "expected_crime": "Rape, Criminal Intimidation"
    },
    {
        "text": "Civil appeal dismissed. No costs.",
        "expected_ipc": [],
        "expected_verdict": "Appeal Dismissed",
        "expected_crime": "Unknown"
    }
]

for i, case in enumerate(test_cases):
    print(f"--- Case {i+1} ---")
    text = case["text"]
    ipc = b3.extract_ipc_sections(text)
    verdict = b3.extract_verdict_type(text)
    crime = b3.extract_crime_category(ipc)
    
    print(f"Text: {text}")
    print(f"IPC: {ipc} (Expected: {case['expected_ipc']})")
    print(f"Verdict: {verdict} (Expected: {case['expected_verdict']})")
    print(f"Crime: {crime} (Expected: {case['expected_crime']})")
    print()
