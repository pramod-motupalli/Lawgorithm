import io
import json
import re
import time
import math
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

# Years to scan in the AWS Supreme Court dataset
START_YEAR = 1950
END_YEAR = 2025

# Target number of cases per category
MIN_EACH = 10_000

# Output JSON files
OUTPUT_CIVIL = "india_civil_verdicts.json"
OUTPUT_CRIMINAL = "india_criminal_verdicts.json"
OUTPUT_TRAFFIC = "india_traffic_verdicts.json"

# Public AWS Open Data URL template (Indian Supreme Court Judgments)
METADATA_URL_TEMPLATE = (
    "https://indian-supreme-court-judgments.s3.amazonaws.com/"
    "metadata/parquet/year={year}/metadata.parquet"
)

# Keyword sets for simple text-based classification

CRIMINAL_KEYWORDS = [
    # "ipc", "indian penal code", "penal code",
    # "offence", "offense", "crime", "criminal",
    # "fir", "f.i.r", "first information report",
    # "charge sheet", "chargesheet",
    # "prosecution", "accused",
    "murder", "homicide", "culpable homicide",
    "rape", "sexual assault", "molestation",
    "kidnapping", "abduction",
    "robbery", "dacoity", "theft", "burglary",
    "assault", "hurt",
    "cheating", "fraud", "forgery",
    "ndps", "narcotic drugs", "psychotropic",
    "bail", "custody", "remand", "sentence", "convicted", "acquitted",
    "pocso", "protection of children from sexual offences",
]

# Traffic: focus on Motor Vehicles Act / road accidents etc.
TRAFFIC_KEYWORDS = [
    "motor vehicles act", "motor vehicle act",
    "mv act", "m.v. act", "m v act", "m.v.act", "mvact",
    "motor vehicles act, 1988", "motor vehicles act 1988",
    "mva", "m.v.a.",
    "road accident", "motor accident", "traffic accident",
    "rash and negligent driving", "rash driving",
    "driving licence", "driving license", "learner's licence",
    "regional transport", "transport authority",
    "hit and run", "hit-and-run",
    "motor accident claims tribunal", "mact",
]

# Civil-ish clues; plus, anything that is not clearly criminal/traffic
# will be treated as civil to ensure enough civil cases.
CIVIL_KEYWORDS = [
    "contract", "specific performance",
    "property", "ownership", "title", "possession",
    "land acquisition", "compensation",
    "service matter", "employment", "dismissal from service",
    "matrimonial", "divorce", "maintenance", "alimony",
    "custody of child", "guardianship",
    "arbitration", "commercial dispute",
    "tax", "income tax", "gst", "excise", "customs",
    "company law", "insolvency", "bankruptcy", "ibc",
    "writ petition", "mandamus", "certiorari", "habeas corpus",
    "civil suit", "injunction", "declaration",
]

# ============================================================
# HELPERS
# ============================================================

def fetch_metadata_parquet(year: int) -> pd.DataFrame | None:
    """
    Download metadata.parquet for a given year and return as a pandas DataFrame.
    Returns None if file not available or on error.
    """
    url = METADATA_URL_TEMPLATE.format(year=year)
    print(f"\n[INFO] Fetching metadata for year {year}: {url}")
    try:
        resp = requests.get(url, timeout=60)
    except Exception as e:
        print(f"[WARN] Request failed for year {year}: {e}")
        return None

    if resp.status_code != 200:
        print(f"[WARN] No metadata for year {year} (status {resp.status_code})")
        return None

    try:
        buf = io.BytesIO(resp.content)
        df = pd.read_parquet(buf)
        print(f"[OK] Loaded {len(df)} records for year {year}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to read parquet for year {year}: {e}")
        return None


def html_to_text(html: str) -> str:
    """
    Convert HTML (raw_html field) into plain text.
    """
    if not isinstance(html, str) or not html.strip():
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def contains_any(text: str, keywords: list[str]) -> bool:
    """
    Return True if 'text' contains any of the 'keywords' (case-insensitive).
    """
    if not isinstance(text, str):
        return False
    t = text.lower()
    return any(k.lower() in t for k in keywords)


def sanitize_value(v):
    """
    Convert values to JSON-safe types:
    - NaN / NA / NaT -> None
    - pd.Timestamp -> ISO string
    - everything else -> unchanged
    """
    # Handle lists/dicts (JSON safe already, usually)
    if isinstance(v, (list, dict)):
        return v

    # Handle pandas NA / NaN / NaT
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        # Some types don't work with pd.isna or return arrays; ignore
        pass

    # Convert timestamps to strings
    if isinstance(v, pd.Timestamp):
        return v.isoformat()

    # Normal Python float NaN (in case it appears)
    if isinstance(v, float) and math.isnan(v):
        return None

    return v


def sanitize_record(rec: dict) -> dict:
    """
    Apply sanitize_value to every field of the record.
    """
    return {k: sanitize_value(v) for k, v in rec.items()}


import re

# ... (existing imports are fine, just adding re if not present, but replace_file_content replaces a block)
# Wait, I need to add 'import re' at the top first.

def extract_ipc_sections(text: str) -> list[str]:
    """
    Extract IPC sections using regex.
    Matches: "Section 302 of IPC", "S. 302 IPC", "u/s 302 IPC", "302/34 IPC" etc.
    """
    if not isinstance(text, str):
        return []
    
    # Normalize text slightly for easier matching
    t = text.replace("Indian Penal Code", "IPC").replace("Penal Code", "IPC")
    
    # Pattern 1: "Section 302 IPC" or "u/s 302 IPC" or "Sections 302, 307 IPC"
    # We look for the keyword IPC and look backwards for sections
    # OR we look for "Section X" and check if IPC is nearby (simplified: just look for "Section X... IPC")
    
    # Robust approach: Find "IPC" and look around it, or specific patterns.
    # Added () to character class to handle 376(2)(n)
    
    patterns = [
        # "Section 302 ... of IPC" or "u/s 302 ... IPC"
        # Captures "302", "302/34", "302, 307", "376(2)(n)"
        r"(?:Section|Sec\.?|s\.?|u/s|u/ss)\.?\s*([\d\w\s,/\(\)]+?)\s+(?:of\s+(?:the\s+)?)?IPC",
        
        # "IPC Section 302"
        r"IPC\s+(?:Section|Sec\.?|s\.?|u/s|u/ss)\.?\s*([\d\w\s,/\(\)]+)",
        
        # "under Section 302 ... IPC"
        r"under\s+(?:Section|Sec\.?|s\.?)\s*([\d\w\s,/\(\)]+?)\s+(?:of\s+(?:the\s+)?)?IPC",
    ]
    
    found_tokens = set()
    for pat in patterns:
        matches = re.findall(pat, t, re.IGNORECASE | re.DOTALL)
        for m in matches:
            # m is a string like "302", "302/34", "302, 307"
            # Split by comma, 'and', 'read with', '/'
            # Also clean up parentheses if they are just enclosing the number like (302) -> 302
            # But keep 376(2) as is? 
            # For now, let's just split and clean.
            raw_secs = re.split(r"[,/]| and | read with ", m)
            for s in raw_secs:
                s = s.strip()
                # Validate it looks like a section (digits, maybe letter suffix, maybe parens)
                # Allow 376(2)(n) -> starts with digit, contains alphanumeric + parens
                if re.match(r"^\d+[A-Za-z0-9\(\)]*$", s):
                    found_tokens.add(s)
    
    return sorted(list(found_tokens))

def extract_verdict_type(text: str) -> str:
    """
    Identify if it involves Jail, Fine, Acquittal, etc.
    """
    if not isinstance(text, str):
        return "Unknown"
    t = text.lower()
    outcomes = []
    if "acquitted" in t or "acquittal" in t:
        outcomes.append("Acquittal")
    if "convicted" in t or "conviction" in t:
        outcomes.append("Conviction")
    
    # Flexible matching for appeal allowed/dismissed
    if "appeal" in t and ("allowed" in t or "accepted" in t):
        outcomes.append("Appeal Allowed")
    if "appeal" in t and "dismissed" in t:
        outcomes.append("Appeal Dismissed")
    
    punishments = []
    if "death sentence" in t or "capital punishment" in t:
        punishments.append("Death Penalty")
    if "life imprisonment" in t:
        punishments.append("Life Imprisonment")
    elif "rigorous imprisonment" in t or "simple imprisonment" in t or "imprisonment" in t or "jail" in t or "incarceration" in t:
        punishments.append("Imprisonment")
    if "fine" in t:
        punishments.append("Fine")
        
    if not outcomes and not punishments:
        return "Unknown"
    
    return ", ".join(sorted(list(set(outcomes + punishments))))

def extract_crime_category(ipc_sections: list[str]) -> str:
    """
    Infer broad crime category from IPC sections.
    """
    if not ipc_sections:
        return "Unknown"
    
    # Simple mapping for common sections
    mapping = {
        "302": "Murder",
        "304": "Culpable Homicide",
        "304A": "Negligence Death",
        "304B": "Dowry Death",
        "307": "Attempt to Murder",
        "323": "Voluntarily Causing Hurt",
        "324": "Hurt by Dangerous Weapons",
        "325": "Grievous Hurt",
        "326": "Grievous Hurt by Dangerous Weapons",
        "354": "Assault on Woman",
        "363": "Kidnapping",
        "364": "Kidnapping for Murder",
        "366": "Kidnapping Woman",
        "376": "Rape",
        "378": "Theft", "379": "Theft", "380": "Theft", "381": "Theft", "382": "Theft",
        "383": "Extortion", "384": "Extortion",
        "390": "Robbery", "391": "Dacoity", "392": "Robbery", "395": "Dacoity",
        "405": "Breach of Trust", "406": "Breach of Trust", "409": "Breach of Trust",
        "415": "Cheating", "420": "Cheating",
        "463": "Forgery", "464": "Forgery", "465": "Forgery", "468": "Forgery", "471": "Forgery",
        "498A": "Cruelty by Husband/Relatives",
        "506": "Criminal Intimidation",
        "120B": "Criminal Conspiracy",
        "34": "Common Intention"
    }
    
    crimes = set()
    for sec in ipc_sections:
        if sec in mapping:
            crimes.add(mapping[sec])
        
    if not crimes:
        return "Other IPC Offense"
    return ", ".join(sorted(list(crimes)))

# ============================================================
# MAIN LOGIC
# ============================================================

def main():
    civil_cases: list[dict] = []
    criminal_cases: list[dict] = []
    traffic_cases: list[dict] = []

    for year in range(END_YEAR, START_YEAR - 1, -1):
        # If all three reached target, we can stop early
        if (
            len(civil_cases) >= MIN_EACH and
            len(criminal_cases) >= MIN_EACH and
            len(traffic_cases) >= MIN_EACH
        ):
            break

        df = fetch_metadata_parquet(year)
        if df is None or df.empty:
            continue

        print(f"[INFO] Scanning cases for year {year} ...")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            # Again stop early if everything filled
            if (
                len(civil_cases) >= MIN_EACH and
                len(criminal_cases) >= MIN_EACH and
                len(traffic_cases) >= MIN_EACH
            ):
                break

            # Build a metadata text string
            meta_pieces = [
                str(row.get("title", "")),
                str(row.get("description", "")),
                str(row.get("disposal_nature", "")),
                str(row.get("citation", "")),
                str(row.get("court", "")),
            ]
            meta_text = " | ".join(meta_pieces)

            # Get verdict text from HTML
            raw_html = row.get("raw_html")
            verdict_text = html_to_text(raw_html)

            # Combine metadata and body for classification
            combined_text = (meta_text or "") + "\n" + (verdict_text or "")

            is_traffic = contains_any(combined_text, TRAFFIC_KEYWORDS)
            is_criminal = contains_any(combined_text, CRIMINAL_KEYWORDS)
            is_civil = contains_any(combined_text, CIVIL_KEYWORDS)

            # Decode year safely
            year_val = row.get("year")
            year_int = None
            try:
                if pd.notna(year_val):
                    year_int = int(year_val)
            except Exception:
                year_int = None

            # Extract extra details
            ipc_list = extract_ipc_sections(combined_text)
            verdict_type = extract_verdict_type(combined_text)
            crime_cat = extract_crime_category(ipc_list)

            # Build base record once
            record = {
                "case_id": row.get("case_id"),
                "title": row.get("title"),
                "petitioner": row.get("petitioner"),
                "respondent": row.get("respondent"),
                "plaintiff": row.get("petitioner"),  # Alias
                "defendant": row.get("respondent"),  # Alias
                "ipc_sections": ipc_list,
                "verdict": verdict_type,
                "crime_committed": crime_cat,
                "description": row.get("description"),
                "judge": row.get("judge"),
                "author_judge": row.get("author_judge"),
                "citation": row.get("citation"),
                "cnr": row.get("cnr"),
                "decision_date": str(row.get("decision_date")),
                "disposal_nature": row.get("disposal_nature"),
                "court": row.get("court"),
                "available_languages": row.get("available_languages"),
                "path": row.get("path"),
                "nc_display": row.get("nc_display"),
                "scraped_at": str(row.get("scraped_at")),
                "year": year_int,
                "judgment_summary": verdict_text[:3000] if verdict_text else "",
                "source": "Indian Supreme Court Judgments (AWS Open Data, CC-BY 4.0)",
            }

            # IMPORTANT:
            # A case can go into multiple buckets (overlap allowed)
            if is_traffic and len(traffic_cases) < MIN_EACH:
                traffic_cases.append({**record, "category": "traffic"})
            if is_criminal and len(criminal_cases) < MIN_EACH:
                criminal_cases.append({**record, "category": "criminal"})
            # Civil: explicit match OR fallback when neither criminal nor traffic
            if (is_civil or (not is_traffic and not is_criminal)) and len(civil_cases) < MIN_EACH:
                civil_cases.append({**record, "category": "civil"})

        print(
            f"[STATUS] So far → "
            f"civil: {len(civil_cases)}, "
            f"criminal: {len(criminal_cases)}, "
            f"traffic: {len(traffic_cases)}"
        )

        # Be polite to the server
        time.sleep(1)

    # ========================================================
    # SAVE JSON FILES (with sanitization)
    # ========================================================

    print("\n[INFO] Finished scanning years.")
    print(
        f"[COUNTS] Civil: {len(civil_cases)}, "
        f"Criminal: {len(criminal_cases)}, "
        f"Traffic: {len(traffic_cases)}"
    )

    print("[INFO] Sanitizing & saving JSON files...")

    civil_clean = [sanitize_record(r) for r in civil_cases]
    criminal_clean = [sanitize_record(r) for r in criminal_cases]
    traffic_clean = [sanitize_record(r) for r in traffic_cases]

    with open(OUTPUT_CIVIL, "w", encoding="utf-8") as f:
        json.dump(civil_clean, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Saved civil cases → {OUTPUT_CIVIL}")

    with open(OUTPUT_CRIMINAL, "w", encoding="utf-8") as f:
        json.dump(criminal_clean, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Saved criminal cases → {OUTPUT_CRIMINAL}")

    with open(OUTPUT_TRAFFIC, "w", encoding="utf-8") as f:
        json.dump(traffic_clean, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Saved traffic cases → {OUTPUT_TRAFFIC}")


if __name__ == "__main__":
    main()
