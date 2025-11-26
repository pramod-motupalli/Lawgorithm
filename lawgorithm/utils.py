import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

def load_json_data(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load data from a JSON file.
    """
    try:
        if not file_path.exists():
            print(f"[WARN] File not found: {file_path}")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return []

def load_laws_data(laws_dir: Path) -> Dict[str, Any]:
    """
    Load all law JSON files from the directory.
    Returns a dictionary where keys are filenames (without extension) and values are the JSON content.
    """
    laws = {}
    if not laws_dir.exists():
        print(f"[WARN] Laws directory not found: {laws_dir}")
        return laws
        
    for file_path in laws_dir.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                laws[file_path.stem] = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load law file {file_path}: {e}")
            
    return laws
