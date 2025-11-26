import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Suppress HuggingFace symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = BASE_DIR
CIVIL_DATA_PATH = DATA_DIR / "india_civil_verdicts.json"
CRIMINAL_DATA_PATH = DATA_DIR / "india_criminal_verdicts.json"
TRAFFIC_DATA_PATH = DATA_DIR / "india_traffic_verdicts.json"

# Vector DB
DB_DIR = DATA_DIR / "chroma_db"

# Laws JSON Directory
LAWS_DIR = DATA_DIR / "laws_json"

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model Config
EMBEDDING_MODEL_NAME = "law-ai/InLegalBERT"
LLM_PROVIDER = "groq" # Options: "google", "groq"
LLM_MODEL_NAME = "llama-3.1-8b-instant" # or "gemini-2.0-flash-exp"
