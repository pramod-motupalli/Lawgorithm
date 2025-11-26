import sqlite3
import sys
print(f"SQLite version: {sqlite3.sqlite_version}")
try:
    import chromadb
    print("ChromaDB imported successfully")
except Exception as e:
    print(f"ChromaDB import failed: {e}")
