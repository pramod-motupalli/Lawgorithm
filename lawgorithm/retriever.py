import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import pandas as pd
import os
import pickle
from typing import List, Dict, Any
from . import config
from . import utils

class HybridRetriever:
    def __init__(self):
        print("[INFO] Initializing Hybrid Retriever...")
        
        # Setup ChromaDB
        self.client = chromadb.PersistentClient(path=str(config.DB_DIR))
        
        # Use a simple default embedding function for now (or SentenceTransformer)
        # To match our previous logic, let's use the same model but via Chroma's wrapper if possible,
        # or just use the raw SentenceTransformer and pass embeddings manually.
        # Using the default all-MiniLM-L6-v2 is easiest for Chroma, but we want InLegalBERT.
        # We will generate embeddings manually to ensure we use InLegalBERT.
        
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        
        self.collections = {}
        self.bm25_indices = {}
        self.doc_stores = {} # In-memory store for BM25 mapping
        
        self._initialize_collections()

    def _initialize_collections(self):
        categories = ["Civil", "Criminal", "Traffic"]
        datasets = {
            "Civil": config.CIVIL_DATA_PATH,
            "Criminal": config.CRIMINAL_DATA_PATH,
            "Traffic": config.TRAFFIC_DATA_PATH
        }

        for cat in categories:
            # Get or Create Collection
            collection = self.client.get_or_create_collection(name=cat.lower())
            self.collections[cat] = collection
            
            # Load Data for BM25 and Metadata check
            data = utils.load_json_data(datasets[cat])
            self.doc_stores[cat] = data
            
            # Check if we need to ingest into Chroma
            if collection.count() == 0 and data:
                print(f"[INFO] Ingesting {cat} data into ChromaDB (First Run)...")
                self._ingest_data(collection, data)
            else:
                print(f"[INFO] {cat} collection ready ({collection.count()} docs).")

            # Build BM25 Index (with caching)
            if data:
                self._build_bm25_index(cat, data)

    def _build_bm25_index(self, category, data):
        cache_file = config.DB_DIR / f"bm25_{category.lower()}.pkl"
        
        # Check if cache exists and is valid
        if cache_file.exists():
            try:
                # Check modification time of source data vs cache
                source_file = None
                if category == "Civil": source_file = config.CIVIL_DATA_PATH
                elif category == "Criminal": source_file = config.CRIMINAL_DATA_PATH
                elif category == "Traffic": source_file = config.TRAFFIC_DATA_PATH
                
                if source_file and source_file.exists():
                    if source_file.stat().st_mtime > cache_file.stat().st_mtime:
                        print(f"[INFO] Cache stale for {category}, rebuilding...")
                        raise ValueError("Cache stale")
                
                with open(cache_file, "rb") as f:
                    print(f"[INFO] Loading BM25 index for {category} from cache...")
                    self.bm25_indices[category] = pickle.load(f)
                    return
            except Exception as e:
                print(f"[WARN] Failed to load cache for {category}: {e}")
        
        # Rebuild Index
        print(f"[INFO] Building BM25 index for {category}...")
        corpus = [d.get("judgment_summary", "") or d.get("description", "") or "" for d in data]
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_indices[category] = bm25
        
        # Save to cache
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(bm25, f)
            print(f"[INFO] Saved BM25 index for {category} to cache.")
        except Exception as e:
            print(f"[ERROR] Failed to save cache for {category}: {e}")

    def _ingest_data(self, collection, data):
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        # Batch processing
        batch_size = 100
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            
            batch_texts = [d.get("judgment_summary", "") or d.get("description", "") or "" for d in batch]
            # Generate unique IDs using global index to avoid duplicates
            batch_ids = [f"doc_{i+j}" for j in range(len(batch))]
            
            # Create simple metadata (Chroma doesn't like lists in metadata sometimes, keep it simple)
            batch_metadatas = [{"title": str(d.get("title", "")), "verdict": str(d.get("verdict", "")), "case_id": str(d.get("case_id", ""))} for d in batch]
            
            # Generate Embeddings
            batch_embeddings = self.encoder.encode(batch_texts).tolist()
            
            collection.add(
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            print(f"   Indexed {i + len(batch)} / {len(data)}")

    def search(self, query: str, category: str, k: int = 5) -> List[Dict[str, Any]]:
        if category not in self.collections:
            return []
            
        # 1. Vector Search
        query_embedding = self.encoder.encode([query]).tolist()
        vector_results = self.collections[category].query(
            query_embeddings=query_embedding,
            n_results=k
        )
        
        # Extract IDs and map back to documents
        ids = vector_results['ids'][0]
        
        results = []
        for doc_id in ids:
            # doc_id format is "doc_X" where X is the index in the original data array
            try:
                idx = int(doc_id.split('_')[1])
                if idx < len(self.doc_stores[category]):
                    results.append(self.doc_stores[category][idx])
            except (ValueError, IndexError):
                continue
                
        return results
