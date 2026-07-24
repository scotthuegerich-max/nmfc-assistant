"""
build_embeddings.py

Builds vector embeddings for every commodity entry in sample_nmfc_dataset.json
and caches them to disk, so the retrieval layer doesn't recompute embeddings
on every API request.

Model: sentence-transformers/all-MiniLM-L6-v2
  - Free, runs locally, no API key required, small enough to embed at build
    time or even cold-start in a lightweight API deployment.
  - If you'd rather standardize on a single AI provider for the whole stack
    (e.g. use Voyage AI embeddings since Claude/Anthropic recommends Voyage
    for retrieval), swap embed_texts() below for a Voyage API call — the
    rest of the pipeline (caching, cosine similarity, search) stays the same.

Usage:
    python build_embeddings.py            # builds and caches embeddings.npz
    python build_embeddings.py --query "wooden dining table, unassembled"
"""

import json
import sys
import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATASET_PATH = Path(__file__).parent / "sample_nmfc_dataset.json"
CACHE_PATH = Path(__file__).parent / "embeddings_cache.npz"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data["commodities"]


def build_embedding_text(entry: dict) -> str:
    """
    Combine description + keywords into one string per commodity.
    Keywords are repeated once to give them modest extra weight in the
    embedding without overwhelming the natural-language description.
    """
    description = entry["commodity_description"]
    keywords = " ".join(entry.get("keywords", []))
    return f"{description}. Related terms: {keywords}"


def build_and_cache_embeddings(
    dataset_path: Path = DATASET_PATH, cache_path: Path = CACHE_PATH
) -> None:
    commodities = load_dataset(dataset_path)
    texts = [build_embedding_text(c) for c in commodities]
    item_ids = [c["item_id"] for c in commodities]

    print(f"Loading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Embedding {len(texts)} commodity entries...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    np.savez(
        cache_path,
        item_ids=np.array(item_ids),
        embeddings=embeddings.astype(np.float32),
    )
    print(f"Cached {len(item_ids)} embeddings to {cache_path}")


def load_cached_embeddings(cache_path: Path = CACHE_PATH):
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cached embeddings at {cache_path}. Run build_and_cache_embeddings() first."
        )
    data = np.load(cache_path, allow_pickle=True)
    return data["item_ids"], data["embeddings"]


def search(
    query: str,
    top_k: int = 5,
    dataset_path: Path = DATASET_PATH,
    cache_path: Path = CACHE_PATH,
    model: SentenceTransformer | None = None,
) -> list[dict]:
    """
    Embed the query and return the top_k closest commodity entries by
    cosine similarity. Since cached embeddings are pre-normalized, cosine
    similarity reduces to a dot product.
    """
    item_ids, embeddings = load_cached_embeddings(cache_path)
    commodities = {c["item_id"]: c for c in load_dataset(dataset_path)}

    if model is None:
        model = SentenceTransformer(MODEL_NAME)

    query_vec = model.encode([query], normalize_embeddings=True)[0]
    similarities = embeddings @ query_vec  # dot product == cosine sim (normalized)

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        item_id = str(item_ids[idx])
        results.append(
            {
                "item_id": item_id,
                "commodity_description": commodities[item_id]["commodity_description"],
                "similarity": float(similarities[idx]),
                "density_based": commodities[item_id]["density_based"],
            }
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Test a search query against the cached embeddings")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    if args.query:
        results = search(args.query, top_k=args.top_k)
        print(f"\nTop {args.top_k} matches for: \"{args.query}\"\n")
        for r in results:
            print(f"  [{r['similarity']:.3f}] {r['item_id']} — {r['commodity_description']}")
    else:
        build_and_cache_embeddings()
