"""
build_embeddings.py

Builds vector embeddings for every commodity entry in sample_nmfc_dataset.json
via the Voyage AI embeddings API, and caches them to disk so the retrieval
layer doesn't recompute embeddings on every request.

Model: voyage-3.5-lite (configurable via VOYAGE_MODEL env var)
  - Hosted embedding API — no local model weights, no PyTorch, minimal memory
    footprint. This replaced an earlier local sentence-transformers approach,
    which used too much RAM to run on a 512MB hosting tier.
  - Requires a VOYAGE_API_KEY environment variable (voyageai.com).

Usage:
    python build_embeddings.py            # builds and caches embeddings_cache.npz
    python build_embeddings.py --query "wooden dining table, unassembled"
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import voyageai

DATASET_PATH = Path(__file__).parent / "sample_nmfc_dataset.json"
CACHE_PATH = Path(__file__).parent / "embeddings_cache.npz"
VOYAGE_MODEL = os.environ.get("VOYAGE_MODEL", "voyage-3.5-lite")


def _client() -> voyageai.Client:
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY must be set as an environment variable.")
    return voyageai.Client(api_key=api_key)


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


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid divide-by-zero on a degenerate vector
    return vectors / norms


def build_and_cache_embeddings(
    dataset_path: Path = DATASET_PATH, cache_path: Path = CACHE_PATH
) -> None:
    commodities = load_dataset(dataset_path)
    texts = [build_embedding_text(c) for c in commodities]
    item_ids = [c["item_id"] for c in commodities]

    print(f"Embedding {len(texts)} commodity entries via Voyage ({VOYAGE_MODEL})...")
    # input_type="document" tells Voyage these are corpus entries being indexed,
    # not search queries — Voyage embeds the two differently for better retrieval.
    result = _client().embed(texts, model=VOYAGE_MODEL, input_type="document")
    embeddings = _normalize(np.array(result.embeddings, dtype=np.float32))

    np.savez(cache_path, item_ids=np.array(item_ids), embeddings=embeddings)
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
) -> list[dict]:
    """
    Embed the query via Voyage (input_type="query" — asymmetric from the
    "document" embeddings above, optimized for retrieval) and return the
    top_k closest commodity entries by cosine similarity. Since both sides
    are normalized, cosine similarity reduces to a dot product.
    """
    item_ids, embeddings = load_cached_embeddings(cache_path)
    commodities = {c["item_id"]: c for c in load_dataset(dataset_path)}

    result = _client().embed([query], model=VOYAGE_MODEL, input_type="query")
    query_vec = _normalize(np.array(result.embeddings, dtype=np.float32))[0]

    similarities = embeddings @ query_vec

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
