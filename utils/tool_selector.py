"""
utils/tool_selector.py
======================
Contribution 2 — Tool Selector Agent

Fixes the 70% tool selection problem by inserting a semantic retrieval
step BEFORE the manager sees the tool list. Instead of choosing from
all 27 tools, the manager only sees the top-k most relevant ones.

Flow:
  User query
      ↓
  Embed query with sentence-transformers
      ↓
  Cosine similarity against all 27 tool embeddings
      ↓
  Return top-k tools (default k=7)
      ↓
  Manager sees only k tools → picks correctly
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from contextlib import contextmanager

# Always include these tools regardless of query score
ALWAYS_INCLUDE = {"make_dir", "get_model_zoo", "get_best_model"}


class ToolSelectorIndex:
    def __init__(self, manager, model_name: str = "all-mpnet-base-v2"):
        self.manager = manager
        self.encoder = SentenceTransformer(model_name)
        self._build_index()

    def _build_index(self):
        tools = self.manager.llm_config["tools"]
        self.tools = tools
        self.tool_names = []
        self.tool_texts = []

        for t in tools:
            fn = t["function"]
            name = fn["name"]
            desc = fn.get("description", "")
            # Include parameter names as extra signal
            params = list(fn.get("parameters", {}).get("properties", {}).keys())
            text = f"{name}. {desc}. Parameters: {', '.join(params)}"
            self.tool_names.append(name)
            self.tool_texts.append(text)

        print(f"[ToolSelector] Encoding {len(self.tool_texts)} tools...")
        self.embeddings = self.encoder.encode(self.tool_texts, normalize_embeddings=True)
        print(f"[ToolSelector] Index ready. Shape: {self.embeddings.shape}")

    def get_top_k(self, query: str, k: int = 7) -> list:
        query_vec = self.encoder.encode([query], normalize_embeddings=True)
        scores = (self.embeddings @ query_vec.T).flatten()

        # Always-include tools get score boosted to top
        pinned = []
        ranked_indices = np.argsort(scores)[::-1].tolist()

        selected_names = set()
        result = []

        # First add always-include tools
        for i, name in enumerate(self.tool_names):
            if name in ALWAYS_INCLUDE:
                result.append(self.tools[i])
                selected_names.add(name)
                pinned.append(name)

        # Then add top-k by score (excluding already pinned)
        added = 0
        for idx in ranked_indices:
            if added >= k:
                break
            name = self.tool_names[idx]
            if name not in selected_names:
                result.append(self.tools[idx])
                selected_names.add(name)
                added += 1

        print(f"[ToolSelector] Query: '{query[:60]}...'")
        print(f"[ToolSelector] Pinned: {pinned}")
        print(f"[ToolSelector] Top-{k} selected: {[t['function']['name'] for t in result if t['function']['name'] not in ALWAYS_INCLUDE]}")
        return result

    def score_debug(self, query: str):
        """Print all tools ranked by similarity score — useful for diagnosing failures."""
        query_vec = self.encoder.encode([query], normalize_embeddings=True)
        scores = (self.embeddings @ query_vec.T).flatten()
        ranked = sorted(zip(self.tool_names, scores), key=lambda x: x[1], reverse=True)
        print(f"\nQuery: '{query}'")
        print(f"{'Rank':<5} {'Score':<8} {'Tool'}")
        print("-" * 60)
        for rank, (name, score) in enumerate(ranked, 1):
            pin = " [pinned]" if name in ALWAYS_INCLUDE else ""
            print(f"{rank:<5} {score:.4f}   {name}{pin}")

    @contextmanager
    def patch_manager(self, query: str, k: int = 7):
        """
        Context manager that temporarily replaces the manager's full tool
        list with the top-k filtered list, then restores it after.

        Usage:
            with tool_index.patch_manager(user_message, k=7):
                user_proxy.initiate_chat(manager, message=user_message)
        """
        full_tools = self.manager.llm_config["tools"]
        filtered = self.get_top_k(query, k=k)
        try:
            self.manager.llm_config["tools"] = filtered
            yield filtered
        finally:
            self.manager.llm_config["tools"] = full_tools
