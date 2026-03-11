"""
rag_engine.py
─────────────
RAG pipeline:
  1. Convert analytics results into natural-language document chunks.
  2. Embed them with sentence-transformers.
  3. Store in a FAISS cosine-similarity index.
  4. At query time: retrieve top-k chunks → send to Groq LLM as context.

All tunable parameters (model names, top-k, temperature) come from config.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    EMBEDDING_MODEL,
    RAG_TOP_K,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    DELAY_THRESHOLD_DAYS,
    is_groq_configured,
)
from analytics import (
    delay_distribution_per_warehouse,
    avg_processing_time_per_warehouse,
    avg_shipping_delay_per_warehouse,
    fastest_shipping_product,
    orders_delayed_more_than_n,
    top_delayed_destinations,
    monthly_order_volume,
    generate_auto_insights,
)



def build_documents(df: pd.DataFrame) -> list[str]:
    """
    Convert analytics results into natural-language sentences for embedding.
    Uses order_processing_time, shipping_delay, and total_lead_time columns.
    """
    docs: list[str] = []
    total = len(df)

    for _, row in avg_processing_time_per_warehouse(df).iterrows():
        docs.append(
            f"{row['Warehouse']} has an average order processing time of "
            f"{row['avg_processing_time']} days (order received to dispatched)."
        )

    for _, row in avg_shipping_delay_per_warehouse(df).iterrows():
        docs.append(
            f"{row['Warehouse']} has an average shipping delay of "
            f"{row['avg_shipping_delay']} days (dispatched to delivered)."
        )

    for _, row in delay_distribution_per_warehouse(df).iterrows():
        docs.append(
            f"{row['Warehouse']} total lead time averages {row['avg_total_lead_time']} days "
            f"({row['avg_processing_time']}d processing + {row['avg_shipping_delay']}d shipping) "
            f"across {row['total_orders']} orders."
        )

    
    median_ship = fastest_shipping_product(df)["avg_shipping_delay"].median()
    for _, row in fastest_shipping_product(df).iterrows():
        speed = "fast" if row["avg_shipping_delay"] <= median_ship else "slow"
        docs.append(
            f"{row['Product']} has an average shipping delay of {row['avg_shipping_delay']} days, "
            f"making it a relatively {speed}-shipping product."
        )

    delayed = orders_delayed_more_than_n(df)
    pct = round(len(delayed) / total * 100, 1)
    docs.append(
        f"{len(delayed)} of {total} orders ({pct}%) had a shipping delay exceeding "
        f"{DELAY_THRESHOLD_DAYS} days."
    )

    for wh, grp in df.groupby("Warehouse"):
        d_wh = grp[grp["shipping_delay"] > DELAY_THRESHOLD_DAYS]
        p_wh = round(len(d_wh) / len(grp) * 100, 1)
        docs.append(
            f"{wh} had {len(d_wh)} orders with shipping delay over {DELAY_THRESHOLD_DAYS} days "
            f"({p_wh}% of its {len(grp)} orders)."
        )

    top_dest = top_delayed_destinations(df, 5)
    dest_list = ", ".join(
        f"{r['Destination']} ({r['avg_shipping_delay']}d)" for _, r in top_dest.iterrows()
    )
    docs.append(f"Top 5 destinations by shipping delay: {dest_list}.")

    monthly = monthly_order_volume(df)
    if not monthly.empty:
        peak = monthly.loc[monthly["order_count"].idxmax()]
        docs.append(
            f"Order volume peaked in {peak['month']} with {peak['order_count']} orders "
            f"over {len(monthly)} months of data."
        )

    avg_proc = round(float(df["order_processing_time"].mean()), 2)
    avg_ship = round(float(df["shipping_delay"].mean()), 2)
    avg_lead = round(float(df["total_lead_time"].mean()), 2)
    docs.append(
        f"Overall averages: order processing time {avg_proc}d, "
        f"shipping delay {avg_ship}d, total lead time {avg_lead}d "
        f"across all {total} orders."
    )

    for (wh, prod), grp in df.groupby(["Warehouse", "Product"]):
        avg_proc_g = round(float(grp["order_processing_time"].mean()), 1)
        avg_ship_g = round(float(grp["shipping_delay"].mean()), 1)
        docs.append(
            f"{wh} ships {prod}: avg processing {avg_proc_g}d, avg shipping {avg_ship_g}d "
            f"({len(grp)} order(s))."
        )

    return docs


class RAGEngine:
    """
    Retrieval-Augmented Generation engine.

    Usage::

        engine = RAGEngine()
        engine.build_index(df)           # once per dataset
        answer = engine.answer("Which warehouse is slowest?")
    """

    def __init__(self) -> None:
        self._encoder: SentenceTransformer | None = None
        self._groq_client: Groq | None = None
        self._index: faiss.Index | None = None
        self._documents: list[str] = []


    def _get_encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def _get_groq(self) -> Groq:
        if not is_groq_configured():
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or set it as an environment variable."
            )
        if self._groq_client is None:
            self._groq_client = Groq(api_key=GROQ_API_KEY)
        return self._groq_client


    def build_index(self, df: pd.DataFrame) -> None:
        """Embed all analytics documents and store them in a FAISS index."""
        self._documents = build_documents(df)
        encoder = self._get_encoder()

        embeddings = encoder.encode(self._documents, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)   
        self._index.add(embeddings)

    def is_ready(self) -> bool:
        """True if the index has been built and contains at least one document."""
        return self._index is not None and len(self._documents) > 0


    def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        """Return the top-k most relevant document chunks for *query*."""
        if not self.is_ready():
            return []

        k = top_k if top_k is not None else RAG_TOP_K
        encoder = self._get_encoder()

        q_emb = encoder.encode([query], show_progress_bar=False)
        q_emb = np.array(q_emb, dtype=np.float32)
        faiss.normalize_L2(q_emb)

        _, indices = self._index.search(q_emb, k)
        return [self._documents[i] for i in indices[0] if i < len(self._documents)]


    def answer(self, question: str) -> str:
        """
        Full RAG pipeline: retrieve → augment prompt → call Groq → return answer.
        """
        context_chunks = self.retrieve(question)
        context = "\n".join(f"- {chunk}" for chunk in context_chunks)

        system_prompt = (
            "You are a Supply Chain AI Copilot. "
            "Analyse logistics data and answer questions about shipping delays, "
            "warehouse performance, and product delivery times. "
            "Use ONLY the provided context. Be concise and data-driven. "
            "If the context lacks enough information, say so clearly."
        )

        user_prompt = (
            f"Context from supply chain analytics:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Provide a clear, insightful answer based on the context above."
        )

        client = self._get_groq()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()

    def generate_insights_narrative(self, df: pd.DataFrame) -> str:
        """Generate a Groq-powered executive summary from auto insights."""
        bullets = "\n".join(generate_auto_insights(df))

        client = self._get_groq()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior supply chain analyst. "
                        "Write a concise executive summary (3–5 sentences) from the bullet points provided. "
                        "Be direct, data-driven, and suggest one actionable recommendation."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Supply chain insights:\n{bullets}\n\nWrite the executive summary.",
                },
            ],
            temperature=0.4,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
