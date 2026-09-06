"""
rag_engine.py
─────────────
Hybrid Retrieval-Augmented Generation pipeline for the
Supply Chain AI Copilot.

INDEXING
--------
Analytics dataframe
        ↓
Structured retrieval documents + metadata
        ↓
SentenceTransformer embeddings ───────┐
        ↓                             │
Pinecone vector database              │
                                      │
BM25 lexical index ───────────────────┤
                                      ↓
QUERY
----
User question
        ↓
Deterministic metadata/entity detection
        ↓
BM25 + Pinecone semantic retrieval
        ↓
Reciprocal Rank Fusion
        ↓
Cross-Encoder reranking
        ↓
Grounded context
        ↓
Groq LLM
        ↓
Final answer

The implementation deliberately keeps retrieval deterministic.
The LLM is only responsible for generating the final grounded answer.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from groq import Groq
from pinecone import Pinecone, ServerlessSpec

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    EMBEDDING_MODEL,
    RAG_TOP_K,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    DELAY_THRESHOLD_DAYS,
    is_groq_configured,

    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    PINECONE_NAMESPACE,

    BM25_TOP_K,
    SEMANTIC_TOP_K,
    RERANK_TOP_K,
    RRF_K,

    RERANKER_MODEL,
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


# ============================================================================
# Device
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class RetrievedDocument:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float = 0.0
    source: str = ""


@dataclass
class DocumentRecord:
    text: str
    metadata: dict[str, Any]


# ============================================================================
# Text utilities
# ============================================================================

def _tokenize(text: str) -> list[str]:
    """
    Lightweight tokenizer for BM25.

    Keeps warehouse/product/destination names searchable while normalizing
    case.
    """
    return re.findall(
        r"\b[a-zA-Z0-9][a-zA-Z0-9_.-]*\b",
        text.lower(),
    )


def _make_document_id(text: str, index: int) -> str:
    """
    Generate deterministic document IDs.
    """
    digest = hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()[:16]

    return f"sc-{index}-{digest}"


# ============================================================================
# Document construction
# ============================================================================

def _build_document_records(
    df: pd.DataFrame,
) -> list[DocumentRecord]:
    """
    Build retrieval documents together with their metadata.

    Metadata is created directly from the source values instead of attempting
    to infer entities by searching generated text.
    """

    records: list[DocumentRecord] = []

    if df.empty:
        return records

    total = len(df)

    # ------------------------------------------------------------------
    # Warehouse processing performance
    # ------------------------------------------------------------------

    processing = avg_processing_time_per_warehouse(df)

    for _, row in processing.iterrows():

        warehouse = str(row["Warehouse"])

        text = (
            f"Warehouse performance. "
            f"Warehouse: {warehouse}. "
            f"Metric: order processing time. "
            f"Average order processing time: "
            f"{row['avg_processing_time']} days. "
            f"Definition: time from order received to dispatched."
        )

        records.append(
            DocumentRecord(
                text=text,
                metadata={
                    "warehouse": warehouse,
                    "document_type": "warehouse_performance",
                    "metric": "processing_time",
                    "source": "supply_chain_analytics",
                },
            )
        )

    # ------------------------------------------------------------------
    # Warehouse shipping delay
    # ------------------------------------------------------------------

    shipping = avg_shipping_delay_per_warehouse(df)

    for _, row in shipping.iterrows():

        warehouse = str(row["Warehouse"])

        text = (
            f"Warehouse performance. "
            f"Warehouse: {warehouse}. "
            f"Metric: shipping delay. "
            f"Average shipping delay: "
            f"{row['avg_shipping_delay']} days. "
            f"Definition: time from dispatched to delivered."
        )

        records.append(
            DocumentRecord(
                text=text,
                metadata={
                    "warehouse": warehouse,
                    "document_type": "warehouse_performance",
                    "metric": "shipping_delay",
                    "source": "supply_chain_analytics",
                },
            )
        )

    # ------------------------------------------------------------------
    # Warehouse total lead time
    # ------------------------------------------------------------------

    distribution = delay_distribution_per_warehouse(df)

    for _, row in distribution.iterrows():

        warehouse = str(row["Warehouse"])

        text = (
            f"Warehouse performance. "
            f"Warehouse: {warehouse}. "
            f"Metric: total lead time. "
            f"Average total lead time: "
            f"{row['avg_total_lead_time']} days. "
            f"Average processing time: "
            f"{row['avg_processing_time']} days. "
            f"Average shipping delay: "
            f"{row['avg_shipping_delay']} days. "
            f"Total orders: {row['total_orders']}."
        )

        records.append(
            DocumentRecord(
                text=text,
                metadata={
                    "warehouse": warehouse,
                    "document_type": "warehouse_performance",
                    "metric": "total_lead_time",
                    "source": "supply_chain_analytics",
                },
            )
        )

    # ------------------------------------------------------------------
    # COMPLETE WAREHOUSE COMPARISON
    #
    # This is critical for questions such as:
    # "Which warehouse has the highest average delay?"
    # "Compare all warehouses."
    # ------------------------------------------------------------------

    if not shipping.empty:

        warehouse_lines = []

        for _, row in shipping.iterrows():

            warehouse_lines.append(
                f"Warehouse {row['Warehouse']}: "
                f"average shipping delay "
                f"{row['avg_shipping_delay']} days."
            )

        comparison_text = (
            "Warehouse comparison. "
            "Complete warehouse shipping-delay comparison: "
            + " ".join(warehouse_lines)
        )

        records.append(
            DocumentRecord(
                text=comparison_text,
                metadata={
                    "document_type": "warehouse_comparison",
                    "comparison_scope": "all_warehouses",
                    "source": "supply_chain_analytics",
                },
            )
        )

    # ------------------------------------------------------------------
    # COMPLETE WAREHOUSE LEAD-TIME COMPARISON
    # ------------------------------------------------------------------

    if not distribution.empty:

        warehouse_lines = []

        for _, row in distribution.iterrows():

            warehouse_lines.append(
                f"Warehouse {row['Warehouse']}: "
                f"processing {row['avg_processing_time']} days, "
                f"shipping delay {row['avg_shipping_delay']} days, "
                f"total lead time {row['avg_total_lead_time']} days, "
                f"{row['total_orders']} orders."
            )

        comparison_text = (
            "Warehouse comparison. "
            "Complete warehouse performance comparison: "
            + " ".join(warehouse_lines)
        )

        records.append(
            DocumentRecord(
                text=comparison_text,
                metadata={
                    "document_type": "warehouse_comparison",
                    "comparison_scope": "all_warehouses",
                    "source": "supply_chain_analytics",
                },
            )
        )

    # ------------------------------------------------------------------
    # Product shipping performance
    # ------------------------------------------------------------------

    product_df = fastest_shipping_product(df)

    if not product_df.empty:

        median_ship = float(
            product_df["avg_shipping_delay"].median()
        )

        for _, row in product_df.iterrows():

            product = str(row["Product"])

            speed = (
                "fast"
                if row["avg_shipping_delay"] <= median_ship
                else "slow"
            )

            text = (
                f"Product shipping performance. "
                f"Product: {product}. "
                f"Average shipping delay: "
                f"{row['avg_shipping_delay']} days. "
                f"Relative shipping classification: "
                f"{speed}-shipping product."
            )

            records.append(
                DocumentRecord(
                    text=text,
                    metadata={
                        "product": product,
                        "document_type": "product_performance",
                        "source": "supply_chain_analytics",
                    },
                )
            )

    # ------------------------------------------------------------------
    # Overall delayed orders
    # ------------------------------------------------------------------

    delayed = orders_delayed_more_than_n(df)

    pct = round(
        len(delayed) / total * 100,
        1,
    )

    records.append(
        DocumentRecord(
            text=(
                f"Overall delay performance. "
                f"Delayed orders: {len(delayed)} out of {total}. "
                f"Delayed order percentage: {pct}%. "
                f"Threshold: shipping delay greater than "
                f"{DELAY_THRESHOLD_DAYS} days."
            ),
            metadata={
                "document_type": "delay_summary",
                "source": "supply_chain_analytics",
            },
        )
    )

    # ------------------------------------------------------------------
    # Warehouse-specific delayed orders
    # ------------------------------------------------------------------

    if (
        "Warehouse" in df.columns
        and "shipping_delay" in df.columns
    ):

        for warehouse, group in df.groupby("Warehouse"):

            warehouse = str(warehouse)

            delayed_wh = group[
                group["shipping_delay"]
                > DELAY_THRESHOLD_DAYS
            ]

            percentage = round(
                len(delayed_wh) / len(group) * 100,
                1,
            )

            records.append(
                DocumentRecord(
                    text=(
                        f"Warehouse delay analysis. "
                        f"Warehouse: {warehouse}. "
                        f"Orders with shipping delay above "
                        f"{DELAY_THRESHOLD_DAYS} days: "
                        f"{len(delayed_wh)}. "
                        f"Delayed order percentage: "
                        f"{percentage}%. "
                        f"Total orders: {len(group)}."
                    ),
                    metadata={
                        "warehouse": warehouse,
                        "document_type": "warehouse_delay",
                        "source": "supply_chain_analytics",
                    },
                )
            )

    # ------------------------------------------------------------------
    # Destination performance
    # ------------------------------------------------------------------

    top_dest = top_delayed_destinations(
        df,
        5,
    )

    if not top_dest.empty:

        for _, row in top_dest.iterrows():

            destination = str(
                row["Destination"]
            )

            records.append(
                DocumentRecord(
                    text=(
                        f"Destination shipping performance. "
                        f"Destination: {destination}. "
                        f"Average shipping delay: "
                        f"{row['avg_shipping_delay']} days."
                    ),
                    metadata={
                        "destination": destination,
                        "document_type": "destination_performance",
                        "source": "supply_chain_analytics",
                    },
                )
            )

        destination_summary = ", ".join(
            f"{row['Destination']} "
            f"({row['avg_shipping_delay']} days)"
            for _, row in top_dest.iterrows()
        )

        records.append(
            DocumentRecord(
                text=(
                    "Top delayed destinations by average shipping delay: "
                    f"{destination_summary}."
                ),
                metadata={
                    "document_type": "destination_comparison",
                    "comparison_scope": "top_destinations",
                    "source": "supply_chain_analytics",
                },
            )
        )

    # ------------------------------------------------------------------
    # Monthly volume
    # ------------------------------------------------------------------

    monthly = monthly_order_volume(df)

    if not monthly.empty:

        peak = monthly.loc[
            monthly["order_count"].idxmax()
        ]

        records.append(
            DocumentRecord(
                text=(
                    f"Monthly order volume. "
                    f"Peak order volume occurred in "
                    f"{peak['month']} with "
                    f"{peak['order_count']} orders. "
                    f"Dataset covers {len(monthly)} months."
                ),
                metadata={
                    "document_type": "monthly_volume",
                    "source": "supply_chain_analytics",
                },
            )
        )

        for _, row in monthly.iterrows():

            records.append(
                DocumentRecord(
                    text=(
                        f"Monthly order volume. "
                        f"Month: {row['month']}. "
                        f"Orders: {row['order_count']}."
                    ),
                    metadata={
                        "document_type": "monthly_volume",
                        "month": str(row["month"]),
                        "source": "supply_chain_analytics",
                    },
                )
            )

    # ------------------------------------------------------------------
    # Overall averages
    # ------------------------------------------------------------------

    avg_proc = round(
        float(
            df["order_processing_time"].mean()
        ),
        2,
    )

    avg_ship = round(
        float(
            df["shipping_delay"].mean()
        ),
        2,
    )

    avg_lead = round(
        float(
            df["total_lead_time"].mean()
        ),
        2,
    )

    records.append(
        DocumentRecord(
            text=(
                "Overall supply chain performance. "
                f"Average order processing time: {avg_proc} days. "
                f"Average shipping delay: {avg_ship} days. "
                f"Average total lead time: {avg_lead} days. "
                f"Total orders analyzed: {total}."
            ),
            metadata={
                "document_type": "overall",
                "source": "supply_chain_analytics",
            },
        )
    )

    # ------------------------------------------------------------------
    # Warehouse × Product performance
    # ------------------------------------------------------------------

    if (
        "Warehouse" in df.columns
        and "Product" in df.columns
    ):

        grouped = df.groupby(
            ["Warehouse", "Product"]
        )

        for (
            warehouse,
            product,
        ), group in grouped:

            warehouse = str(warehouse)
            product = str(product)

            avg_proc_group = round(
                float(
                    group[
                        "order_processing_time"
                    ].mean()
                ),
                1,
            )

            avg_ship_group = round(
                float(
                    group[
                        "shipping_delay"
                    ].mean()
                ),
                1,
            )

            records.append(
                DocumentRecord(
                    text=(
                        "Warehouse-product performance. "
                        f"Warehouse: {warehouse}. "
                        f"Product: {product}. "
                        f"Average processing time: "
                        f"{avg_proc_group} days. "
                        f"Average shipping delay: "
                        f"{avg_ship_group} days. "
                        f"Number of orders: "
                        f"{len(group)}."
                    ),
                    metadata={
                        "warehouse": warehouse,
                        "product": product,
                        "document_type": "warehouse_product",
                        "source": "supply_chain_analytics",
                    },
                )
            )

    # ------------------------------------------------------------------
    # COMPLETE PRODUCT × WAREHOUSE COMPARISON
    #
    # This specifically fixes:
    # "Show me the delay breakdown by product and warehouse."
    # ------------------------------------------------------------------

    if (
        "Warehouse" in df.columns
        and "Product" in df.columns
        and "shipping_delay" in df.columns
    ):

        grouped = (
            df.groupby(
                ["Warehouse", "Product"]
            )["shipping_delay"]
            .agg(["mean", "count"])
            .reset_index()
        )

        comparison_lines = []

        for _, row in grouped.iterrows():

            comparison_lines.append(
                f"Warehouse {row['Warehouse']} "
                f"- Product {row['Product']}: "
                f"average shipping delay "
                f"{round(float(row['mean']), 1)} days "
                f"across {int(row['count'])} orders."
            )

        comparison_text = (
            "Product and warehouse delay breakdown. "
            "Complete Warehouse × Product shipping-delay comparison: "
            + " ".join(comparison_lines)
        )

        records.append(
            DocumentRecord(
                text=comparison_text,
                metadata={
                    "document_type": "product_warehouse_comparison",
                    "comparison_scope": "all_warehouses_all_products",
                    "source": "supply_chain_analytics",
                },
            )
        )

    return records


def build_documents(
    df: pd.DataFrame,
) -> list[str]:
    """
    Backward-compatible helper returning only document text.
    """

    return [
        record.text
        for record in _build_document_records(df)
    ]


# ============================================================================
# RAG Engine
# ============================================================================

class RAGEngine:
    """
    Hybrid RAG engine.

    BM25
       +
    Pinecone semantic retrieval
       ↓
    RRF fusion
       ↓
    CrossEncoder reranking
       ↓
    Groq GPT generation
    """

    def __init__(self) -> None:

        self._encoder: SentenceTransformer | None = None
        self._reranker: CrossEncoder | None = None
        self._groq_client: Groq | None = None

        self._pinecone: Pinecone | None = None
        self._pinecone_index: Any | None = None

        self._bm25: BM25Okapi | None = None

        self._documents: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        self._document_ids: list[str] = []

        self._id_to_document: dict[
            str,
            RetrievedDocument,
        ] = {}

        self._dimension: int | None = None

    # ========================================================================
    # Models
    # ========================================================================

    def _get_encoder(
        self,
    ) -> SentenceTransformer:

        if self._encoder is None:

            self._encoder = SentenceTransformer(
                EMBEDDING_MODEL,
                device=DEVICE,
            )

        return self._encoder

    def _get_reranker(
        self,
    ) -> CrossEncoder:

        if self._reranker is None:

            self._reranker = CrossEncoder(
                RERANKER_MODEL,
                device=DEVICE,
            )

        return self._reranker

    def _get_groq(self) -> Groq:

        if not is_groq_configured():

            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or set it as "
                "an environment variable."
            )

        if self._groq_client is None:

            self._groq_client = Groq(
                api_key=GROQ_API_KEY,
            )

        return self._groq_client

    # ========================================================================
    # Pinecone
    # ========================================================================

    def _get_pinecone(self) -> Pinecone:

        if not PINECONE_API_KEY:

            raise EnvironmentError(
                "PINECONE_API_KEY is not configured. "
                "Add it to .env."
            )

        if self._pinecone is None:

            self._pinecone = Pinecone(
                api_key=PINECONE_API_KEY,
            )

        return self._pinecone

    def _get_pinecone_index(
        self,
        dimension: int,
    ) -> Any:

        if self._pinecone_index is not None:
            return self._pinecone_index

        pc = self._get_pinecone()

        existing_indexes = (
            pc.list_indexes().names()
        )

        if PINECONE_INDEX_NAME not in existing_indexes:

            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION,
                ),
            )

        self._pinecone_index = pc.Index(
            PINECONE_INDEX_NAME
        )

        return self._pinecone_index

    # ========================================================================
    # Index building
    # ========================================================================

    def build_index(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Build the complete hybrid retrieval index.
        """

        records = _build_document_records(df)

        self._documents = [
            record.text
            for record in records
        ]

        self._metadata = [
            record.metadata
            for record in records
        ]

        self._document_ids = []
        self._id_to_document = {}

        if not self._documents:

            self._bm25 = None
            return

        # --------------------------------------------------------------------
        # BM25
        # --------------------------------------------------------------------

        tokenized_documents = [
            _tokenize(document)
            for document in self._documents
        ]

        self._bm25 = BM25Okapi(
            tokenized_documents
        )

        # --------------------------------------------------------------------
        # Dense embeddings
        # --------------------------------------------------------------------

        encoder = self._get_encoder()

        embeddings = encoder.encode(
            self._documents,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        norms[norms == 0] = 1.0

        embeddings = (
            embeddings / norms
        )

        dimension = int(
            embeddings.shape[1]
        )

        self._dimension = dimension

        # --------------------------------------------------------------------
        # Pinecone
        # --------------------------------------------------------------------

        index = self._get_pinecone_index(
            dimension
        )

        # Clear this application's namespace before rebuilding.
        #
        # This prevents stale vectors from previous CSV uploads from being
        # mixed with the current dataset.
        try:
            index.delete(
                delete_all=True,
                namespace=PINECONE_NAMESPACE,
            )
        except Exception:
            pass

        vectors = []

        for (
            i,
            (document, embedding, metadata),
        ) in enumerate(
            zip(
                self._documents,
                embeddings,
                self._metadata,
            )
        ):

            document_id = _make_document_id(
                document,
                i,
            )

            self._document_ids.append(
                document_id
            )

            self._id_to_document[
                document_id
            ] = RetrievedDocument(
                id=document_id,
                text=document,
                metadata=metadata,
            )

            vectors.append(
                {
                    "id": document_id,
                    "values": embedding.tolist(),
                    "metadata": metadata,
                }
            )

        # --------------------------------------------------------------------
        # Batched upsert
        # --------------------------------------------------------------------

        batch_size = 100

        for start in range(
            0,
            len(vectors),
            batch_size,
        ):

            index.upsert(
                vectors=vectors[
                    start:start + batch_size
                ],
                namespace=PINECONE_NAMESPACE,
            )

    # ========================================================================
    # Readiness
    # ========================================================================

    def is_ready(self) -> bool:

        return (
            self._bm25 is not None
            and len(self._documents) > 0
        )

    # ========================================================================
    # Query intent
    # ========================================================================

    @staticmethod
    def _is_comparison_query(
        query: str,
    ) -> bool:
        """
        Detect queries where retrieving a single entity is insufficient.

        These queries require broader context so the answer can compare
        warehouses/products rather than describing whichever chunk happened
        to rank highest.
        """

        q = query.lower()

        comparison_terms = [
            "compare",
            "comparison",
            "across all",
            "all warehouses",
            "all products",
            "breakdown",
            "by warehouse",
            "by product",
            "warehouse and product",
            "product and warehouse",
            "highest",
            "lowest",
            "best",
            "worst",
            "fastest",
            "slowest",
            "rank",
            "ranking",
            "difference",
            "differences",
        ]

        return any(
            term in q
            for term in comparison_terms
        )

    # ========================================================================
    # Metadata filtering
    # ========================================================================

    def _extract_metadata_filters(
        self,
        query: str,
    ) -> dict[str, Any] | None:
        """
        Extract explicit entity filters from the user's query.

        IMPORTANT:
        Generic comparison questions do not get an inferred warehouse filter.

        Example:

            "How is Warehouse C performing?"
                -> {"warehouse": "Warehouse C"}

            "Compare all warehouses."
                -> None
        """

        if not self._metadata:
            return None

        # Never restrict a comparison query to one accidentally matched entity.
        if self._is_comparison_query(query):
            return None

        filters: dict[str, Any] = {}

        query_lower = query.lower()

        for metadata_key in (
            "warehouse",
            "product",
            "destination",
        ):

            values = sorted(
                {
                    str(item[metadata_key])
                    for item in self._metadata
                    if metadata_key in item
                },
                key=len,
                reverse=True,
            )

            for value in values:

                # Word-boundary matching prevents accidental substring matches.
                pattern = (
                    rf"(?<!\w)"
                    rf"{re.escape(value.lower())}"
                    rf"(?!\w)"
                )

                if re.search(
                    pattern,
                    query_lower,
                ):

                    filters[
                        metadata_key
                    ] = value

                    break

        return filters or None

    # ========================================================================
    # BM25
    # ========================================================================

    def _bm25_retrieve(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:

        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(
            _tokenize(query)
        )

        ranked_indices = np.argsort(
            scores
        )[::-1]

        results: list[
            RetrievedDocument
        ] = []

        for idx in ranked_indices:

            idx = int(idx)

            if idx >= len(
                self._documents
            ):
                continue

            metadata = self._metadata[
                idx
            ]

            if metadata_filter:

                if not self._matches_filter(
                    metadata,
                    metadata_filter,
                ):
                    continue

            results.append(
                RetrievedDocument(
                    id=self._document_ids[idx],
                    text=self._documents[idx],
                    metadata=metadata,
                    score=float(
                        scores[idx]
                    ),
                    source="bm25",
                )
            )

            if len(results) >= top_k:
                break

        return results

    # ========================================================================
    # Pinecone semantic retrieval
    # ========================================================================

    def _semantic_retrieve(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:

        if self._dimension is None:
            return []

        encoder = self._get_encoder()

        query_embedding = encoder.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        norm = np.linalg.norm(
            query_embedding,
            axis=1,
            keepdims=True,
        )

        norm[norm == 0] = 1.0

        query_embedding = (
            query_embedding / norm
        )

        index = self._get_pinecone_index(
            self._dimension
        )

        kwargs: dict[str, Any] = {
            "vector": query_embedding[0].tolist(),
            "top_k": top_k,
            "include_metadata": True,
            "namespace": PINECONE_NAMESPACE,
        }

        if metadata_filter:
            kwargs["filter"] = metadata_filter

        response = index.query(
            **kwargs
        )

        results: list[
            RetrievedDocument
        ] = []

        matches = getattr(
            response,
            "matches",
            [],
        )

        for match in matches:

            match_id = getattr(
                match,
                "id",
                None,
            )

            if (
                match_id is None
                or match_id
                not in self._id_to_document
            ):
                continue

            original = (
                self._id_to_document[
                    match_id
                ]
            )

            results.append(
                RetrievedDocument(
                    id=original.id,
                    text=original.text,
                    metadata=original.metadata,
                    score=float(
                        getattr(
                            match,
                            "score",
                            0.0,
                        )
                    ),
                    source="pinecone",
                )
            )

        return results

    # ========================================================================
    # Metadata matching
    # ========================================================================

    @staticmethod
    def _matches_filter(
        metadata: dict[str, Any],
        metadata_filter: dict[str, Any],
    ) -> bool:

        for key, expected in (
            metadata_filter.items()
        ):

            if str(
                metadata.get(key, "")
            ).lower() != str(
                expected
            ).lower():

                return False

        return True

    # ========================================================================
    # RRF
    # ========================================================================

    def _rrf_fusion(
        self,
        result_lists: list[
            list[RetrievedDocument]
        ],
    ) -> list[RetrievedDocument]:
        """
        Reciprocal Rank Fusion.
        """

        fused_scores: dict[
            str,
            float,
        ] = {}

        documents: dict[
            str,
            RetrievedDocument,
        ] = {}

        for results in result_lists:

            for rank, document in enumerate(
                results,
                start=1,
            ):

                documents[
                    document.id
                ] = document

                fused_scores[
                    document.id
                ] = (
                    fused_scores.get(
                        document.id,
                        0.0,
                    )
                    + 1.0
                    / (
                        RRF_K + rank
                    )
                )

        ranked_ids = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )

        output = []

        for document_id in ranked_ids:

            document = documents[
                document_id
            ]

            document.score = fused_scores[
                document_id
            ]

            document.source = "hybrid"

            output.append(
                document
            )

        return output

    # ========================================================================
    # CrossEncoder
    # ========================================================================

    def _rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int,
    ) -> list[RetrievedDocument]:

        if not documents:
            return []

        reranker = self._get_reranker()

        pairs = [
            [
                query,
                document.text,
            ]
            for document in documents
        ]

        scores = reranker.predict(
            pairs,
            show_progress_bar=False,
        )

        scores = np.asarray(
            scores
        ).reshape(-1)

        reranked = []

        for (
            document,
            score,
        ) in zip(
            documents,
            scores,
        ):

            reranked.append(
                RetrievedDocument(
                    id=document.id,
                    text=document.text,
                    metadata=document.metadata,
                    score=float(score),
                    source="reranker",
                )
            )

        reranked.sort(
            key=lambda x: x.score,
            reverse=True,
        )

        return reranked[:top_k]

    # ========================================================================
    # Public retrieval
    # ========================================================================

    def retrieve_with_metadata(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:

        if not self.is_ready():
            return []

        final_k = (
            top_k
            if top_k is not None
            else RAG_TOP_K
        )

        metadata_filter = (
            self._extract_metadata_filters(
                query
            )
        )

        comparison_query = (
            self._is_comparison_query(
                query
            )
        )

        bm25_results = (
            self._bm25_retrieve(
                query,
                top_k=BM25_TOP_K,
                metadata_filter=metadata_filter,
            )
        )

        semantic_results = (
            self._semantic_retrieve(
                query,
                top_k=SEMANTIC_TOP_K,
                metadata_filter=metadata_filter,
            )
        )

        hybrid_results = self._rrf_fusion(
            [
                bm25_results,
                semantic_results,
            ]
        )

        # ------------------------------------------------------------------
        # For comparison questions, explicitly promote complete comparison
        # documents. This prevents a collection of Warehouse C documents
        # from crowding out the all-warehouse evidence.
        # ------------------------------------------------------------------

        if comparison_query:

            comparison_types = {
                "warehouse_comparison",
                "product_warehouse_comparison",
                "destination_comparison",
            }

            comparison_documents = [
                document
                for document in self._id_to_document.values()
                if document.metadata.get(
                    "document_type"
                ) in comparison_types
            ]

            query_tokens = set(
                _tokenize(query)
            )

            scored_comparisons = []

            for document in comparison_documents:

                doc_tokens = set(
                    _tokenize(
                        document.text
                    )
                )

                lexical_overlap = len(
                    query_tokens
                    & doc_tokens
                )

                scored_comparisons.append(
                    (
                        lexical_overlap,
                        document,
                    )
                )

            scored_comparisons.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            # Put complete comparison evidence first.
            promoted_ids = {
                document.id
                for _, document
                in scored_comparisons[:2]
            }

            promoted = [
                document
                for _, document
                in scored_comparisons[:2]
            ]

            remaining = [
                document
                for document
                in hybrid_results
                if document.id
                not in promoted_ids
            ]

            hybrid_results = (
                promoted + remaining
            )

        rerank_candidates = hybrid_results[
            :max(
                RERANK_TOP_K,
                final_k,
            )
        ]

        return self._rerank(
            query,
            rerank_candidates,
            top_k=final_k,
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[str]:

        return [
            document.text
            for document in self.retrieve_with_metadata(
                query,
                top_k=top_k,
            )
        ]

    # ========================================================================
    # Generation
    # ========================================================================

    def answer(
        self,
        question: str,
    ) -> str:

        retrieved_documents = (
            self.retrieve_with_metadata(
                question
            )
        )

        if not retrieved_documents:

            return (
                "I could not find enough relevant "
                "supply-chain information in the "
                "available data to answer that question."
            )

        context = "\n".join(
            f"- {document.text}"
            for document in retrieved_documents
        )

        system_prompt = (
            "You are a Supply Chain AI Copilot. "
            "You analyse logistics, shipment, warehouse, "
            "product, destination, and order-performance data.\n\n"

            "Grounding rules:\n"
            "1. Use ONLY the supplied context.\n"
            "2. Do not invent numbers, warehouses, products, "
            "destinations, or trends.\n"
            "3. If the context is insufficient, say so clearly.\n"
            "4. Preserve numerical units from the context.\n"
            "5. For comparison questions, compare every relevant "
            "entity present in the supplied comparison context.\n"
            "6. Do not claim that only one warehouse or product is "
            "available merely because other retrieved chunks are "
            "more detailed.\n"
            "7. Prefer concise, decision-oriented answers.\n"
            "8. When useful, explain the operational implication.\n"
            "9. When presenting tabular data, use a valid Markdown table "
            "with exactly one header row and one separator row.\n"
            "10. Keep every table column separated by the | character.\n"
            "11. Never output HTML tags such as <div>, </div>, <table>, "
            "<tr>, or <td>."
            "12. For a product-and-warehouse delay breakdown, always use exactly "
            "this column structure: Warehouse | Product | Avg. Shipping Delay (days) | Orders. "
            "Create one row for every Warehouse × Product combination present in the "
            "retrieved context. Do not transpose the table into a matrix.\n"
        )
        user_prompt = (
            f"Retrieved supply-chain context:\n"
            f"{context}\n\n"
            f"User question:\n"
            f"{question}\n\n"
            "Answer using only the retrieved context."
        )

        client = self._get_groq()

        response = (
            client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
        )

        if not response.choices:
            return "The model returned an empty response."

        content = (
            response.choices[0]
            .message
            .content
        )

        return self._clean_llm_output(content)

    # ========================================================================
    # LLM output cleaning
    # ========================================================================

    def _clean_llm_output(self, text: str) -> str:
        """
        Clean accidental HTML and Markdown code fences from the LLM
        response without altering the analytical content.

        Also un-escapes backslash-escaped Markdown markers that Groq
        sometimes emits (e.g. \\*\\*bold\\*\\* → **bold**).
        """
        if not text:
            return "The model returned an empty response."

        # Remove accidental HTML tags such as </div></div>.
        text = re.sub(r"<[^>]+>", "", text)

        # Remove accidental Markdown code fences.
        text = re.sub(
            r"```(?:markdown|md)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s*```", "", text)

        # Un-escape backslash-escaped Markdown emphasis markers.
        text = re.sub(r"\\\*\\\*(.+?)\\\*\\\*", r"**\1**", text)
        text = re.sub(r"\\\*(.+?)\\\*", r"*\1*", text)

        return text.strip()

    # ========================================================================
    # Executive insights
    # ========================================================================

    def generate_insights_narrative(
        self,
        df: pd.DataFrame,
    ) -> str:

        bullets = "\n".join(
            generate_auto_insights(df)
        )

        client = self._get_groq()

        response = (
            client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior supply chain analyst. "
                            "Write a concise executive summary in "
                            "3–5 sentences from the supplied analytics. "
                            "Be direct, data-driven, and provide one "
                            "actionable operational recommendation. "
                            "Do not invent information."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Supply chain insights:\n"
                            f"{bullets}\n\n"
                            "Write the executive summary."
                        ),
                    },
                ],
                temperature=0.4,
                max_tokens=300,
            )
        )

        if not response.choices:
            return "Unable to generate the executive summary."

        content = (
            response.choices[0]
            .message
            .content
        )

        return self._clean_llm_output(content) or "Unable to generate the executive summary."