# ⛓ Supply Chain AI Copilot

> Intelligent logistics analytics using Hybrid RAG · BM25 · Pinecone · CrossEncoder Reranking · Groq LLM · Streamlit

---

## Project Overview

Supply Chain AI Copilot transforms raw logistics CSV data into actionable intelligence.

Users ask natural-language questions about orders, warehouses, products, destinations, shipping delays, and logistics performance. The system combines lexical and semantic retrieval to identify relevant analytics from the dataset, reranks the retrieved results, and passes the most relevant context to a Groq-hosted large language model for data-grounded answers.

The application also provides interactive analytics, charts, and a filterable data explorer through a Streamlit interface.

---


## Architecture

```
supply_chain_ai_copilot/
│
├── app.py                        # Entry point — page config, routing only
│
├── pages/                        # One module per UI section
│   ├── sidebar.py                # Dataset upload + stats sidebar
│   ├── chat_page.py              # AI chat tab
│   ├── analytics_page.py         # Charts + KPI tables tab
│   └── explorer_page.py          # Filterable raw-data tab
│
├── src/                          # Business logic (framework-agnostic)
│   ├── config.py                 # All settings loaded from .env
│   ├── data_processing.py        # CSV loading, date parsing, delay_days
│   ├── analytics.py              # Pure KPI functions (no Streamlit)
│   ├── rag_engine.py             # FAISS index + Groq LLM pipeline
│   └── ui/
│       ├── styles.py             # All custom CSS (injected once)
│       ├── components.py         # Reusable HTML/st component builders
│       └── charts.py             # Plotly chart factories + intent detection
│
├── data/
│   └── sample_dataset.csv        # 1000-row bundled dataset (synthetically generated)
│
├── .env                          
├── .gitignore
├── requirements.txt
└── README.md
```

### Data flow

```
                         CSV Dataset
                              │
                              ▼
                    data_processing.py
                              │
                     Clean DataFrame
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
        analytics.py                     rag_engine.py
              │                               │
        KPI calculations              Document generation
              │                               │
              ▼                         ┌─────┴─────┐
        ui/charts.py                     │           │
              │                       BM25       Embeddings
              │                     Retrieval    + Pinecone
              │                       │           │
              │                       └─────┬─────┘
              │                             │
              │                        RRF Fusion
              │                             │
              │                       CrossEncoder
              │                         Reranking
              │                             │
              └──────────────┐              ▼
                             │        Relevant Context
                             │              │
                             │              ▼
                             │          Groq LLM
                             │              │
                             └──────► Grounded Answer
                                            │
                                            ▼
                                     Streamlit Chat
```

---

## Setup

### 1. Clone / unzip

```bash
cd supply_chain_ai_copilot
```

### 2. Virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set your Groq API key:

```dotenv
GROQ_API_KEY=gsk_your_key_here
```

Optional overrides (all have sensible defaults):

```dotenv
GROQ_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=all-MiniLM-L6-v2
RAG_TOP_K=6
DELAY_THRESHOLD_DAYS=3
```

```Structure
# ============================================================
# Groq
# ============================================================

GROQ_API_KEY=gsk_****************************************************
GROQ_MODEL=open**************

LLM_MAX_TOKENS=2048
LLM_TEMPERATURE=0.0


# ============================================================
# Embeddings
# ============================================================

EMBEDDING_MODEL=all-************


# ============================================================
# Pinecone
# ============================================================

PINECONE_API_KEY=pcsk***********************************************************************
PINECONE_INDEX_NAME=supp****************
PINECONE_CLOUD=aws
PINECONE_REGION=us-e*****
PINECONE_NAMESPACE=supp********


# ============================================================
# Retrieval
# ============================================================

BM25_TOP_K=20
SEMANTIC_TOP_K=20
RERANK_TOP_K=8
RAG_TOP_K=6

RRF_K=60


# ============================================================
# Reranker
# ============================================================

RERANKER_MODEL=cros*******************************


# ============================================================
# Analytics
# ============================================================

DELAY_THRESHOLD_DAYS=3
TOP_DESTINATIONS_N=10
```

### 5. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

---

## Example Questions

| Question | Relevant chart |
|---|---|
| Which warehouse has the highest delay? | Bar chart — warehouse delay |
| What is the fastest shipping product? | Horizontal bar — product delay |
| How many orders were delayed more than 3 days? | — |
| Which destination has the worst shipping time? | Top-destination bar |
| Show me the delay breakdown | Heatmap: Warehouse × Product |
| What's the monthly order trend? | Line chart — volume |
| What is the overall average delay? | — |

---

## Tech Stack

| Component          | Technology                      |
| ------------------ | ------------------------------- |
| UI                 | Streamlit                       |
| Data Processing    | Pandas                          |
| Visualization      | Plotly                          |
| Lexical Retrieval  | BM25                            |
| Semantic Retrieval | SentenceTransformers + Pinecone |
| Retrieval Fusion   | Reciprocal Rank Fusion (RRF)    |
| Reranking          | CrossEncoder                    |
| LLM                | Groq · `openai/gpt-oss-20b`     |
| Configuration      | python-dotenv                   |
| RAG Evaluation     | RAGAS                           |


---

## Limitations

-AI chat requires a valid Groq API key. \n
-Semantic retrieval requires a configured Pinecone index. \n
-Retrieval quality depends on the quality and structure of the generated analytical documents. \n
-Edge-case facts may be missed when the relevant information is not present in the retrieved context. \n
-Uploaded CSV files must contain the required logistics columns. \n
-Chat history is stored in Streamlit session state and resets when the session is refreshed. \n
-The first run downloads the SentenceTransformer embedding model. \n
-LLM responses depend on the quality of the retrieved context. \n
-RAGAS scores depend on the quality and representativeness of the evaluation dataset.
