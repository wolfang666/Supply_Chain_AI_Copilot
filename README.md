# ⛓ Supply Chain AI Copilot

> Intelligent logistics analytics using RAG · LLaMA 3 70B (Groq) · FAISS · Streamlit

---

## Project Overview

Supply Chain AI Copilot transforms raw logistics CSV data into actionable intelligence.
Users ask natural-language questions; the system retrieves relevant analytics from a FAISS
vector index and passes them as context to a Groq-hosted LLaMA 3 70B model for accurate,
data-grounded answers.

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
CSV  →  data_processing.py  →  clean DataFrame + delay_days
                                     │
                    ┌────────────────┴──────────────────┐
                    ▼                                   ▼
              analytics.py                       rag_engine.py
           (KPI DataFrames)                  build_documents()
                    │                               │
              ui/charts.py               SentenceTransformer embed
           (Plotly Figures)              FAISS IndexFlatIP store
                    │                               │
                    └──────── app.py / pages ───────┘
                                     │
                              User question
                                     │
                          rag_engine.retrieve()  ← top-k chunks
                                     │
                            Groq API (LLaMA 3)
                                     │
                             Answer rendered in chat
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
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=all-MiniLM-L6-v2
RAG_TOP_K=6
DELAY_THRESHOLD_DAYS=3
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

| Component | Technology |
|---|---|
| UI | Streamlit |
| Data | Pandas |
| Charts | Plotly |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector store | FAISS (CPU) |
| LLM | Groq · `llama-3.3-70b-versatile` |
| Config | python-dotenv |

---

## Limitations

- AI chat requires a valid `GROQ_API_KEY` in `.env`.
- The RAG pipeline retrieves top-6 chunks; edge-case facts may be missed on very large datasets.
- Uploaded CSVs must include: `Order_ID`, `Warehouse`, `Product`, `Order_Date`, `Ship_Date`, `Destination`.
- Chat history resets on page refresh (Streamlit session state).
- First run downloads the `all-MiniLM-L6-v2` model (~80 MB).
