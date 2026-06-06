# IR Microservices Project

This project is a Python Information Retrieval system exposed through a FastAPI backend and a Streamlit web UI. It supports text preprocessing, indexing, query representation, matching/ranking, query refinement, and evaluation.

## Main Services

- **Preprocessing**: cleans and normalizes text using tokenization, lowercasing, stop-word removal, stemming, and lemmatization.
- **Indexing**: builds an inverted index from documents and prepares TF-IDF, BM25, and embedding representations.
- **Matching and Ranking**: ranks documents using TF-IDF, BM25, embeddings, or hybrid ranking.
- **Evaluation**: computes IR metrics such as MAP, Recall, Precision@10, and nDCG.
- **Query Refinement**: applies basic query correction/refinement before search.
- **Web UI**: Streamlit interface for running search and evaluation from the browser.

## Supported Datasets

The project includes small demo datasets and two datasets from `https://ir-datasets.com/`:

- `msmarco-passage`
- `beir/quora`

Large datasets are loaded with a document limit to avoid memory overload. The default limit is `250,000`, but for local testing it is better to start with `1,000` or `5,000`.

## Requirements

- Python 3.10+
- Internet connection for first-time `ir-datasets` downloads

Install dependencies:

```powershell
pip install -r requirements.txt
```

If you use the existing virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run The Project

Run backend and UI together:

```powershell
python main.py --mode all
```

Backend URL:

```text
http://127.0.0.1:8000
```

Streamlit UI:

```text
http://127.0.0.1:8501
```

Run backend only:

```powershell
python main.py --mode api
```

Run UI only:

```powershell
python main.py --mode web
```

You can also run them manually:

```powershell
uvicorn main:app --reload
```

```powershell
streamlit run web_ui/app.py
```

## API Docs

After starting the backend, open:

```text
http://127.0.0.1:8000/docs
```

## Example Dataset Indexing Request

Index a small subset of MSMARCO:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/index?dataset_name=msmarco-passage&max_documents=1000" `
  -ContentType "application/json" `
  -Body '{"Documents":[],"Representation_Type":"tfidf","K1":1.5,"B":0.75,"Vector_Size":64}'
```

Index a small subset of BEIR Quora:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/index?dataset_name=beir/quora&max_documents=1000" `
  -ContentType "application/json" `
  -Body '{"Documents":[],"Representation_Type":"bm25","K1":1.5,"B":0.75,"Vector_Size":64}'
```

## Project Structure

```text
api/        FastAPI route handlers
core/       IR logic, indexing, ranking, preprocessing, dataset loading
schemas/    Pydantic request and response schemas
web_ui/     Streamlit frontend
main.py     Application entrypoint
```

