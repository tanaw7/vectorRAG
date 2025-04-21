# VectorRAG Sandbox

> A quickstart guide for using an in-memory vector store with LangChain Core and OpenAI embeddings.

---

## 🚀 Features

- **InMemoryVectorStore Setup**: Initialize with `OpenAIEmbeddings` for fast local testing.  
- **Document Ingestion**: Add multiple documents with custom IDs and metadata.  
- **Document Inspection**: List and preview stored vectors and their source texts.  
- **Similarity Search**: Run single or multiple queries against the vector store and retrieve top hits with similarity scores.  
- **Automated Testing**: Validate returned document IDs against an expected-results list (`queries_list.py`).

---

## 🛠️ Prerequisites

- **Python**: Version 3.8 or higher

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/tanaw7/vectorRAG.git
   cd vectorRAG
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
---

## ⚙️ Usage

1. **Inspect the code files**
   - `main.py`: Contains the sandbox logic:
     - Initializes an `InMemoryVectorStore` with `OpenAIEmbeddings(model="text-embedding-3-large")`.

   - `queries_list.py`: Exports a `queries` list of objects with fields:
     ```python
     queries = [
         {"query": "coco", "expected_result": "doc1"},
         {"query": "rainy days", "expected_result": "doc2"},
         # ... more test cases ...
     ]
     ```

2. **Run the sandbox**
   ```bash
   python main.py
   ```
   **Sample output**:
   ```text
   doc1: I had chocolate chip pancakes and scrambled eggs for breakfast this morning.
   doc2: The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.

   Your Query 1: coco
   Result doc:  doc1  :  <Document id="doc1">
   Correct expected result?: (doc1 == doc1) True
   Similarity Score:  score = 0.5777
   -----------------------------------
   Your Query 2: rainy days
   Result doc:  doc2  :  <Document id="doc2">
   Correct expected result?: (doc2 == doc2) True
   Similarity Score:  score = 0.6565
   -----------------------------------
   # ...
   ```

---

## 🔧 Customization

- **Add documents**: Extend `main.py` with more `Document` objects and unique IDs.  
- **Extend test cases**: Edit `queries_list.py` to add new queries with expected results.  
- **Change model**: Swap the embedding model (e.g., `text-embedding-ada-002`) by updating the `OpenAIEmbeddings(model=...)` call.  
- **Adjust retrieval count**: Modify `k` in `similarity_search_with_score(query, k=...)` for top‑K results.

---

## 📁 Project Structure

```plaintext
vectorRAG/
├── main.py          # Example usage and test logic
├── queries_list.py  # List of queries + expected results
├── requirements.txt # Project dependencies
├── README.md        # This documentation
└── .env             # Environment variables (not committed)
```

---

## ⚖️ License

Distributed under the **MIT License**. Feel free to copy, modify, and distribute.

---

