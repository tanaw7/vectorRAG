# VectorRAG Sandbox

> A quickstart guide for using an in-memory vector store with LangChain Core and OpenAI embeddings.

---

## 🚀 Features

- **InMemoryVectorStore Setup**: Initialize with `OpenAIEmbeddings` for fast local testing.
- **Document Ingestion**: Add multiple documents with custom IDs and metadata.
- **Document Inspection**: List and preview stored vectors and their source texts.
- **Similarity Search**: Query against the vector store and retrieve the most relevant documents.

---

## 🛠️ Prerequisites

- **Python**: Version 3.8 or higher
- **OpenAI API Key**: Exported to `OPENAI_API_KEY` or stored in a `.env` file
- **Internet Connection**: Required to call the OpenAI Embeddings API

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

4. **Configure your API key**

   - Create a `.env` file at the project root:
     ```env
     OPENAI_API_KEY=your_api_key_here
     ```
   - Or export directly:
     ```bash
     export OPENAI_API_KEY="your_api_key_here"
     ```

---

## ⚙️ Usage

1. **Open and review **``

   - Two sample `Document` instances are created (metadata included).
   - An `InMemoryVectorStore` is initialized with `OpenAIEmbeddings`.
   - Documents are added via `vector_store.add_documents(...)`.
   - The script lists up to 10 stored entries and performs a similarity search.

2. **Run the example**

   ```bash
   python sandbox.py
   ```

   **Expected output**:

   ```text
   doc1: I had chocolate chip pancakes and scrambled eggs for breakfast this morning.
   doc2: The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.
   Similar doc: doc2 : <Document id="doc2">
   ```

---

## 🔧 Customization

- **Add documents**: Extend `sandbox.py` with more `Document` objects and unique IDs.
- **Change queries**: Modify the `query` variable to test different search scenarios.
- **Adjust retrieval count**: Tweak `k` in `vector_store.similarity_search(query, k=...)`.

---

## 📁 Project Structure

```plaintext
vectorRAG/
├── sandbox.py       # Example usage and test script
├── requirements.txt # Project dependencies
├── README.md        # This documentation
└── .env             # Environment variables (not committed)
```

---

## ⚖️ License

Distributed under the **MIT License**. Feel free to copy, modify, and distribute.

---

*Polished for clarity, consistency, and ease of use.*

