from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Initialize with an embedding model
vector_store = InMemoryVectorStore(embedding=OpenAIEmbeddings(model="text-embedding-3-large"))

document_1 = Document(
    page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
)

documents = [document_1, document_2]

vector_store.add_documents(documents=documents, ids=["doc1", "doc2"])

top_n = 10
for index, (id, doc) in enumerate(vector_store.store.items()):
    if index < top_n:
        # docs have keys 'id', 'vector', 'text', 'metadata'
        print(f"{id}: {doc['text']}")
    else:
        break

query = 'Yesterday I had omlette for breakfast with coffee' # or use like 'coco' if you want to return doc1
doc, score = vector_store.similarity_search_with_score(query, k=1)[0]

print("Your Query: ", query)
print("Result doc: ", doc.id, " : ", doc)
print("Similarity Score: ", f"score = {score:.4f}")