from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Initialize with an embedding model
vector_store = InMemoryVectorStore(embedding=OpenAIEmbeddings())

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

query = 'sun and rain' # or use like 'coco' if you want to return doc1
docs = vector_store.similarity_search(query, k=1)

print(type(docs))
print("Similar doc: ", docs[0].id, " : ", docs[0])