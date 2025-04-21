from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from queries_list import queries

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

for index, item in enumerate(queries):
    q = item["query"]
    res = item["expected_result"]
    doc, score = vector_store.similarity_search_with_score(q, k=1)[0]
    print(f"Your Query {index + 1}: {q}")
    print("Result doc: ", doc.id, " : ", doc)
    print(f"Correct expected result?: ({doc.id} == {res}) {res==doc.id}")
    print("Similarity Score: ", f"score = {score:.4f}")

    docScores = vector_store.similarity_search_with_score(q, k=2)
    doc1, score1 = docScores[0]
    doc2, score2 = docScores[1]
    print(f"simi: {doc1.id, score1} and simi: {doc2.id, score2}")
    print("-----------------------------------")