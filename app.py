# app.py

from src.loader import load_pdf
from src.splitter import split_pages
from src.embedder import Embedder
from src.vector_store import FaissVectorStore
from src.retriever import Retriever
from src.llm import LLMClient


pdf_path = "data/papers/paper01.pdf"

# 1. 读取PDF
pages = load_pdf(pdf_path)

# 2. 切块
chunks = split_pages(pages)

print("Chunks:", len(chunks))

# 3. Embedding
embedder = Embedder()

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = embedder.encode_documents(
    texts
)

# 4. 建立向量库
vector_store = FaissVectorStore()

vector_store.build(
    embeddings
)

# 5. Retriever
retriever = Retriever(
    embedder,
    vector_store,
    chunks
)

query = (
    "How does gasification temperature "
    "affect hydrogen production?"
)

results = retriever.retrieve(
    query,
    top_k=5
)

# 6. 打印检索结果
for i, r in enumerate(results, 1):

    print("\n====================")
    print(f"Top {i}")
    print("Score:", r["score"])
    print("Source:", r["source"])
    print("Page:", r["page"])
    print(r["text"][:300])

# 7. LLM
llm = LLMClient()

answer = llm.answer(
    query,
    results
)

print("\nAnswer:\n")
print(answer)