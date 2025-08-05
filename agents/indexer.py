# import os
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_core.embeddings import Embeddings
# from langchain_chroma import Chroma
#
# import google.generativeai as genai
# from dotenv import load_dotenv
#
# # Load environment variables
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     raise ValueError("GEMINI_API_KEY is missing")
#
# genai.configure(api_key=GEMINI_API_KEY)
#
# # Define a LangChain-compatible embedding wrapper for Gemini
# class GeminiEmbedding(Embeddings):
#     def embed_documents(self, texts):
#         results = []
#         for text in texts:
#             if not text.strip():
#                 continue  # skip empty strings
#             try:
#                 result = genai.embed_content(
#                     model="models/embedding-001",
#                     content=text,
#                     task_type="RETRIEVAL_DOCUMENT",
#                     title="Gemini Doc"
#                 )["embedding"]
#                 results.append(result)
#             except Exception as e:
#                 print(f"[Embed Error] Skipping chunk due to: {e}")
#         return results
#
#     def embed_query(self, text):
#         return self.embed_documents([text])[0] if text.strip() else []
#
# class IndexerAgent:
#     def __init__(self, data_dir: str):
#         self.data_dir = data_dir
#         self.db = None
#
#     def create_index(self):
#         docs = []
#         for file in os.listdir(self.data_dir):
#             if file.endswith(".pdf"):
#                 loader = PyPDFLoader(os.path.join(self.data_dir, file))
#                 docs.extend(loader.load())
#
#         splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         splits = splitter.split_documents(docs)
#         splits = [doc for doc in splits if doc.page_content.strip()]  # skip empty docs
#
#         if not splits:
#             print("No valid content found in PDFs. Skipping index creation.")
#             return
#
#         embeddings = GeminiEmbedding()
#         self.db = Chroma.from_documents(
#             documents=splits,
#             embedding=embeddings,
#             persist_directory="chroma"
#         )
#
#     def search(self, query: str):
#         if not self.db:
#             embeddings = GeminiEmbedding()
#             self.db = Chroma(
#                 persist_directory="chroma",
#                 embedding_function=embeddings
#             )
#
#         results = self.db.similarity_search(query, k=4)
#         return [doc.page_content for doc in results]










import os
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing")
genai.configure(api_key=GEMINI_API_KEY)

class GeminiEmbedding(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embs = []
        for text in texts:
            if not text.strip():
                continue
            try:
                e = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="RETRIEVAL_DOCUMENT",
                    title="Gemini Doc"
                )["embedding"]
                embs.append(e)
            except Exception as e:
                print(f"[Embed Error] {e}")
        return embs

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0] if text.strip() else []

class IndexerAgent:
    def __init__(self, data_dir: str, persist_dir: str = "chroma"):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.db: Optional[Chroma] = None
        self.embedder = GeminiEmbedding()

    def create_index(self):
        # load all PDFs
        docs = []
        for fn in os.listdir(self.data_dir):
            if fn.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(self.data_dir, fn))
                docs.extend(loader.load())

        # split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        splits = [d for d in splits if d.page_content.strip()]
        if not splits:
            print("No content to index.")
            return

        # build Chroma index
        self.db = Chroma.from_documents(
            documents=splits,
            embedding=self.embedder,
            persist_directory=self.persist_dir
        )

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        # lazy-load index if needed
        if not self.db:
            self.db = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedder
            )

        # compute query embedding
        q_emb = self.embedder.embed_query(query)
        # low-level query to get both documents and distances
        resp = self.db._collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "distances"]
        )
        texts: List[str]     = resp["documents"][0]
        distances: List[float] = resp["distances"][0]

        # convert distances → similarity score in (0,1]
        results = []
        for doc, dist in zip(texts, distances):
            sim = 1.0 / (1.0 + dist)
            results.append({"text": doc, "score": sim})
        return results
