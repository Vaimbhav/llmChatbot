import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing")

genai.configure(api_key=GEMINI_API_KEY)

# Define a LangChain-compatible embedding wrapper for Gemini
class GeminiEmbedding(Embeddings):
    def embed_documents(self, texts):
        results = []
        for text in texts:
            if not text.strip():
                continue  # skip empty strings
            try:
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="RETRIEVAL_DOCUMENT",
                    title="Gemini Doc"
                )["embedding"]
                results.append(result)
            except Exception as e:
                print(f"[Embed Error] Skipping chunk due to: {e}")
        return results

    def embed_query(self, text):
        return self.embed_documents([text])[0] if text.strip() else []

class IndexerAgent:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.db = None

    def create_index(self):
        docs = []
        for file in os.listdir(self.data_dir):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(self.data_dir, file))
                docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        splits = [doc for doc in splits if doc.page_content.strip()]  # skip empty docs

        if not splits:
            print("No valid content found in PDFs. Skipping index creation.")
            return

        embeddings = GeminiEmbedding()
        self.db = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="chroma"
        )

    def search(self, query: str):
        if not self.db:
            embeddings = GeminiEmbedding()
            self.db = Chroma(
                persist_directory="chroma",
                embedding_function=embeddings
            )

        results = self.db.similarity_search(query, k=4)
        return [doc.page_content for doc in results]