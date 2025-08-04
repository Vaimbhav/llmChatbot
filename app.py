from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import shutil
import uuid
from pathlib import Path
import asyncio  # NEW

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from PyPDF2 import PdfReader

from agents.indexer import IndexerAgent
from agents.researcher import ResearchAgent
from agents.indexer import GeminiEmbedding

# NEW: load .env so AUTO_SCAN_FOLDER is available when using uvicorn app:app
from dotenv import load_dotenv  # NEW
load_dotenv()  # NEW

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

indexer = IndexerAgent(data_dir=DATA_DIR)
indexer.create_index()
researcher = ResearchAgent(indexer)

# ===== Local-folder scan settings (unchanged) =====
MAX_LOCAL_FILE_SIZE_MB = 10
MAX_LOCAL_FILE_SIZE_BYTES = MAX_LOCAL_FILE_SIZE_MB * 1024 * 1024
SCAN_EXTS = {".pdf", ".txt", ".md", ".json", ".csv", ".log", ".yaml", ".yml"}
# ==================================================

def extract_text(file_path: str | Path) -> str:
    file_path = str(file_path)
    if file_path.lower().endswith(".pdf"):
        text = ""
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text.strip()
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()

def _is_hidden(base: Path, p: Path) -> bool:
    try:
        rel_parts = p.relative_to(base).parts
    except Exception:
        return False
    return any(part.startswith(".") for part in rel_parts)

def scan_folder_and_index(
    folder_path: str | Path,
    recursive: bool = True,
    include_exts: Optional[set[str]] = None
) -> dict:
    base = Path(folder_path).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Folder not found or not a directory: {base}")

    include_exts = {e.lower() for e in (include_exts or SCAN_EXTS)}

    files_seen, files_indexed, files_skipped = 0, 0, []
    it = base.rglob("*") if recursive else base.glob("*")

    embeddings = GeminiEmbedding()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for p in it:
        if p.is_dir():
            continue
        files_seen += 1

        if _is_hidden(base, p):
            files_skipped.append({"path": str(p), "reason": "hidden"})
            continue

        ext = p.suffix.lower()
        if ext not in include_exts:
            files_skipped.append({"path": str(p), "reason": f"extension_not_allowed({ext})"})
            continue

        try:
            size = p.stat().st_size
        except Exception as e:
            files_skipped.append({"path": str(p), "reason": f"stat_error({e})"})
            continue

        if size > MAX_LOCAL_FILE_SIZE_BYTES:
            files_skipped.append({"path": str(p), "reason": f"too_large(>{MAX_LOCAL_FILE_SIZE_MB}MB)"})
            continue

        try:
            content = extract_text(p)
        except UnicodeDecodeError:
            files_skipped.append({"path": str(p), "reason": "decode_error(non_utf8)"})
            continue
        except Exception as e:
            files_skipped.append({"path": str(p), "reason": f"read_error({e})"})
            continue

        if not content.strip():
            files_skipped.append({"path": str(p), "reason": "no_text_content"})
            continue

        try:
            chunks = splitter.create_documents([content], metadatas=[{"source": str(p)}])
            if indexer.db:
                indexer.db.add_documents(chunks, embeddings=embeddings)
            else:
                indexer.db = Chroma.from_documents(
                    chunks,
                    embedding=embeddings,
                    persist_directory="chroma"
                )
            if hasattr(indexer.db, "persist") and getattr(indexer.db, "_persist_directory", None):
                indexer.db.persist()
            files_indexed += 1
        except Exception as e:
            files_skipped.append({"path": str(p), "reason": f"index_error({e})"})

    summary = {
        "folder": str(base),
        "recursive": recursive,
        "seen": files_seen,
        "indexed": files_indexed,
        "skipped": files_skipped,
        "max_local_file_size_mb": MAX_LOCAL_FILE_SIZE_MB,
        "allowed_exts": sorted(include_exts),
    }
    print("Folder scan summary:", summary)
    return summary

# NEW: Run autoscan at FastAPI startup (works with `uvicorn app:app`)
@app.on_event("startup")
async def autoscan_on_startup():
    autoscan_dir = os.getenv("AUTO_SCAN_FOLDER")
    print("AUTO_SCAN_FOLDER =", autoscan_dir)
    if autoscan_dir:
        try:
            print(f"Starting auto-scan of: {autoscan_dir}")
            # Offload sync scanning to a thread to avoid blocking the event loop
            await asyncio.to_thread(scan_folder_and_index, autoscan_dir, True, None)
            print("Auto-scan complete.")
        except Exception as e:
            print("Auto-scan failed:", e)

# ---- existing endpoints below (unchanged) ----
@app.post("/api/report")
async def generate_report(req: Request):
    # NOTE: If you still have the previous stream-consumed error,
    # apply the one-parser fix from my last message.
    content_type = (req.headers.get("content-type") or "").lower()
    form = None
    body = None
    files: List[UploadFile] = []
    query: Optional[str] = None
    mode: str = "normal"

    try:
        if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
            form = await req.form()
            query = form.get("query")
            mode = form.get("mode", "normal")
            if hasattr(form, "getlist"):
                files = form.getlist("files")
        else:
            body = await req.json()
            query = (body or {}).get("query")
            mode = (body or {}).get("mode", "normal")
            files = []
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid request payload: {e}"})

    try:
        if files:
            for uploaded_file in files:
                for existing_file in os.listdir(DATA_DIR):
                    if existing_file.endswith(".meta"):
                        continue
                    if uploaded_file.filename in existing_file:
                        return JSONResponse(
                            status_code=400,
                            content={"error": f"A file named '{uploaded_file.filename}' already exists."}
                        )
                file_id = str(uuid.uuid4())
                file_ext = os.path.splitext(uploaded_file.filename)[1]
                save_path = os.path.join(DATA_DIR, f"{file_id}{file_ext}")
                with open(save_path, "wb") as f:
                    shutil.copyfileobj(uploaded_file.file, f)
                uploaded_file.file.close()

                content = extract_text(save_path)
                if content.strip():
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = splitter.create_documents([content], metadatas=[{"source": uploaded_file.filename}])
                    embeddings = GeminiEmbedding()
                    if indexer.db:
                        indexer.db.add_documents(chunks, embeddings=embeddings)
                    else:
                        indexer.db = Chroma.from_documents(
                            chunks,
                            embedding=embeddings,
                            persist_directory="chroma"
                        )
                    if hasattr(indexer.db, "persist") and getattr(indexer.db, "_persist_directory", None):
                        indexer.db.persist()
                else:
                    print(f"No valid content found in {uploaded_file.filename}. Skipping index.")
        report = researcher.generate_report(query, mode=mode)
        return report
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Something went wrong: {str(e)}"})

@app.get("/greet")
async def greet():
    return {"message": "Welcome to the Research API!"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    # Keeping your original __main__ autoscan for CLI runs
    autoscan_dir = os.getenv("AUTO_SCAN_FOLDER")
    print('Folder name is -> ', autoscan_dir)
    if autoscan_dir:
        try:
            print(f"AUTO_SCAN_FOLDER set to: {autoscan_dir}. Starting scan...")
            scan_folder_and_index(autoscan_dir, recursive=True)
            print("Auto-scan complete.")
        except Exception as e:
            print("Auto-scan failed:", e)

    indexer.create_index()
    query = input("Enter a research query: ")
    report = researcher.generate_report(query)
    print("\n===== Research Report =====\n")
    print(report)




# @app.post("/api/index-folder")
# async def index_folder(req: Request):
#     try:
#         payload = await req.json()
#         folder_path = payload.get("folder_path")
#         if not folder_path:
#             return JSONResponse(status_code=400, content={"error": "folder_path is required"})
#         recursive = bool(payload.get("recursive", True))
#         exts = payload.get("exts")
#         include_exts = {e.lower() for e in exts} if exts else None
#         summary = scan_folder_and_index(folder_path, recursive=recursive, include_exts=include_exts)
#         return summary
#     except FileNotFoundError as e:
#         return JSONResponse(status_code=400, content={"error": str(e)})
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": f"Something went wrong: {str(e)}"})





# import warnings
# warnings.filterwarnings("ignore")
#
# from fastapi import FastAPI, Request, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from typing import List
# import os
# import shutil
# import uuid
#
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from PyPDF2 import PdfReader
#
# from agents.indexer import IndexerAgent
# from agents.researcher import ResearchAgent
# from agents.indexer import GeminiEmbedding  # ✅ Fixed import
#
# app = FastAPI()
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# # ✅ Initialize agents
# DATA_DIR = "data"
# os.makedirs(DATA_DIR, exist_ok=True)
#
# indexer = IndexerAgent(data_dir=DATA_DIR)
# indexer.create_index()
# researcher = ResearchAgent(indexer)
#
# # ✅ Helper to extract text
# def extract_text(file_path):
#     if file_path.endswith(".pdf"):
#         text = ""
#         with open(file_path, "rb") as f:
#             reader = PdfReader(f)
#             for page in reader.pages:
#                 text += page.extract_text() or ""
#         return text.strip()
#     else:
#         with open(file_path, "r", encoding="utf-8") as f:
#             return f.read().strip()
#
#
# @app.post("/api/report")
# async def generate_report(req: Request):
#     print("Request received:", req)
#
#     form = await req.form()
#     body  = await req.json()
#     print ("Form data:", form)
#     query = form.get("query") or body.get("query")
#     mode = form.get("mode", "normal") or body.get("mode", "normal")
#     print("Query: ", query)
#     print("Mode: ", mode)
#     files: List[UploadFile] = form.getlist("files")  # ✅ Multiple file support
#
#     try:
#         # ✅ Save and index any uploaded files
#         if files:
#             print('Files received ->', [f.filename for f in files])
#
#             for uploaded_file in files:
#                 # ✅ Check if file with same original name already exists
#                 for existing_file in os.listdir(DATA_DIR):
#                     if existing_file.endswith(".meta"):
#                         continue
#                     if uploaded_file.filename in existing_file:
#                         return JSONResponse(
#                             status_code=400,
#                             content={"error": f"A file named '{uploaded_file.filename}' already exists."}
#                         )
#
#                 # ✅ Save file with UUID but keep original name in metadata
#                 file_id = str(uuid.uuid4())
#                 file_ext = os.path.splitext(uploaded_file.filename)[1]
#                 save_path = os.path.join(DATA_DIR, f"{file_id}{file_ext}")
#
#                 with open(save_path, "wb") as f:
#                     shutil.copyfileobj(uploaded_file.file, f)
#                 uploaded_file.file.close()
#
#                 # ✅ Extract text and index
#                 content = extract_text(save_path)
#                 if content.strip():
#                     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#                     chunks = splitter.create_documents([content], metadatas=[{"source": uploaded_file.filename}])
#                     embeddings = GeminiEmbedding()
#
#                     if indexer.db:
#                         indexer.db.add_documents(chunks, embeddings=embeddings)
#                     else:
#                         indexer.db = Chroma.from_documents(
#                             chunks,
#                             embedding=embeddings,
#                             persist_directory="chroma"
#                         )
#
#                     # ✅ Only persist if DB supports it
#                     if hasattr(indexer.db, "persist") and getattr(indexer.db, "_persist_directory", None):
#                         indexer.db.persist()
#
#                 else:
#                     print(f"No valid content found in {uploaded_file.filename}. Skipping index.")
#
#         # ✅ Generate the report from updated local + web sources
#         report = researcher.generate_report(query, mode=mode)
#         return report
#
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": f"Something went wrong: {str(e)}"})
#
#
# @app.get("/greet")
# async def greet():
#     return {"message": "Welcome to the Research API!"}
#
#
# @app.get("/health")
# async def health_check():
#     return {"status": "ok"}
#
#
# if __name__ == "__main__":
#     indexer = IndexerAgent(data_dir=DATA_DIR)
#     researcher = ResearchAgent(indexer)
#
#     indexer.create_index()
#     query = input("Enter a research query: ")
#     report = researcher.generate_report(query)
#     print("\n===== Research Report =====\n")
#     print(report)
#
#
#
#
