import os
import zipfile
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
import uvicorn
import re

# === CONFIG ===
ZIP_PATH = "your_docs.zip"            # Your zip file path
EXTRACT_DIR = "extracted_docs"        # Where to extract
ATTACHMENTS_PATH = os.path.join(EXTRACT_DIR, ".attachments")

CHUNK_SIZE = 500                      # tokens/chars approx chunk size for splitting
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# === FASTAPI INIT ===
app = FastAPI()

# Serve attachments folder statically
if os.path.exists(ATTACHMENTS_PATH):
    app.mount("/attachments", StaticFiles(directory=ATTACHMENTS_PATH), name="attachments")

app.mount("/static", StaticFiles(directory="src/static"), name="static")

# === Pydantic Schemas ===
class Message(BaseModel):
    text: str

class ChatRequest(BaseModel):
    message: Message

# === Globals ===
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
index = None
documents = []  # to store (text_chunk, metadata)


# === UTILS ===

def extract_zip(zip_path=ZIP_PATH, extract_to=EXTRACT_DIR):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def load_md_files(root_dir=EXTRACT_DIR):
    md_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.md'):
                full_path = os.path.join(root, file)
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                md_files.append((full_path, content))
    return md_files

def chunk_text(text, size=CHUNK_SIZE):
    # Simple chunk by sentences or paragraphs roughly, fallback to slicing
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    chunk = ""
    for s in sentences:
        if len(chunk) + len(s) < size:
            chunk += s + " "
        else:
            chunks.append(chunk.strip())
            chunk = s + " "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def build_faiss_index(docs: List[str]):
    embeddings = model.encode(docs, convert_to_numpy=True)
    dim = embeddings.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(embeddings)
    return idx

# === LOAD & PREPARE ===
@app.on_event("startup")
def startup_event():
    global index, documents
    print("Extracting ZIP and loading documents...")
    extract_zip()
    md_files = load_md_files()
    print(f"Found {len(md_files)} markdown files.")

    # Chunk all docs & prepare flat list for embedding
    all_chunks = []
    documents = []
    for filepath, content in md_files:
        chunks = chunk_text(content)
        for c in chunks:
            all_chunks.append(c)
            documents.append({"text": c, "source": filepath})

    print(f"Created {len(all_chunks)} chunks, embedding now...")
    index = build_faiss_index(all_chunks)
    print("FAISS index built!")

# === CHAT ROUTE ===

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    query = request.message.text
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")

    # Embed query
    q_emb = model.encode([query], convert_to_numpy=True)

    # Search in FAISS index
    D, I = index.search(q_emb, k=3)  # top 3 matches
    hits = [documents[i] for i in I[0]]

    # Combine hits as context for LLaMA prompt (mock or call your model here)
    context_text = "\n---\n".join(hit["text"] for hit in hits)

    # For demonstration: simple answer with context
    answer = f"Based on the documents, here are some relevant excerpts:\n\n{context_text[:1000]}..."

    # TODO: replace above with your LLaMA call using `context_text` + `query`

    return {"text": answer}

# === SERVE FRONTEND ===
from fastapi.responses import FileResponse

@app.get("/")
async def serve_index():
    return FileResponse("src/static/index.html")


# === RUN ===
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
