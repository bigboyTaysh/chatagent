# ingest_docs.py

import os
import glob
import faiss
import pickle

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from pathlib import Path

# === CONFIG ===
MD_DIR = "mock_docs"
INDEX_FILE = "vector_index.faiss"
CHUNK_SIZE = 500  # characters

# === MOCK SETUP ===
MOCK_DOCS = {
    "intro.md": "# Azure Wiki\nThis is a sample introduction.",
    "networking.md": "## Networking\nAzure VNET allows private network configurations.",
    "identity.md": "## Identity\nAzure Active Directory manages users and roles."
}

os.makedirs(MD_DIR, exist_ok=True)
for fname, content in MOCK_DOCS.items():
    with open(os.path.join(MD_DIR, fname), "w") as f:
        f.write(content)

# === EMBEDDING SETUP ===
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

docs = []
metadata = []

# === CHUNK AND EMBED DOCS ===
for file_path in glob.glob(f"{MD_DIR}/*.md"):
    with open(file_path, "r") as f:
        text = f.read()
        for i in range(0, len(text), CHUNK_SIZE):
            chunk = text[i:i+CHUNK_SIZE]
            docs.append(chunk)
            metadata.append({"source": file_path, "start": i})

print(f"Chunks: {len(docs)}")

embeddings = model.encode(docs, convert_to_tensor=False)
embeddings = normalize(embeddings, axis=1)

dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and metadata
faiss.write_index(index, INDEX_FILE)
with open("metadata.pkl", "wb") as f:
    pickle.dump((docs, metadata), f)

print("✅ FAISS index created and saved.")
