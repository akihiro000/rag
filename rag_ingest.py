import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os

# PDFを読み込む
pdf_path = "data/sample.pdf"
doc = fitz.open(pdf_path)
text = "\n".join([page.get_text() for page in doc])

# チャンクに分ける（500文字ごと）
chunks = [text[i:i+500] for i in range(0, len(text), 500)]

# ベクトル化
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks).tolist()

# Chromaに保存（新しい初期化方法 + 削除処理あり）
client = chromadb.PersistentClient(path="vectorstore/chroma_db")

# 初回実行で delete_collection が失敗するので try-except で対応
try:
    client.delete_collection("pdf_chunks")
except:
    pass  # 存在しない場合は無視

collection = client.get_or_create_collection(name="pdf_chunks")

# データを追加
for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
    collection.add(
        documents=[chunk],
        embeddings=[emb],
        ids=[str(i)]
    )

print("✅ Chromaに保存完了")
