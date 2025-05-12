from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import chromadb
from chromadb.config import Settings
import torch

# Chromaからコレクションを読み込む
client = chromadb.PersistentClient(path="vectorstore/chroma_db")
collection = client.get_or_create_collection(name="pdf_chunks")

# モデル読み込み（ベクトル・LLM）
retriever = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

print("🔄 モデル読み込み完了")

# 対話ループ
while True:
    query = input("🔍 質問を入力してください: ")
    if query.lower() in ["exit", "quit"]:
        break

    # 質問をベクトル化して類似文検索
    q_emb = retriever.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[q_emb], n_results=3)

    # 検索結果を結合してプロンプト化
    context = "\n\n".join(results["documents"][0])

    prompt = f"""
以下の文章に基づいて、質問に日本語で丁寧に答えてください。

文章:
{context}

質問:
{query}

答え:
"""

    # LLMで回答生成
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n🧠 回答:\n", answer)
