import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import chromadb
from chromadb.config import Settings
import torch

client = chromadb.EphemeralClient()

# タイトルのみ表示
st.set_page_config(page_title="RAG QA Demo", layout="centered")
st.title("🧠 RAG 質問応答デモ")

# モデルとDBの読み込み
@st.cache_resource
def load_resources():
    retriever = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    client = chromadb.PersistentClient(path="vectorstore/chroma_db")
    collection = client.get_or_create_collection(name="pdf_chunks")
    return retriever, tokenizer, model, collection

retriever, tokenizer, model, collection = load_resources()

# 入力UI
st.subheader("🔍 質問を入力してください")
query = st.text_input("例：What is Nomad Academy?")

if st.button("質問する") and query:
    with st.spinner("検索中..."):
        # ベクトル検索（複数チャンク取得）
        q_emb = retriever.encode([query]).tolist()[0]
        results = collection.query(query_embeddings=[q_emb], n_results=3)

        if not results["documents"][0]:
            st.error("関連する文章が見つかりませんでした。")
        else:
            context = "\n\n".join(results["documents"][0])

            # FLAN用プロンプト（日本語）
            prompt = f"""
以下の文脈を参考にして、次の質問に日本語でわかりやすく答えてください。

文脈:
{context}

質問: {query}
答え:
"""

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                )
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 結果表示
            st.markdown("### 💬 回答")
            st.success(answer.strip())
