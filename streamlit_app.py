import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import chromadb
import torch

# モデル読み込み
retriever = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")

client = chromadb.PersistentClient(path="vectorstore/chroma_db")
collection = client.get_or_create_collection(name="pdf_chunks")

# UI設定
st.set_page_config(page_title="PDF質問AI", page_icon="📄")
st.title("📄 PDFから答えるAI")
st.markdown("PDFに基づいて、質問に答えます。")

# 入力フォーム
query = st.text_input("質問を入力してください")

if query:
    with st.spinner("検索と生成中..."):
        q_emb = retriever.encode([query]).tolist()[0]
        results = collection.query(query_embeddings=[q_emb], n_results=3)
        context = "\n\n".join(results["documents"][0])

        prompt = f"""
        以下の文章を参考にして、次の質問に対する日本語の答えを簡潔に1つ書いてください。
        
        文章:
        {context}
        
        質問: {query}
        """


        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.5,
                top_k=30,
                top_p=0.85,
                repetition_penalty=1.2
            )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.markdown("### 🧠 回答")
        st.text_area("🧠 回答", value=answer.replace(prompt, "").strip(), height=200)

