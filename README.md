# 🧠 RAG × Streamlit PDF QA アプリ

このプロジェクトは、PDFの内容に基づいてユーザーの質問に答えるローカル実行型のRAGアプリです。
PDFをベースに、Streamlit UIから質問を入力することで、該当箇所を探し出し、LLMが日本語で回答します。
商用API（OpenAIなど）を一切使わず、完全ローカル環境で動作することを重視して構築しました。

---

## 🔍 できること

- PDFを自動で分割・ベクトル化し、ChromaDBに保存
- 類似文検索に基づいたプロンプトを作成
- ローカルLLM（rinna / GPT2ベース）で日本語回答を生成
- Streamlit UIから誰でも簡単に質問・確認が可能

---

## 🖥 イメージ

https://github.com/user-attachments/assets/3376d636-4ec0-456e-b0c7-2e058c69a7ce

---

## ⚙️ 使用技術

- ## ⚙️ 使用技術
- Python
- PyMuPDF：PDFからテキストを抽出
- sentence-transformers：テキストをベクトルに変換（検索用）
- ChromaDB：ベクトルデータベースとして類似文検索を実現
- google/flan-t5-base：質問応答用の自然言語生成モデル（英語対応LLM）
- Streamlit：Webベースのユーザーインターフェース

---

## 📝 使用方法

1. 仮想環境を作成・起動  
2. 必要なパッケージをインストール

```bash
pip install -r requirements.txt
```

3. PDFをベクトル化して保存
```bash
python rag_ingest.py
```
4. アプリを起動
```bash
streamlit run streamlit_app.py
```

## 📁 ディレクトリ構成例
```bash
rag-portfolio/
├── streamlit_app.py       # Streamlit UIアプリ
├── rag_ingest.py          # PDFからベクトル登録
├── rag_answer.py          # ターミナルベースのQA実行
├── data/
│   └── sample.pdf         # テストPDF（精密栄養学）
├── vectorstore/           # ChromaDB（git除外推奨）
├── requirements.txt       # ライブラリ一覧
├── .gitignore             # キャッシュ除外
└── README.md              # ← このファイル

```
