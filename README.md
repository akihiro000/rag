# 🧠 RAG × Streamlit PDF QA アプリ

このプロジェクトは、PDFの内容に基づいてユーザーの質問に答える**ローカル実行型のRAGアプリ**です。  
ノマドアカデミーに関するPDFをベースに、Streamlit UIから質問を入力することで、該当箇所を探し出し、LLMが日本語で回答します。

---

## 🔍 できること

- PDFを自動で分割・ベクトル化し、ChromaDBに保存
- 類似文検索に基づいたプロンプトを作成
- ローカルLLM（rinna / GPT2ベース）で日本語回答を生成
- Streamlit UIから誰でも簡単に質問・確認が可能

---

## 🖥 画面イメージ

![screenshot](./your_screenshot_here.png)  
※実行画面のスクリーンショットをここに貼ってください

---

## ⚙️ 使用技術

- Python
- [PyMuPDF](https://pymupdf.readthedocs.io/)：PDFからテキスト抽出
- [sentence-transformers](https://www.sbert.net/)：テキスト埋め込みベクトル化
- [ChromaDB](https://docs.trychroma.com/)：ベクトルデータベース
- [rinna/japanese-gpt2-medium](https://huggingface.co/rinna/japanese-gpt2-medium)：日本語LLM
- [Streamlit](https://streamlit.io/)：WebベースUI

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

## 💡 補足
- 本リポジトリは非公開で、限定的にアクセスを許可しています
- モデルは小型の日本語LLMを使用しており、精度や回答の自然さに限界はあるものの、動作確認済みのRAGプロトタイプとして提出可能です