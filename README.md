# Rehab\_RAG: 理学療法ガイドラインのためのRAG手法評価・検証リポジトリ

[![Project Status: Experimental](https://img.shields.io/badge/status-experimental-orange)](https://github.com/YouSayH/Rehab_RAG)

## 概要 (Overview)

このリポジトリは、理学療法（リハビリテーション）領域の臨床ガイドラインなどの専門文書に対して、最適な**RAG (Retrieval-Augmented Generation)** パイプラインを模索・評価するために作成されました。

最終的な目標は、臨床現場の専門家が持つ疑問に対し、膨大なガイドラインの中から、迅速かつ正確に、そして根拠を持って回答を提示できるAIアシスタントを構築することです。

そのために、様々なRAGの構成要素（チャンキング、クエリ拡張、フィルタリング等）を組み合わせた手法を個別のディレクトリで実装し、それぞれの性能（精度、速度、コスト）を比較・検証します。

## RAG手法の構成要素 (Components of RAG Techniques)

このリポジトリでは、以下の構成要素を様々に組み合わせてRAGパイプラインを構築・評価します。

1.  **チャンキング戦略 (Chunking Strategies)**
    *   **概要**: 知識源となる長文のドキュメントを、LLMが扱いやすいサイズのかたまり（チャンク）に分割する手法。
    *   **検証中の手法**:
        *   `structured_semantic_chunk`: Markdownの構造（見出し）を維持しつつ、意味のある段落単位で分割するハイブリッド戦略。

2.  **ベクトルDBとEmbeddingモデル (Vector DB & Embeddings)**
    *   **概要**: テキストチャンクを意味を捉えたベクトル（Embedding）に変換し、高速な類似度検索を可能にするデータベース。
    *   **検証中の手法**:
        *   `chromadb`: ローカル環境で手軽に構築できるベクトルデータベース。
        *   `sentence-transformers`: 多言語対応など、様々な特性を持つEmbeddingモデルライブラリ。(`intfloat/multilingual-e5-large`などを使用)

3.  **クエリ拡張 (Query Expansion)**
    *   **概要**: ユーザーの短い質問文だけでは検索精度が上がらない場合に、質問文をより検索に適した形に変換・拡張する手法。
    *   **検証中の手法**:
        *   `hyde_prf` (Hypothetical Document Embeddings / Pseudo Relevance Feedback): ユーザーの質問からLLMが「架空の理想的な回答」を生成し、その回答文でベクトル検索を行うことで、検索精度を向上させる。

4.  **プロンプト構築 (Prompt Construction)**
    *   **概要**: 検索で得られた文書チャンクを、LLMへの指示（プロンプト）にどのように組み込むかという戦略。
    *   **検証中の手法**:
        *   `retrieval-injected`: 検索結果を「参考情報」としてプロンプトに明記し、LLMにその情報のみに基づいて回答するよう厳密に指示する。

5.  **後処理とフィルタリング (Post-processing & Filtering)**
    *   **概要**: 検索で得られた文書チャンクの中には、ノイズや質問と矛盾する内容が含まれる場合があります。これらを最終的な回答生成前に除去する手法。
    *   **検証中の手法**:
        *   `nli_filter` (Natural Language Inference): NLIモデルを使い、検索結果と元の質問文が「矛盾」していないかを判定し、矛盾する情報をフィルタリングする。

## ディレクトリ構成 (Directory Structure)

各ディレクトリが、上記の手法を組み合わせた一つの独立したRAGパイプラインを表します。ディレクトリ名は、採用している手法を連結したものです。

```
Rehab_RAG/
├── source_documents/
│   └── (ガイドラインなどのMarkdownファイル)
│
├── structured_semantic_chunk-hyde_prf-chromadb_sentencetransformers-nli_filter/
│   ├── build_database.py     # チャンキングとベクトルDB構築用のスクリプト
│   ├── query_rag.py          # RAGパイプラインを実行し、質問応答を行うスクリプト
│   └── db/                   # この手法で使用するChromaDBのデータ
│
├── (今後追加される別の手法のディレクトリ...)
│   ├── ...
│
└── README.md
```

## セットアップ (Setup)

### 1. リポジトリのクローン

```bash
git clone https://github.com/YouSayH/Rehab_RAG.git
cd Rehab_RAG
```

### 2. 仮想環境の作成と有効化

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. 必要なライブラリのインストール

各手法のディレクトリに移動し、`requirements.txt`（もしあれば）を使ってインストールしてください。なければ、以下の主要なライブラリをインストールします。

```bash
pip install -r requirements.txt
```

### 4. APIキーの設定

プロジェクトのルートディレクトリに `.env` という名前のファイルを作成し、お使いのGoogle AI (Gemini) のAPIキーを記述します。

```.env
GEMINI_API_KEY="ここにあなたのAPIキーを貼り付けてください"
```

### 5. 知識ソースの配置

`source_documents` ディレクトリに、知識ベースとしたいMarkdownファイルを配置してください。

## 使い方 (Usage)

各手法は、以下の2ステップで実行します。

1.  **データベースの構築**: `source_documents` の内容をベクトル化します。
2.  **RAGパイプラインの実行**: 対話形式で質問応答を行います。

### 実行例

試したい手法のディレクトリに移動して、以下のコマンドを実行します。

```bash
# (例) 今回の手法のディレクトリに移動
cd structured_semantic_chunk-hyde_prf-chromadb_sentencetransformers-nli_filter

# 1. データベースの構築（初回のみ、または文書更新時に実行）
python build_database.py

# 2. RAGパイプLINEの実行
python query_rag.py
```

## 今後の計画 (Roadmap)

*   [ ] **ベースライン手法の実装**:
    *   単純な固定長チャンキングなど、基本的なRAGパイプラインを実装し、比較の基準点とする。
*   [ ] **異なる手法の追加**:
    *   チャンキング: `RecursiveCharacterTextSplitter` など
    *   クエリ拡張: `Multi-Query Retriever` など
    *   後処理: `Re-Ranking` モデルの導入
*   [ ] **評価指標の導入**:
    *   各手法の精度（Faithfulness, Answer Relevancy）、速度、コストを定量的に評価するスクリプトを追加する。
*   [ ] **異なるEmbeddingモデル、LLMの検証**:
    *   モデルの変更による性能の変化を記録・比較する。

---