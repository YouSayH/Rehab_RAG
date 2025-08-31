# Rehab\_RAG: 理学療法ガイドラインのためのRAG手法評価・検証リポジトリ

[](https://github.com/YouSayH/Rehab_RAG)

## 概要 (Overview)

このリポジトリは、理学療法（リハビリテーション）領域の臨床ガイドラインなどの専門文書に対し、**最適なRAG (Retrieval-Augmented Generation) パイプラインを体系的に探求・評価する**ための実験フレームワークです。

最終的な目標は、臨床現場の専門家が持つ疑問に対し、膨大なガイドラインの中から、迅速かつ正確に、そして根拠を持って回答を提示できるAIアシスタントを構築することです。

### 🎯 このプロジェクトの特徴

  * **🧪 コンポーネント化された設計**: RAGの各プロセス（チャンキング、検索、リランキング等）を独立した部品（コンポーネント）として管理。
  * **⚙️ 柔軟な実験設定**: `config.yaml`ファイルを編集するだけで、様々なコンポーネントの組み合わせを簡単に試し、手法のON/OFFを切り替えられます。
  * **📊 定量的な評価**: `Ragas`フレームワークを導入し、「回答の忠実性」や「文脈の再現率」といった客観的な指標で各パイプラインの性能を評価します。

-----

## 📂 ディレクトリ構成

このプロジェクトは、RAGの各コンポーネントをモジュールとして管理し、それらを柔軟に組み合わせて実験・評価できるよう設計されています。

```
Rehab_RAG/
│
├── 📁 source_documents/
│   └── 📄 脳卒中理学療法_ガイドライン.md
│
├── 📁 rag_components/
│   ├── 📁 chunkers/
│   ├── 📁 embedders/
│   ├── 📁 query_enhancers/
│   ├── 📁 retrievers/
│   ├── 📁 rerankers/
│   ├── 📁 filters/
│   └── 📁 llms/
│
├── 📁 experiments/
│   └── 📁 <実験名>/
│       ├── 📜 config.yaml
│       ├── 🐍 build_database.py
│       ├── 🐍 query_rag.py
│       └── 🗃️ db/
│
├── 📁 evaluation/
│   ├── 🐍 evaluate_rag.py
│   ├── 📄 test_dataset.jsonl
│   └── 📁 logs/
│
├── 🔑 .env
└── 📋 requirements.txt
```

### \#\#\# 各要素の詳細

#### `📁 source_documents/`

**RAGシステムの知識源となるMarkdownファイル**を格納します。ここに置かれた文書がチャンキング（分割）され、ベクトルデータベースに登録されます。

#### `📁 rag_components/`

RAGパイプラインを構成する各機能が、**再利用可能なPythonクラスとしてモジュール化**されています。新しい手法を試す際は、このディレクトリに新しいコンポーネントを追加します。

  * `chunkers/`: テキストを意味のある塊（チャンク）に分割するロジック。
  * `embedders/`: テキストをベクトルに変換するモデルのラッパー。
  * `query_enhancers/`: ユーザーの質問を検索用に拡張・変換する処理。
  * `retrievers/`: ベクトルDBとのやり取り（保存・検索）を担当。
  * `rerankers/`: 検索結果をより精度の高いモデルで並べ替える処理。
  * `filters/`: 検索結果からノイズや矛盾する情報を除去する処理。
  * `llms/`: 回答生成モデル（Geminiなど）とのAPI通信を管理するラッパー。

#### `📁 experiments/`

様々なRAGコンポーネントの組み合わせ（＝パイプライン）を試すための**実験室**です。各サブディレクトリが、それぞれ独立した一つの実験設定に対応します。

  * `📁 <実験名>/`: ディレクトリ名は、その実験で採用している手法の組み合わせを表します。
      * `📜 config.yaml`: その実験で使用する**コンポーネントやモデルを指定する、最も重要な設定ファイル**です。
      * `🐍 build_database.py`: `config.yaml`に従い、ベクトルDBを構築するスクリプト。
      * `🐍 query_rag.py`: `config.yaml`で定義されたパイプラインを使い、対話形式で動作確認するスクリプト。
      * `🗃️ db/`: `build_database.py`によって構築されたChromaDBのデータが保存される場所。

#### `📁 evaluation/`

構築したRAGパイプラインの性能を**定量的に評価するための専用ディレクトリ**です。

  * `🐍 evaluate_rag.py`: 指定した実験設定（パイプライン）を、テストデータセットを用いて自動評価するスクリプト。
  * `📄 test_dataset.jsonl`: 評価に使用する「質問」と「理想的な回答（正解データ）」のペアを格納したファイル。
  * `📁 logs/`: `evaluate_rag.py`を実行した際の評価スコア（CSV形式）と詳細な実行ログが保存されます。

-----

## セットアップ (Setup)

1.  **リポジトリのクローン**

    ```bash
    git clone https://github.com/YouSayH/Rehab_RAG.git
    cd Rehab_RAG
    ```

2.  **仮想環境の作成と有効化**

    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **必要なライブラリのインストール**

    ```bash
    pip install -r requirements.txt
    ```

4.  **APIキーの設定**
    プロジェクトのルートディレクトリに `.env` という名前のファイルを作成し、お使いのAPIキーを記述します。

    ```.env
    GEMINI_API_KEY="ここにあなたのAPIキーを貼り付けてください"
    ```

5.  **（ハイブリッド検索利用時）MeCabのインストール**
    ハイブリッド検索機能 (`HybridRetriever`) を使用するには、形態素解析エンジンMeCabのインストールが別途必要です。

    a. **MeCab本体のインストール**

      * **Windows**: [こちらのリリース](https://github.com/ikegami-yukino/mecab/releases/tag/v0.996.2)から `mecab-0.996-64.exe` 等のインストーラーをダウンロードして実行します。**必ず辞書は `UTF-8` を選択してください。**
      * **Mac**: `brew install mecab mecab-ipadic`
      * **Linux**: `sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8`

    b. **環境変数の設定 (Windowsのみ)**

    Windowsでは、`mecab-python3`がMeCab本体を見つけられるように、システム環境変数を設定する必要があります。

      * **変数名**: `MECABRC`
      * **変数値**: `C:\Program Files\MeCab\etc\mecabrc` （※インストール先のパスに合わせて変更してください）

-----

## 使い方 (Usage)

RAGパイプラインの実験と評価は、以下の4ステップで行います。

### ステップ1: 実験内容の定義 (`config.yaml`)

`experiments/<実験名>/config.yaml` を開き、使用したいコンポーネントを定義します。

**手法のON/OFFは、該当セクションをコメントアウトするだけ**で簡単に行えます。

```yaml
query_components:
  # ... (llm, embedder, retrieverは必須)

  # [クエリ拡張] HyDEを無効にする場合は、以下の3行をコメントアウト
  query_enhancer:
    module: rag_components.query_enhancers.hyde_generator
    class: HydeQueryEnhancer

  # [リランキング] Rerankerを無効にする場合は、以下の4行をコメントアウト
  reranker:
    module: rag_components.rerankers.cross_encoder_reranker
    class: CrossEncoderReranker
    params:
      model_name: "BAAI/bge-reranker-v2-m3"

  # [フィルタリング] NLI Filterを無効にする場合は、以下の4行をコメントアウト
  filter:
    module: rag_components.filters.nli_filter
    class: NLIFilter
    params:
      model_name: "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
```

### ステップ2: データベースの構築 (`build_database.py`)

`config.yaml`で定義したチャンカーとエンベッダーを使い、知識源からベクトルDBを構築します。

```bash
# プロジェクトのルートディレクトリから実行
python .\experiments\<実験名>\build_database.py
```

### ステップ3: パイプラインの動作確認 (`query_rag.py`)

構築したパイプラインが意図通りに動くか、対話形式で定性的に確認します。

```bash
# プロジェクトのルートディレクトリから実行
python .\experiments\<実験名>\query_rag.py
```

### ステップ4: 定量的評価の実行 (`evaluate_rag.py`)

テストデータセットを使い、RAGパイプラインの性能を客観的なスコアで評価します。

```bash
# プロジェクトのルートディレクトリから実行
python evaluation/evaluate_rag.py experiments/<実験名>

# (オプション) 最初の3件だけを評価する場合
python evaluation/evaluate_rag.py experiments/<実験名> --limit 3
```

実行後、`evaluation/logs/`に`scores_...csv`と`log_...log`が生成されます。

-----

## 評価指標 (Evaluation with Ragas)

このプロジェクトでは、`Ragas`フレームワークを用いてRAGパイプラインの性能を多角的に評価します。主な指標は以下の通りです。

| 指標 | 評価する内容 |
| :--- | :--- |
| **Faithfulness** | 生成された回答が、検索された文脈（参考情報）にどれだけ忠実か。ハルシネーション（幻覚）の度合いを測る。 |
| **Answer Relevancy** | 生成された回答が、元の質問にどれだけ的確か。質問の意図を汲み取れているかを測る。 |
| **Context Precision** | 検索された文脈の中で、実際に回答生成に必要だった情報の割合。検索結果のノイズの少なさを測る。 |
| **Context Recall** | 正解データと比較して、検索された文脈が回答に必要な情報をどれだけ網羅できているか。検索の再現率を測る。 |

これらのスコアを比較することで、どのコンポーネントの組み合わせが最も優れたパイプラインを構築できるかを判断します。

-----

## RAGコンポーネント解説

このプロジェクトで使用されている主要なコンポーネントです。

| フェーズ | コンポーネント | 手法解説 |
| :--- | :--- | :--- |
| **1. Chunking** | `StructuredMarkdownChunker` | 単純な文字数ではなく、Markdownの見出し構造（章、節）を維持しながら意味のある段落単位で分割。文脈が破壊されるのを防ぎ、検索精度を高めます。 |
| **2. Embedding** | `SentenceTransformerEmbedder` / `GeminiEmbedder` | テキストチャンクを、意味的な類似度を計算できる数値ベクトルに変換します。モデルを変えることで性能や特性を比較できます。 |
| **3. Retrieval** | `ChromaDBRetriever` / `HybridRetriever` | **ChromaDB**: ベクトル化された情報を保存・管理し、質問ベクトルと最も意味が近い文書を高速に検索します。\<br\>**Hybrid**: キーワード検索(BM25)とベクトル検索を組み合わせ、両者の長所を活かして検索精度を向上させます。 |
| **4. Query Expansion** | `HydeQueryEnhancer` | ユーザーの短い質問から、LLMが架C空の理想的な回答（Hypothetical Document）を生成。この架空の回答で検索することで、検索キーワードを補い精度を向上させます。 |
| **5. Reranking** | `CrossEncoderReranker` | Retrieverが高速に絞り込んだ候補文書を、より計算コストが高いが高精度なCross-Encoderモデルで再評価。「質問」と「文書」をペアで比較するため、文脈的な関連性をより正確に捉え、ランキングを最適化します。 |
| **6. Filtering** | `NLIFilter` | 自然言語推論（NLI）モデルを使い、検索結果が元の質問と「矛盾(Contradiction)」していないかを判定。矛盾する、あるいは無関係な情報を除去し、LLMに渡す情報の品質を高めます。 |
| **7. Generation** | `GeminiLLM` | 最終的に絞り込まれた高品質な文脈（参考情報）と元の質問を元に、LLM（Gemini）がユーザーへの最終的な回答を生成します。 |

-----

## 🗺️ 今後の検証ロードマップ (Future Validation Roadmap)

このプロジェクトでは、以下のステップで段階的にRAGパイプラインを高度化させていきます。

### STEP 1: ハイブリッド検索 (Hybrid Search) の導入

  * **手法**: キーワード検索 (BM25) とベクトル検索を組み合わせ、Reciprocal Rank Fusion (RRF) で結果を統合する。
  * **目的**: ベクトル検索が苦手な固有名詞や専門用語の検索精度を向上させる。

### STEP 2: 複数クエリ生成 (Multi-Query Retriever)

  * **手法**: 1つの複雑な質問をLLMで複数のサブ質問に分解し、それぞれで検索を実行して結果を統合する。
  * **目的**: 多角的な質問に対する情報の網羅性を高め、回答の漏れを防ぐ。

### STEP 3: RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)

  * **手法**: 文書群を再帰的にクラスタリング＆要約し、知識を階層的なツリー構造で表現する。
  * **目的**: 質問の抽象度に応じて、要約から詳細なチャンクまで、最適な粒度の情報を提供できるようにする。

### STEP 4: Self-RAG (Self-Reflective RAG)

  * **手法**: LLM自身が「検索は必要か？」「検索結果は適切か？」などを自己評価しながら、動的にプロセスを制御する。
  * **目的**: パイプラインに自律性をもたらし、不要な処理をスキップしたり、不適切な情報を自己判断で破棄することで、効率と信頼性を向上させる。

### STEP 5: GraphRAG (ナレッジグラフの活用)

  * **手法**: 文書からエンティティと関係性を抽出し、ナレッジグラフを構築。グラフを探索して回答を生成する。
  * **目的**: 文書単体では見えにくい、エンティティ間の複雑な関係性を捉えた高度な推論を可能にする。