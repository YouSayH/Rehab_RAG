import os
import chromadb
import time
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- [RAG学習ポイント] 設定項目 ---
# RAGパイプラインで使用するモデルやデータベースの情報を一元管理します。
# モデル名を変更するだけで、システムの性能比較が容易になります。
# ------------------------------------------
load_dotenv(dotenv_path='../.env')

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# [エラー回避/安定化のポイント]
# gemini-2.5-flashは高性能ですが、指示に対して非常に冗長な回答を生成する傾向があり、
# トークン上限エラー(MAX_TOKENS)が多発しました。
# 一方、gemini-2.5-flash-liteは、より簡潔で指示に忠実な回答を生成するため、
# このRAGパイプラインのような、厳密な制御が求められるタスクに適しています。
# そのため、安定動作した `flash-lite` を正式に採用します。
GENERATION_MODEL_NAME = "gemini-2.5-flash-lite" 

CHROMA_PATH = "./db"
COLLECTION_NAME = "rag_guidelines"

class RAGPipeline:
    """
    [RAG学習ポイント]
    RAGパイプライン全体を管理・実行するクラスです。
    HyDEによるクエリ生成、ベクトル検索、NLIによるフィルタリング、最終的な回答生成という
    一連の流れをメソッドとしてカプセル化し、再利用しやすくしています。
    """
    def __init__(self):
        print("--- RAGパイプラインの初期化を開始 ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用デバイス: {self.device}")

        # [RAG学習ポイント]
        # パイプラインの初期化時に、時間のかかるモデルのロードを一度だけ行います。
        # これにより、クエリ毎にモデルをロードする無駄がなくなり、高速な応答が可能になります。
        print(f"Embeddingモデル ({EMBEDDING_MODEL_NAME}) をロード中...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device)

        print(f"NLIモデル ({NLI_MODEL_NAME}) をロード中...")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(self.device)

        print("ChromaDBに接続中...")
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)

        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("環境変数 'GEMINI_API_KEY' が.envファイルに設定されていません。")
        self.genai_client = genai.Client()
        
        # [エラー回避/安定化のポイント]
        # 医療系の質問は、モデルのセーフティ機能によって回答がブロックされることがあります。
        # このRAGでは、信頼できる情報源のみを参考にしているため、モデル自体の安全フィルタは
        # 無効化(BLOCK_NONE)し、意図した回答が生成されるようにします。
        self.safety_settings = [
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        ]
        
        print(f"--- 初期化完了 (使用モデル: {GENERATION_MODEL_NAME}) ---")

    def generate_hypothetical_answer(self, query: str) -> str:
        """
        [RAG学習ポイント] - HyDE (Hypothetical Document Embeddings)
        ステップ1: 架空の回答生成
        短い質問文で検索するより、質問に対する「理想的な回答」を生成し、その回答文で検索する方が、
        意味的に近い文書を見つけやすいという考え方です。
        ここではLLMに架空の完璧な回答を作らせ、それを次の検索ステップの入力として使います。
        """
        prompt = f"""以下の質問に対して、あなたが完璧な知識を持つ専門家であると仮定して、理想的な回答を生成してください。
この回答は検索の精度を高めるために使います。質問の意図を汲み取った具体的で詳細な文章にしてください。

質問: {query}

理想的な回答:"""

        try:
            response = self.genai_client.models.generate_content(
                model=GENERATION_MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=512,
                    safety_settings=self.safety_settings
                )
            )
            if response.text:
                return response.text
            else:
                print(f"HyDEの生成結果が空です。Finish Reason: {response.candidates[0].finish_reason}")
                return query
        except Exception as e:
            print(f"HyDEの生成中にエラーが発生しました: {e}")
            return query

    def retrieve_documents(self, text_for_retrieval: str, n_results: int = 10) -> dict:
        """
        [RAG学習ポイント] - Retrieval (検索)
        ステップ2: 関連文書の検索
        HyDEで生成した文章（または元の質問文）をEmbeddingモデルでベクトル化し、
        ChromaDB（ベクトルデータベース）に問い合わせて、意味的に類似度の高い文書チャンクを
        複数件（ここでは10件）取得します。
        """
        embedding = self.embedding_model.encode(text_for_retrieval, convert_to_tensor=True).tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )
        return results

    def filter_with_nli(self, query: str, documents: list, metadatas: list) -> tuple[list, list]:
        """
        [RAG学習ポイント] - NLI Filtering (自然言語推論によるフィルタリング)
        ステップ3: 矛盾する情報の除去 (高度なテクニック)
        ベクトル検索では、単語が似ているだけで文脈が全く違う文書も取得されることがあります。
        NLIモデルを使い、「取得した文書(premise)」と「元の質問(hypothesis)」の関係を
        「矛盾(contradiction)」「中立(neutral)」「含意(entailment)」に分類し、
        「矛盾」する文書を除外します。これにより、回答の信頼性を高めます。
        """
        filtered_docs = []
        filtered_metadatas = []
        
        for doc, meta in zip(documents, metadatas):
            premise = doc
            hypothesis = query
            input_data = self.nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.nli_model(**input_data)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            contradiction_score = probabilities[0]
            entailment_score = probabilities[2]
            if contradiction_score < 0.5 and (entailment_score > 0.1 or probabilities[1] > 0.1): 
                filtered_docs.append(doc)
                filtered_metadatas.append(meta)
        
        return filtered_docs, filtered_metadatas

    def construct_prompt(self, query: str, context_docs: list, context_metadatas: list) -> str:
        """
        [RAG学習ポイント] - Prompt Construction (プロンプト構築)
        ステップ4: 最終プロンプトの構築
        検索・フィルタリングした文書を「参考情報」としてプロンプトに埋め込みます。
        LLMに役割を与え、「参考情報のみを元に回答すること」「出典を明記すること」といった
        厳密な指示を与えることで、幻覚(ハルシネーション)を抑制し、根拠のある回答を生成させます。
        """
        context_str = ""
        for i, (doc, meta) in enumerate(zip(context_docs, context_metadatas)):
            source_info = f"出典: {meta.get('source', 'N/A')}, 疾患: {meta.get('disease', 'N/A')}, セクション: {meta.get('section', 'N/A')}"
            context_str += f"[参考情報 {i+1}] ({source_info})\n"
            context_str += f"{doc}\n\n"

        prompt = f"""あなたは理学療法の専門家アシスタントです。
以下の参考情報を主な根拠として、ユーザーの質問に日本語で回答してください。

### 指示
- 回答は参考情報の内容を要約・抽出し、あなたの個人的な知識や意見は含めないでください。
- もし参考情報の中に、質問に答えるための情報が全く見つからない場合にのみ、「参考情報の中に関連する情報が見つかりませんでした。」と回答してください。
- 回答の各文末には、根拠として利用した参考情報の番号を [番号] の形式で必ず付記してください。複数の情報を参考にした場合は [番号1, 番号2] のように記述してください。

### ユーザーの質問
{query}

### 参考情報
{context_str}

### あなたの回答
"""
        return prompt

    def generate_answer(self, prompt: str) -> str:
        """
        [RAG学習ポイント] - Generation (生成)
        ステップ5: 最終回答の生成
        構築したプロンプトをLLMに渡し、最終的な回答を生成させます。
        これまでのステップにより、LLMは必要な情報だけを与えられた状態で回答生成に集中できます。
        """
        # [エラー回避/安定化のポイント]
        # API呼び出しは、ネットワークの問題やサーバー側の負荷で一時的に失敗することがあります(500 Internal Errorなど)。
        # そのため、簡単なリトライ処理を実装することで、システムの安定性を向上させています。
        for attempt in range(2): # 最大2回試行
            try:
                response = self.genai_client.models.generate_content(
                    model=GENERATION_MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=2048,
                        safety_settings=self.safety_settings
                    )
                )
                if response.text:
                    return response.text
                else:
                    finish_reason_name = response.candidates[0].finish_reason.name
                    if finish_reason_name == "MAX_TOKENS":
                        return f"回答を生成できませんでした。理由: {finish_reason_name} (出力トークン上限超過)"
            except Exception as e:
                print(f"回答生成中にエラー発生 (試行 {attempt + 1} 回目): {e}")
                if attempt == 0:
                    time.sleep(3) # 2回目の試行の前に3秒待機
                else:
                    return f"回答の生成中にエラーが繰り返し発生しました: {e}"
        finish_reason_name = response.candidates[0].finish_reason.name
        return f"回答を生成できませんでした。理由: {finish_reason_name}"

    def query(self, query: str):
        """
        RAGパイプライン全体を順番に実行するメインメソッド。
        """
        print(f"\n[ユーザーの質問]: {query}")

        print("\n[ステップ1/5] HyDEで架空の回答を生成中...")
        hypothetical_answer = self.generate_hypothetical_answer(query)
        print(f"  - 生成された架空回答 (検索クエリとして使用):\n'{hypothetical_answer[:150]}...'")

        print("\n[ステップ2/5] 関連文書をChromaDBから検索中...")
        retrieved_results = self.retrieve_documents(hypothetical_answer)
        retrieved_docs = retrieved_results['documents'][0]
        retrieved_metadatas = retrieved_results['metadatas'][0]
        print(f"  - {len(retrieved_docs)}件の文書を取得しました。")

        print("\n[ステップ3/5] NLIモデルで矛盾する情報をフィルタリング中...")
        filtered_docs, filtered_metadatas = self.filter_with_nli(query, retrieved_docs, retrieved_metadatas)
        print(f"  - フィルタリング後、{len(filtered_docs)}件の文書が残りました。 ({len(retrieved_docs) - len(filtered_docs)}件を除外)")

        if not filtered_docs:
            print("\n[最終回答]\n参考情報の中に関連する情報が見つかりませんでした。")
            return
            
        print("\n[ステップ4/5] LLM用のプロンプトを構築中...")
        # [エラー回避/安定化のポイント]
        # 参考情報が多すぎると、モデルが混乱したりトークン上限に達する原因になります。
        # 検索結果の中から、フィルタリングを通過し、かつ類似度が高い上位5件に絞ることで、
        # プロンプトの品質と安定性を両立させています。
        top_k = 5
        final_docs = filtered_docs[:top_k]
        final_metadatas = filtered_metadatas[:top_k]
        print(f"  - 最も関連性の高い上位{len(final_docs)}件を最終プロンプトに使用します。")
        final_prompt = self.construct_prompt(query, final_docs, final_metadatas)

        print("\n[ステップ5/5] LLMで最終回答を生成中...")
        # [エラー回避/安定化のポイント]
        # 無料利用枠のAPIは、1分間あたりのリクエスト数に制限があります。
        # 各ステップでAPIを呼び出すため、短い待機時間を設けることで、
        # レート制限エラー(429 RESOURCE_EXHAUSTED)の発生を防ぎます。
        time.sleep(1)
        final_answer = self.generate_answer(final_prompt)
        
        print("\n" + "="*50)
        print("[最終回答]")
        print(final_answer)
        print("="*50 + "\n")


if __name__ == "__main__":
    try:
        pipeline = RAGPipeline()
        test_queries = [
            "大腿骨近位部骨折の術後、理学療法の頻度を増やすとどうなりますか？",
            "変形性股関節症の発症を予防するには、運動療法だけで十分ですか？",
            "脳卒中患者に対する有酸素運動は推奨されますか？",
            "肩関節周囲炎の炎症期に、痛みを我慢して積極的に運動したほうがいいですか？"
        ]
        for q in test_queries:
            pipeline.query(q)
            time.sleep(1) # レート制限対策
        print("\n\n対話モードを開始します。終了するには 'q' または 'exit' と入力してください。")
        while True:
            user_input = input("\n質問を入力してください: ")
            if user_input.lower() in ["q", "exit"]:
                print("終了します。")
                break
            if user_input:
                pipeline.query(user_input)
    except Exception as e:
        print(f"パイプラインの実行中にエラーが発生しました: {e}")