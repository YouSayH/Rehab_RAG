import os
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

class GraphRetriever:
    """
    [Retriever解説: GraphRetriever]
    ユーザーの自然言語の質問をCypherクエリに変換し、Neo4jナレッジグラフから
    回答の根拠となる情報を検索するコンポーネント。
    内部でLangChainの機能を利用し、Text-to-Cypher変換と検索を自動化します。

    [処理の流れ]
    1. Neo4jデータベースに接続し、グラフのスキーマ（ノードのラベル、関係のタイプなど）を読み込む。
    2. ユーザーの質問とグラフスキーマをLLMに渡し、最適なCypherクエリを生成させる。
    3. 生成されたCypherクエリをNeo4jで実行する。
    4. クエリの実行結果を、最終的な回答のコンテキストとして返す。
    """
    def __init__(self, path: str, collection_name: str, embedder, **kwargs):
        # .envファイルからNeo4jとGeminiの接続情報を読み込む
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        api_key = os.getenv("GEMINI_API_KEY")

        if not all([uri, user, password, api_key]):
            raise ValueError("環境変数 NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GEMINI_API_KEY のいずれかが設定されていません。")

        # LangChainのNeo4jGraphオブジェクトを初期化
        self.graph = Neo4jGraph(
            url=uri, username=user, password=password
        )
        # グラフのスキーマ情報をリフレッシュ（LLMがクエリを生成するために必要）
        self.graph.refresh_schema()

        # Text-to-Cypher変換とQAのためのLLMを初期化
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key,
                                     temperature=0.0)
        
        # LangChainのGraphCypherQAChainをセットアップ
        self.cypher_chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=self.graph,
            verbose=True, # デバッグ用に生成されたCypherクエリをコンソールに表示
            return_intermediate_steps=True # 中間ステップ（クエリなど）も返す
        )
        print("Graph Retrieverが初期化され、Neo4jに接続しました。")

    def retrieve(self, query_text: str, n_results: int = 10) -> dict:
        """
        自然言語の質問からCypherクエリを生成し、グラフから情報を検索する。
        """
        print(f"  - クエリ '{query_text}' をCypherに変換して検索中...")
        
        result = self.cypher_chain.invoke({"query": query_text})

        # LangChainの出力を、既存のRAGパイプラインの形式に合わせる
        # ここでは、中間ステップのコンテキスト（検索結果）を文書として扱う
        context_str = str(result['intermediate_steps'][0]['context'])
        
        # 結果が空でないかチェック
        if not context_str or "[]" in context_str:
            return {'documents': [[]], 'metadatas': [[]]}

        # シンプルに検索結果の文字列をドキュメントとして返す
        # メタデータは、グラフ検索であることを示す情報を付与
        final_documents = [context_str]
        final_metadatas = [{"source": "Knowledge Graph"}]

        return {
            'documents': [final_documents],
            'metadatas': [final_metadatas]
        }

    def add_documents(self, chunks: list[dict]):
        """
        GraphRetrieverでは、検索のみを担当するため、このメソッドは何もしません。
        実際のデータ追加はGraphBuilderが担当します。
        """
        print("GraphRetrieverのadd_documentsは呼び出されましたが、処理はGraphBuilderが担当するためスキップします。")
        pass