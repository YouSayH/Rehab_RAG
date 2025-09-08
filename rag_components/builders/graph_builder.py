import os
import importlib
from neo4j import GraphDatabase
from tqdm import tqdm
import time
import json
from dotenv import load_dotenv

class GraphBuilder:
    """
    [Builder解説: GraphBuilder]
    テキストからナレッジグラフを構築し、Neo4jに格納するコンポーネント。
    LLMを利用して、非構造化テキストからエンティティ（ノード）と
    その関係性（リレーションシップ）を抽出し、グラフデータベースを構築します。

    [処理の流れ]
    1. Neo4jデータベースに接続する。
    2. Chunkerで分割されたテキストチャンクを一つずつ処理する。
    3. 各チャンクの内容をLLMに渡し、グラフ構造（ノード、関係）をJSON形式で抽出させる。
    4. 抽出された情報を元に、Neo4jにデータを格納するためのCypherクエリを生成する。
    5. Cypherクエリを実行し、知識グラフを構築・更新する。
    """
    def __init__(self, config: dict, db_path: str, **kwargs):
        self.config = config
        
        # .envファイルからNeo4jの接続情報を読み込む
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        
        if not all([uri, user, password]):
            raise ValueError("環境変数 NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD のいずれかが設定されていません。")

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.llm = self._get_instance('llm', params_override={'safety_block_none': True})

    def _get_instance(self, component_type: str, params_override={}):
        """設定に応じてコンポーネントのインスタンスを生成する内部ヘルパー"""
        cfg = self.config['build_components'][component_type]
        params = {**cfg.get('params', {}), **params_override}
        module = importlib.import_module(cfg['module'])
        class_ = getattr(module, cfg['class'])
        return class_(**params)

    def _extract_graph_from_chunk(self, chunk_text: str) -> dict:
        """LLMを使って単一のチャンクからグラフ構造を抽出する"""
        prompt = f"""あなたはテキストから情報を抽出し、ナレッジグラフを構築する専門家です。
以下のテキストから、主要なエンティティ（ノード）とそれらの間の関係（リレーションシップ）を抽出し、指定されたJSON形式で出力してください。

# 指示
- ノードは必ず「id」（テキスト内の名称）と「label」（カテゴリ名、例: 疾患, 治療法, 評価指標）を持ってください。
- 関係は「source」（始点ノードのid）、「target」（終点ノードのid）、「type」（関係の種類、例: HAS_SYMPTOM, EFFECTIVE_FOR）を持ってください。
- 抽出する情報は、テキストの内容に厳密に基づいたものだけにしてください。

# テキスト
"{chunk_text}"

# 出力形式 (JSON)
{{
  "nodes": [
    {{"id": "エンティティ名1", "label": "カテゴリ名1"}},
    {{"id": "エンティティ名2", "label": "カテゴリ名2"}}
  ],
  "relationships": [
    {{"source": "エンティティ名1", "target": "エンティティ名2", "type": "関係タイプ"}}
  ]
}}
"""
        # レート制限を考慮
        time.sleep(1)
        response = self.llm.generate(prompt, max_output_tokens=2048)
        
        try:
            # LLMの出力からJSON部分だけを抽出
            json_part = response[response.find('{'):response.rfind('}')+1]
            return json.loads(json_part)
        except (json.JSONDecodeError, IndexError):
            print(f"警告: LLMの出力からJSONの解析に失敗しました。スキップします。出力: {response}")
            return None

    def _write_to_neo4j(self, graph_data: dict):
        """抽出したグラフデータをNeo4jに書き込む"""
        if not graph_data or not graph_data.get('nodes'):
            return

        with self.driver.session() as session:
            # ノードの作成 (MERGEは存在しない場合のみ作成)
            for node in graph_data.get('nodes', []):
                session.run("MERGE (n:`{label}` {{id: $id}})".format(label=node['label']), 
                            id=node['id'])

            # 関係の作成
            for rel in graph_data.get('relationships', []):
                session.run(
                    "MATCH (a {{id: $source}}), (b {{id: $target}}) "
                    "MERGE (a)-[r:`{type}`]->(b)".format(type=rel['type']),
                    source=rel['source'], target=rel['target']
                )

    def build(self):
        """ナレッジグラフ構築のメイン処理"""
        # 既存のグラフデータを削除する（毎回クリーンな状態から始めるため）
        with self.driver.session() as session:
            print("既存のグラフデータを削除しています...")
            session.run("MATCH (n) DETACH DELETE n")

        # 1. Chunkerを準備
        chunker = self._get_instance('chunker')
        
        # 2. ドキュメントを読み込み、チャンクに分割
        config_dir = os.path.dirname(self.config['builder']['params']['config_path'])
        source_path = os.path.abspath(os.path.join(config_dir, self.config['source_documents_path']))
        
        print(f"'{source_path}' からドキュメントを読み込み、チャンクを生成中...")
        all_chunks = []
        for filename in os.listdir(source_path):
            if filename.endswith(".md"):
                file_path = os.path.join(source_path, filename)
                chunks = chunker.chunk(file_path)
                all_chunks.extend(chunks)

        # 3. 各チャンクからグラフを抽出し、Neo4jに書き込み
        print(f"合計 {len(all_chunks)} 個のチャンクからナレッジグラフを構築します...")
        for chunk in tqdm(all_chunks, desc="Building Knowledge Graph"):
            graph_data = self._extract_graph_from_chunk(chunk['text'])
            if graph_data:
                self._write_to_neo4j(graph_data)
        
        print("ナレッジグラフの構築が完了しました。")
        self.driver.close()