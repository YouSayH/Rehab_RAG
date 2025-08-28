import chromadb
import os

class ChromaDBRetriever:
    """
    [手法解説: Vector Database & Retrieval]
    ベクトル化された情報を保存し、高速に検索するためのデータベースとのやり取りを担当します。
    ChromaDBはオープンソースで手軽に始められるベクトルDBの一つです。

    役割:
    - (データベース構築時) EmbeddingされたチャンクをID、メタデータと共に保存（インデックス化）。
    - (クエリ実行時) 質問文のベクトルを受け取り、それに最も近い（類似度が高い）チャンクをDBから検索（リトリーブ）する。
    
    このコンポーネントは、RAGの「Retrieval(検索)」部分の心臓部です。
    """
    def __init__(self, path: str, collection_name: str, embedder):
        """
        コンストラクタ。ChromaDBに接続し、コレクション（テーブルのようなもの）を準備します。
        
        Args:
            path (str): データベースファイルを保存するディレクトリのパス。
            collection_name (str): データのグループ名。
            embedder: テキストをベクトル化するためのEmbedderインスタンス。
        """
        if not os.path.exists(path):
            os.makedirs(path)
        # `PersistentClient` を使うことで、データベースがファイルとしてディスクに保存され、
        # プログラム終了後もデータが保持されます。
        self.client = chromadb.PersistentClient(path=path)
        self.embedder = embedder

        # コレクションを取得または新規作成します。
        # `metadata={"hnsw:space": "cosine"}` は、ベクトル間の類似度を計算する方法として
        # 「コサイン類似度」を使うという設定です。テキストのセマンティック検索で最も一般的に用いられます。

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, chunks: list[dict], batch_size: int = 100):
        """
        チャンクのリストをデータベースに追加（または更新）します。
        一度に大量のデータを処理するとメモリを圧迫するため、バッチ処理を行います。
        
        Args:
            chunks (list[dict]): チャンク情報の辞書のリスト。
            batch_size (int): 一度に処理するチャンクの数。
        """
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            ids = [chunk["id"] for chunk in batch]
            texts = [chunk["text"] for chunk in batch]
            metadatas = [chunk["metadata"] for chunk in batch]
            
            # Embedderを使ってテキストをベクトルに変換
            embeddings = self.embedder.embed_documents(texts)

            # `upsert` を使うと、IDが既に存在すればデータを更新、なければ新規追加します。
            # これにより、スクリプトを再実行しても安全にデータを追加・更新できます。
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            print(f"  - バッチ {i//batch_size + 1} を処理しました。({min(i+batch_size, len(chunks))}/{len(chunks)})")

    def retrieve(self, query_text: str, n_results: int = 10) -> dict:
        """
        与えられたクエリテキストに意味的に最も類似したドキュメントを検索します。
        
        Args:
            query_text (str): ユーザーからの質問文、またはHyDEで生成された文章。
            n_results (int): 取得する検索結果の数。

        Returns:
            dict: ChromaDBからの検索結果。
        """
        # まずクエリテキスト自体をベクトル化する
        query_embedding = self.embedder.embed_query(query_text)

        # [エラー解決の記録: InvalidArgumentError / 次元数不一致]
        # 課題: 当初、ChromaDBの `.query()` メソッドに `query_texts=[query_text]` を
        #       直接渡していました。この場合、ChromaDBは内部のデフォルトモデル(384次元)で
        #       テキストをベクトル化しようとします。しかし、私たちがDBに格納したデータは
        #       `multilingual-e5-large` (1024次元)で作成されているため、次元数が合わずエラーになりました。
        #
        # 解決策: 検索時も、DB構築時と全く同じEmbedderインスタンスを使ってクエリテキストを
        #         明示的にベクトル化します(`embedder.embed_query`)。
        #         そして、`query_embeddings` 引数を使ってベクトルで検索するように修正しました。
        #         これにより、次元の不一致が解消され、意図通りの検索が可能になりました。
        # ベクトルを使ってデータベースに問い合わせる
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
    
    def count(self) -> int:
        """データベースに保存されているアイテムの総数を返す。"""
        return self.collection.count()