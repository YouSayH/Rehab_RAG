import os
import re
import chromadb
from sentence_transformers import SentenceTransformer
import hashlib
import shutil

# --- RAG学習用コメント ----
# このスクリプトは、RAG (Retrieval-Augmented Generation) パイプラインの根幹である
# 「知識ベース（ベクトルデータベース）」を構築する役割を担います。
# 
# 処理の大きな流れ：
# 1. source_documents/ ディレクトリからMarkdownファイル（知識ソース）を読み込む。
# 2. 各ファイルを意味のあるかたまり（チャンク）に分割する（チャンキング）。
# 3. 各チャンクに、それが文書のどの部分かを示す情報（メタデータ）を付与する。
# 4. Sentence Transformerモデル（Embeddingモデル）を使い、各チャンクのテキストを「意味を捉えたベクトル」に変換する。
# 5. ベクトル、元のテキスト、メタデータをセットにして、ベクトルデータベース（ChromaDB）に保存（インデックス化）する。
#
# このデータベースが完成することで、質問文と意味的に近い情報を高速に検索できるようになります。
# -------------------------


# 使用するEmbeddingモデル
# テキストをベクトルに変換するためのモデル。日本語を含む多言語に対応し、
# 高性能な`intfloat/multilingual-e5-large`を選択。このモデルは1024次元のベクトルを生成します。
MODEL_NAME = "intfloat/multilingual-e5-large"

# ドキュメントが格納されているディレクトリ
DOCUMENTS_PATH = "../source_documents"

# ChromaDBのデータベースを保存するパス
# 各RAG手法の実験を独立させるため、このスクリプトと同じディレクトリ内に'db'フォルダを作成します。
CHROMA_PATH = "./db"

# ChromaDBのコレクション名
# コレクションは、リレーショナルデータベースにおける「テーブル」のようなものです。
COLLECTION_NAME = "rag_guidelines"


def parse_markdown_to_chunks(file_path):
    """
    Markdownファイルを解析し、構造に基づいたチャンクとメタデータのリストを生成する。
    この関数は「ハイブリッド・チャンキング戦略」の第一段階（構造化チャンキング）を担います。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    chunks = []
    
    # 章タイトルを抽出
    chapter_match = re.search(r'^#\s*(.*?)\n', content)
    chapter_title = chapter_match.group(1).strip() if chapter_match else "不明な章"

    # H2見出し(##)を「疾患」の区切りとみなし、文書を大きなセクションに分割。
    # これにより、「大腿骨近位部骨折」と「変形性股関節症」の情報が混ざるのを防ぎ、文脈を維持します。
    sections = re.split(r'\n(##\s)', content)
    
    current_content = sections[0]
    current_disease = "疾患総論"

    # ファイル内でユニークなIDを振るためのカウンター
    chunk_counter = 0
    chunk_counter = process_section(chunks, current_content, file_path, chapter_title, current_disease, chunk_counter)

    for i in range(1, len(sections), 2):
        current_content = sections[i] + sections[i+1]
        
        disease_match = re.search(r'##\s*(.*?)\n', current_content)
        if disease_match:
            current_disease = disease_match.group(1).strip()
        
        chunk_counter = process_section(chunks, current_content, file_path, chapter_title, current_disease, chunk_counter)
        
    return chunks

def process_section(chunks_list, section_content, file_path, chapter, disease, start_index):
    """
    疾患ごとのセクションをさらにチャンクに分割し、メタデータを付与する。
    ハイブリッド・チャンキングの第二段階（段落単位での分割）とメタデータ付与を行います。
    """
    # 2つ以上の連続した改行を段落の区切りとみなし、セクションをさらに細かく分割します。
    paragraphs = re.split(r'\n{2,}', section_content)
    
    current_section = "N/A"
    current_subsection = "N/A"
    
    chunk_index = start_index

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # ヘッダー情報を抽出し、メタデータとして利用
        h3_match = re.search(r'^###\s*(.*?)\n', paragraph)
        h4_match = re.search(r'^####\s*(.*?)\n', paragraph)

        if h3_match:
            cq_bq_match = re.search(r'(Clinical Question|BQ)\s*(\d+)', h3_match.group(1), re.IGNORECASE)
            if cq_bq_match:
                current_section = f"{cq_bq_match.group(1).upper()} {cq_bq_match.group(2)}"
            else:
                current_section = h3_match.group(1).strip()
            current_subsection = "N/A"

        if h4_match:
            current_subsection = h4_match.group(1).strip()

        # 短すぎるチャンクはノイズになる可能性があるため除外
        if len(paragraph.split()) < 5:
            continue

        # RAGの検索精度を向上させるための重要なメタデータを作成
        metadata = {
            "source": os.path.basename(file_path),
            "chapter": chapter,
            "disease": disease,
            "section": current_section,
            "subsection": current_subsection
        }
        
        # --- [エラー回避策 1: DuplicateIDError] --------
        # 原因：以前は `paragraph` (テキスト内容) のみからIDを生成していました。
        #      そのため、異なるファイルや同じファイル内の異なる場所に全く同じテキストが存在すると、
        #      IDが重複してしまい、ChromaDBがエラーを発生させていました。
        #
        # 対策：IDの生成に「ファイルパス」と「ファイル内での連番(chunk_index)」を追加しました。
        #      これにより、たとえテキスト内容が同じでも、由来する場所が違えば必ずユニークなIDが
        #      生成されるようになり、IDの重複が完全に防がれます。
        # ----------------------------------------------
        unique_string = f"{file_path}:{chunk_index}:{paragraph}"
        chunk_id = hashlib.sha256(unique_string.encode()).hexdigest()

        chunks_list.append({
            "id": chunk_id,
            "text": paragraph,
            "metadata": metadata
        })
        
        chunk_index += 1

    return chunk_index


def create_or_update_database():
    """
    ChromaDBデータベースを作成または更新するメイン関数。
    """
    # 開発・実験中は、コードを変更して再実行するたびにクリーンな状態から始めるのが安全です。
    # そのため、既存のデータベースフォルダがあれば一度削除しています。
    if os.path.exists(CHROMA_PATH):
        print(f"既存のデータベース '{CHROMA_PATH}' を削除します。")
        shutil.rmtree(CHROMA_PATH)
        
    print("--- データベース構築開始 ---")
    
    print(f"Embeddingモデル ({MODEL_NAME}) をロード中...")
    model = SentenceTransformer(MODEL_NAME)
    print("モデルのロード完了。")

    # `PersistentClient` を使うことで、データベースがファイルとしてディスクに保存され、
    # スクリプト終了後もデータが保持されます。
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # コレクションを取得または新規作成します。
    # `metadata={"hnsw:space": "cosine"}` は、ベクトル間の類似度を計算する方法として
    # 「コサイン類似度」を使うという設定です。これはセマンティック検索で一般的に用いられます。
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    all_chunks = []
    print(f"'{DOCUMENTS_PATH}' からドキュメントを読み込みます...")
    for filename in os.listdir(DOCUMENTS_PATH):
        if filename.endswith(".md"):
            file_path = os.path.join(DOCUMENTS_PATH, filename)
            print(f"\nファイル '{filename}' を処理中...")
            
            chunks = parse_markdown_to_chunks(file_path)
            all_chunks.extend(chunks)
            print(f"-> {len(chunks)} 個のチャンクを抽出しました。")
    
    if not all_chunks:
        print(f"警告: '{DOCUMENTS_PATH}' 内に処理対象のMarkdownファイルが見つかりませんでした。")
        return

    print(f"\n合計 {len(all_chunks)} 個のチャンクをデータベースに格納します。")

    # 一度に全てのデータを処理するとメモリを大量に消費するため、バッチ処理で分割して投入します。
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        
        ids = [chunk["id"] for chunk in batch]
        texts = [chunk["text"] for chunk in batch]
        metadatas = [chunk["metadata"] for chunk in batch]
        
        # ここで実際にテキストがベクトルに変換されます。
        embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False).tolist()

        # `upsert` を使うと、IDが既に存在すればデータを更新、なければ新規追加します。
        # これにより、スクリプトを再実行しても安全にデータを追加・更新できます。
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        print(f"  - バッチ {i//batch_size + 1} を処理しました。({min(i+batch_size, len(all_chunks))}/{len(all_chunks)})")

    print("\n--- データベース構築完了 ---")
    print(f"データベースのパス: {os.path.abspath(CHROMA_PATH)}")
    print(f"コレクション名: {COLLECTION_NAME}")
    print(f"格納されたアイテム数: {collection.count()}")
    
    print("\n--- データベース確認クエリ ---")
    # --- [エラー回避策 2: InvalidArgumentError] ----------
    # 原因：以前は `query_texts` を使って生のテキストで検索していました。
    #      この場合、ChromaDBは内部のデフォルトモデル(`all-MiniLM-L6-v2`、384次元)を使って
    #      テキストをベクトル化しようとします。しかし、私たちがDBに格納したデータは
    #      `multilingual-e5-large` (1024次元)で作成されているため、次元数が合わずエラーになりました。
    #
    # 対策：検索時も、DB構築時と同じ `model` オブジェクトを使ってクエリテキストを明示的に
    #      ベクトル化します。そして、`query_embeddings` を使ってベクトルで検索するように修正しました。
    #      これにより、次元の不一致が解消されます。
    # ----------------------------------------------------
    query_text = ["変形性股関節症の理学療法について"]
    query_embedding = model.encode(query_text).tolist()

    results = collection.query(
        query_embeddings=query_embedding, # `query_texts` の代わりに `query_embeddings` を使用
        n_results=2
    )
    
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"\n[検索結果 {i+1}]")
        print(f"  - 出典: {meta.get('source', 'N/A')}")
        print(f"  - 疾患: {meta.get('disease', 'N/A')}")
        print(f"  - セクション: {meta.get('section', 'N/A')}")
        print(f"  - テキスト抜粋: {doc[:150]}...")

if __name__ == "__main__":
    if not os.path.exists(DOCUMENTS_PATH):
        print(f"エラー: '{DOCUMENTS_PATH}' ディレクトリが見つかりません。")
        print("プロジェクトルートに 'source_documents' フォルダを作成し、Markdownファイルを入れてください。")
    else:
        create_or_update_database()