### セットアップ手順

#### **ステップ1：リポジトリのクローンと移動**

```bash
git clone https://github.com/YouSayH/Rehab_RAG.git
cd Rehab_RAG
```

#### **ステップ2：Python仮想環境の作成と有効化**

```bash
# Windows
python -m venv venv_Rehab_RAG
.\venv_Rehab_RAG\Scripts\activate

# macOS / Linux
python3 -m venv venv_Rehab_RAG
source venv_Rehab_RAG/bin/activate
```

#### **ステップ3：依存ライブラリのインストール**

```bash
pip install -r requirements.txt
```

#### **ステップ4：`.env`ファイルの作成**

APIキーなどの機密情報を格納するため、プロジェクトのルートに`.env`という名前のファイルを手動で作成し、以下の内容を記述します。値はご自身の環境に合わせてください。

```
# --- Google APIキー ---
GOOGLE_API_KEY="your_google_api_key_here" 
```