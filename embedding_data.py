import sqlite3
import chromadb
# from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from tqdm import tqdm

# 初始化 FastEmbedEmbeddings
# embedding_model = FastEmbedEmbeddings()

# 初始化 SentenceTransformerEmbeddings
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 初始化 ChromaDB 客戶端
client = chromadb.PersistentClient("w3c_standards_chromaDB")

# 從 SQLite 資料庫讀取資料
def fetch_material_data():
    """
    從 SQLite 資料庫中讀取資料
    """
    conn = sqlite3.connect("w3c_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT Title, Content FROM w3c_standards")
    rows = cursor.fetchall()
    conn.close()
    return rows

# 生成嵌入並存入 ChromaDB
def process_and_store_embeddings(data):
    """
    生成嵌入並將資料存儲到 ChromaDB
    """
    # 創建或獲取 Chroma 集合
    collection = client.create_collection("w3c_standards_collection")

    for title, content in tqdm(data, desc="處理資料", total=len(data)):
        # 生成文本嵌入
        embedding = embedding_model.embed_documents([content])[0]
        
        # 插入嵌入向量及其元數據到 ChromaDB
        collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[{"filename": title}],
            ids=[title]  # 使用文件名稱作為唯一標識
        )
        print(f"成功將檔案 {title} 的嵌入向量存入 ChromaDB")

if __name__ == "__main__":
    # 從資料庫讀取資料
    material_data = fetch_material_data()
    
    if material_data:
        # 處理並儲存到 ChromaDB
        process_and_store_embeddings(material_data)
        print("所有資料已成功存入 ChromaDB")
    else:
        print("資料庫中無資料")
