# index_data.py
import pandas as pd
import chromadb
import os
import shutil
from sentence_transformers import SentenceTransformer

# ၁။ Database အဟောင်းရှိလျှင် ဖျက်ထုတ်ပစ်ရန် (Clean Start ဖြစ်စေရန်)
db_path = "./ybs_vector_db"
if os.path.exists(db_path):
    print(f"Cleaning up old database at {db_path}...")
    shutil.rmtree(db_path)

# ၂။ Load Data (tsv format ဖြစ်သည့်အတွက် sep='\t' သုံးရပါမည်)
# သင့် file path က 'data/stops.tsv' ဖြစ်ကြောင်း သေချာပါစေ
df = pd.read_csv("data/stops.tsv", sep='\t')

# ၃။ Setup Models & ChromaDB
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
client = chromadb.PersistentClient(path=db_path)
collection = client.create_collection(name="ybs_stops")

print(f"Starting indexing for {len(df)} stops...")

# ၄။ Indexing Loop
for index, row in df.iterrows():
    # နေရာအမည်၊ လမ်းအမည်နှင့် မြို့နယ်တို့ကို ပေါင်းစပ်၍ vector ပြုလုပ်မည်
    combined_text = f"{str(row['name_mm'])} {str(row['name_en'])} {str(row['road_mm'])} {str(row['township_mm'])}"

    embedding = model.encode(combined_text).tolist()

    # Data သိမ်းဆည်းသည့်အခါ metadata ထဲတွင် နာမည်များကို သီးသန့်ခွဲထည့်ရမည်
    collection.add(
        ids=[str(row['id'])],
        embeddings=[embedding],
        documents=[combined_text],
        metadatas=[{
            "id": int(row['id']),
            "name_mm": str(row['name_mm']),  # Precise search အတွက် အဓိက လိုအပ်သည်
            "name_en": str(row['name_en']),
            "township": str(row['township_mm'])
        }]
    )

    if (index + 1) % 100 == 0:
        print(f"✅ Processed {index + 1} / {len(df)} stops...")

print("\n🚀 Indexing Complete! 'ybs_vector_db' folder ကို အသစ်ဖန်တီးပြီးပါပြီ။")