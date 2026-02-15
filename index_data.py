import json
import chromadb
from sentence_transformers import SentenceTransformer

# ၁။ Model နှင့် Database Connection တည်ဆောက်ခြင်း
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
client = chromadb.PersistentClient(path="./ybs_vector_db")
collection = client.get_or_create_collection(name="ybs_stops")


# ၂။ Bus Stops များကို Index လုပ်ခြင်း (JSON မှ)
def index_bus_stops(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_stops = set()
        for line in data["ybs_bus_lines"]:
            for stop in line["stops"]:
                all_stops.add(stop.strip())

        for idx, stop_name in enumerate(all_stops):
            embedding = model.encode(stop_name).tolist()
            collection.add(
                ids=[f"stop_{idx}"],
                embeddings=[embedding],
                documents=[stop_name],
                metadatas=[{"type": "bus_stop", "name": stop_name}]
            )
        print(f"✅ {len(all_stops)} stops from JSON indexed successfully!")
    except Exception as e:
        print(f"❌ Error indexing JSON: {e}")

# index_data.py ထဲ (index_general_knowledge အောက်မှာ) ထည့်ပါ
def index_stop_tsv(tsv_path: str):
    with open(tsv_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # stops.tsv format ကို သင့်ဖိုင်အတိုင်း split လုပ်ပါ (ဥပမာ: mm \t en)
    for i, row in enumerate(lines):
        parts = row.split("\t")
        if len(parts) < 2:
            continue
        roman_name = parts[0].strip()
        mm_name = parts[1].strip()  # Burmese canonical
        en_name = parts[0].strip()

        # ✅ ၂ခုလုံးကို "documents" အဖြစ်ထည့် (embedding က bilingual model နဲ့ ok)
        collection.add(
            ids=[f"alias_mm_{i}"],
            embeddings=[model.encode(mm_name).tolist()],
            documents=[mm_name],
            metadatas=[{"type": "stop_alias", "mm": mm_name, "en": roman_name}]
        )
        collection.add(
            ids=[f"alias_en_{i}"],
            embeddings=[model.encode(en_name).tolist()],
            documents=[en_name],
            metadatas=[{"type": "stop_alias", "mm": mm_name, "en": roman_name}]
        )

# ၃။ General Knowledge များကို Index လုပ်ခြင်း (TXT မှ)
def index_general_knowledge(file_path):  # ဒီနေရာမှာ variable name ပဲ ထားရပါမယ်
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # စာသားများကို စာပိုဒ်အလိုက် (Paragraph) ခွဲထုတ်ခြင်း
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        for i, para in enumerate(paragraphs):
            embedding = model.encode(para).tolist()
            collection.add(
                ids=[f"kb_{i}"],
                embeddings=[embedding],
                documents=[para],
                metadatas=[{"type": "knowledge", "name": "general_info"}]
            )
        print(f"✅ Indexed {len(paragraphs)} paragraphs from {file_path}")
    except Exception as e:
        print(f"❌ Error indexing text file: {e}")


# --- စတင် Run ခြင်း ---
if __name__ == "__main__":
    # မှတ်တိုင်ဒေတာများထည့်ရန်
    index_bus_stops("data/YBS_data.json")
    # index_data.py __main__ ထဲ ထည့်
    index_stop_tsv("data/stops.tsv")

    # ဗဟုသုတဒေတာများထည့်ရန်
    # ဖိုင်လမ်းကြောင်း မှန်ကန်ပါစေ (ဥပမာ- data folder ထဲမှာရှိရင် "data/New Text Document.txt")
    index_general_knowledge("data/New Text Document.txt")