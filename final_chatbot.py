import json
import os
import difflib
from collections import deque
from openai import OpenAI
import re

from index_data import collection, model

# --- Local Ollama (OpenAI-compatible) ---
OLLAMA_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3:latest")

# Ollama မှာ API key မလိုပေမယ့် OpenAI SDK က field လိုတတ်လို့ dummy ထား
client_ai = OpenAI(base_url=OLLAMA_BASE_URL, api_key=os.getenv("OPENAI_API_KEY", "ollama"))

def translate_to_english(text: str) -> str:
    prompt = (
         "Translate the following Burmese text into clear and natural English. "
         "Keep bus stop names, numbers, and special terms like YBS, YPS unchanged. "
         "Return ONLY the translation text. Do NOT add explanations, headings, or extra sentences.\n\n"
          f"{text}"
    )

    try:
        response = client_ai.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return text  # fallback


def translate_to_myanmar(text: str) -> str:
    prompt = (
        "Translate the following English text into natural Burmese language. "
        "Keep bus stop names, numbers, and special terms like YBS, YPS unchanged.\n\n"
        f"{text}"
    )

    try:
        response = client_ai.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return text  # fallback (don't break system)



def load_json_routing_data():
    stop_to_routes, route_to_stops = {}, {}
    json_path = "data/YBS_data.json"

    if not os.path.exists(json_path):
        return {}, {}, []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for line_data in data["ybs_bus_lines"]:
            route_id = f"YBS {line_data['line']}"

            stops = [" ".join(str(s).strip().replace("စျေး", "ဈေး").split()) for s in line_data["stops"]]
            route_to_stops[route_id] = stops
            for s in stops:
                if route_id not in stop_to_routes.setdefault(s, []):
                    stop_to_routes[s].append(route_id)
    return stop_to_routes, route_to_stops, list(stop_to_routes.keys())


S_TO_R, R_TO_S, MASTER_STOP_LIST = load_json_routing_data()

# (final_chatbot.py)  <-- S_TO_R, R_TO_S, MASTER_STOP_LIST = ... အောက်တန်း (around line 69)

def _norm_en(s: str) -> str:
    # Han Tar Waddy  ==  HanTharWaddy  ဖြစ်အောင် normalize
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())

EN_ALIAS_MAP = {}  # norm_en -> list of {"mm":..., "en":...}

def _load_stop_tsv_map(tsv_path="data/stops.tsv"):
    if not os.path.exists(tsv_path):
        return
    with open(tsv_path, "r", encoding="utf-8") as f:
        for row in f:
            row = row.strip()
            if not row:
                continue
            parts = row.split("\t")
            if len(parts) < 2:
                continue
            en = parts[0].strip()
            mm = parts[1].strip()
            k = _norm_en(en)
            EN_ALIAS_MAP.setdefault(k, []).append({"mm": mm, "en": en})

_load_stop_tsv_map()


def extract_entities_with_gemini(query):
    prompt = (
        "Extract the start and destination bus stop phrases from the input.\n"
        "Return ONLY JSON: {\"from\": \"...\", \"to\": \"...\"}\n"
        "Do NOT add extra keys.\n"
    )

    try:
        completion = client_ai.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": query}],
            temperature=0
        )
        return json.loads(completion.choices[0].message.content)
    except:
        return {"from": None, "to": None}


def get_best_stop_matches(name):
    if not name:
        return []

    clean_name = " ".join(
        str(name)
        .replace("စျေး", "ဈေး")
        .replace("၀", "ဝ")
        .split()
    ).strip()

    # (final_chatbot.py) inside get_best_stop_matches() after clean_name computed (after line ~100)

    # English/Myanglish input detection (Myanmar unicode မပါရင် en)
    is_en_input = not any("\u1000" <= ch <= "\u109F" for ch in clean_name)

    if is_en_input:
        qk = _norm_en(clean_name)

        hits = []
        seen = set()
        # startswith priority, then contains
        for k, lst in EN_ALIAS_MAP.items():
            if k.startswith(qk) or (qk in k):
                for item in lst:
                    mm = item["mm"]
                    if mm in MASTER_STOP_LIST and mm not in seen:
                        seen.add(mm)
                        hits.append({"name": mm, "label": item["en"]})

        # ✅ HanTharWaddy() variants ကို အရင်ထုတ်ချင်လို့ sort
        def _rank(x):
            lk = _norm_en(x["label"])
            return (0 if lk.startswith(qk) else 1, len(lk))

        hits.sort(key=_rank)

        if hits:
            return hits[:10]

    # 1️⃣ Exact Burmese match (fast path)
    if clean_name in MASTER_STOP_LIST:
        return [{"name": clean_name}]

    # 2️⃣ Vector search in stop.tsv (stop_alias type)
    results = collection.query(
        query_embeddings=[model.encode(clean_name).tolist()],
        n_results=20,
        where={"type": "stop_alias"}
    )
    print("🔎 DEBUG - Searching stop alias for:", clean_name)
    print("🔎 DEBUG - Raw vector results:", results)

    # ✅ 2.5) Boost: MASTER_STOP_LIST ထဲက query ပါဝင်တဲ့ stop names ကို အရင်ထည့်
    seen = set()
    output = []

    q_norm = clean_name

    # startswith first, then contains
    prefix_hits = [s for s in MASTER_STOP_LIST if s.startswith(q_norm)]
    contain_hits = [s for s in MASTER_STOP_LIST if (q_norm in s and s not in prefix_hits)]

    strict_hits = prefix_hits + contain_hits

    if strict_hits:
        print("🔥 DEBUG - Strict substring hits:", strict_hits[:10])
        return [{"name": s, "label": s} for s in strict_hits[:10]]

    # limit to avoid huge list
    for s in (prefix_hits + contain_hits)[:10]:
        if s not in seen:
            seen.add(s)
            output.append({"name": s, "label": s})  # Burmese UI ဆို label=s OK (English UI ကို later map)

    metas = results.get("metadatas", [[]])[0]
    if not metas:
        return []

    # 3️⃣ Canonical Burmese name only (for routing)
    seen = set()
    output = []

    for m in metas:
        mm_name = m.get("mm")
        if mm_name and mm_name in MASTER_STOP_LIST and mm_name not in seen:
            seen.add(mm_name)
            output.append({"name": mm_name, "label": m.get("en", mm_name)})

    print("🔎 DEBUG - Canonical Burmese Matches:", output)

    # ✅ loop အပြင်မှာ rerank
    q_norm = clean_name

    def rank_key(item):
        name = item.get("name", "")
        label = item.get("label", "")
        # contains = (q_norm in name) or (q_norm in label)
        length_penalty = len(name)
        starts_with = name.startswith(q_norm) or label.startswith(q_norm)
        contains = (q_norm in name) or (q_norm in label)
        return (0 if starts_with else 1 if contains else 2, length_penalty)

    output = sorted(output, key=rank_key)
    return output


def mm_stop_to_en(mm_name: str) -> str:
    if not mm_name:
        return mm_name

    q = " ".join(str(mm_name).replace("စျေး", "ဈေး").replace("၀", "ဝ").split()).strip()

    results = collection.query(
        query_embeddings=[model.encode(q).tolist()],
        n_results=1,
        where={"type": "stop_alias"}
    )
    metas = results.get("metadatas", [[]])[0]

    print("🌍 DEBUG - Converting MM -> EN:", mm_name)
    print("🌍 DEBUG - EN match result:", metas)

    if metas and metas[0].get("en"):
        return metas[0]["en"]
    return mm_name  # fallback



def find_simplest_path(start_name, end_name):
    def normalize_stop(name):
        return " ".join(
            str(name).strip()
            .replace("စျေး", "ဈေး")
            .replace("၀", "ဝ")  # ✅ add this
            .split()
        )

    s_clean = normalize_stop(start_name)
    e_clean = normalize_stop(end_name)

    print("🛣 DEBUG - Normalized Start:", s_clean)
    print("🛣 DEBUG - Normalized End:", e_clean)

    if s_clean == e_clean: return None


    start_routes = S_TO_R.get(s_clean, [])
    end_routes = S_TO_R.get(e_clean, [])
    common_routes = set(start_routes) & set(end_routes)

    if common_routes:
        best_path = None
        min_stops = float('inf')
        print("🛣 DEBUG - Direct Common Routes:", common_routes)

        for r_id in common_routes:
            stops = R_TO_S[r_id]

            s_indices = [i for i, x in enumerate(stops) if x == s_clean]
            e_indices = [i for i, x in enumerate(stops) if x == e_clean]

            for s_idx in s_indices:
                for e_idx in e_indices:
                    if s_idx < e_idx:
                        seg = stops[s_idx: e_idx + 1]
                        if len(seg) < min_stops:
                            min_stops = len(seg)
                            best_path = [{
                                "from": s_clean, "to": e_clean,
                                "route": r_id, "segment": seg
                            }]

                            print("🛣 DEBUG - Final Path:", best_path)

        if best_path: return best_path


    queue = deque([(s_clean, [])])
    visited = {s_clean}

    while queue:
        curr_stop, path = queue.popleft()
        if len(path) >= 2: continue

        for r_id in S_TO_R.get(curr_stop, []):
            stops = R_TO_S[r_id]
            curr_indices = [i for i, x in enumerate(stops) if x == curr_stop]

            for c_idx in curr_indices:
                for next_stop_idx in range(c_idx + 1, len(stops)):
                    next_stop = stops[next_stop_idx]
                    if next_stop == e_clean:
                        return path + [{
                            "from": curr_stop, "to": e_clean,
                            "route": r_id, "segment": stops[c_idx: next_stop_idx + 1]
                        }]
                    if next_stop not in visited:
                        visited.add(next_stop)
                        queue.append((next_stop, path + [{
                            "from": curr_stop, "to": next_stop,
                            "route": r_id, "segment": stops[c_idx: next_stop_idx + 1]
                        }]))
                        print("🛣 DEBUG - Transfer Path Found:", path)

    return None


def generate_smart_reply(user_query, path_data):
    if not path_data:
        return "စိတ်မရှိပါနဲ့ရှင်၊ လမ်းကြောင်း ရှာမတွေ့ပါဘူး။"

    try:
        # စီးရမည့် ကားလိုင်းများ စာရင်း
        routes_list = list(dict.fromkeys([p['route'] for p in path_data]))
        routes_str = " နှင့် ".join(routes_list)

        start_stop = path_data[0]['from']
        end_stop = path_data[-1]['to']

        # 🔄 Narrative Info (ကားစီးရမည့် ပုံစံကို စာသားဖြင့် ဖော်ပြခြင်း)
        if len(path_data) == 1:
            # တိုက်ရိုက်သွားလို့ရတဲ့အခါ
            p = path_data[0]
            transfer_text = f"**{p['from']}** တွင် **{p['route']}** ကို တိုက်ရိုက်စီးပြီး **{p['to']}** သို့ သွားနိုင်ပါတယ်ရှင်။ ဘတ်စ်ကား ပြောင်းရန် မလိုပါ။"
        else:
            # ကားပြောင်းစီးရမည့်အခါ
            p1 = path_data[0]
            p2 = path_data[1]
            transfer_text = (
                f"**{p1['from']}** တွင် **{p1['route']}** စီးပြီး "
                f"**{p1['to']}** မှတ်တိုင်တွင် **{p2['route']}** ကားတစ်ဆင့်ပြောင်းစီးကာ "
                f"**{p2['from']}** မှ **{p2['to']}** သို့ သွားရပါမည်။"
            )

        # စုစုပေါင်း ဖြတ်သန်းရမည့် မှတ်တိုင်အရေအတွက် တွက်ချက်ခြင်း
        # total_stops = 0
        # for p in path_data:
        #     total_stops += len(p['segment'])
        # # ပြန်ထပ်နေတဲ့ Transfer point ကို နှုတ်ခြင်း
        # if len(path_data) > 1:
        #     total_stops -= 1

        return (
            f"မင်္ဂလာပါရှင်။ **{start_stop}** မှ **{end_stop}** သို့ သွားရန် လမ်းကြောင်း တွေ့ရှိပါတယ်ရှင်။\n\n"
            f"🚌 **စီးနင်းရမည့်ကား:** {routes_str}  "
            f"📍 **စီးရမည့်မှတ်တိုင်:** {start_stop}  "
            f"🏁 **ဆင်းရမည့်မှတ်တိုင်:** {end_stop}\n\n"
            f"🔄 **အချက်အလက်:** {transfer_text}\n\n"
            # f"စုစုပေါင်းမှတ်တိုင် **({total_stops})** ခု ဖြတ်သန်းသွားပါမည်။ ဘေးကင်းစွာ သွားလာနိုင်ပါစေရှင်။"
        )
    except Exception as e:
        return f"လမ်းကြောင်း အချက်အလက် ထုတ်ပြန်ရာတွင် အမှားအယွင်း ရှိနေပါသည်ရှင်။"

def generate_smart_reply_en(path_data):
    if not path_data:
        return "Sorry, I couldn't find a route."

    routes_list = list(dict.fromkeys([p['route'] for p in path_data]))
    routes_str = " and ".join(routes_list)

    start_stop = mm_stop_to_en(path_data[0]['from'])
    end_stop = mm_stop_to_en(path_data[-1]['to'])



    if len(path_data) == 1:
        p = path_data[0]
        transfer_text = (
            f"Take **{p['route']}** directly from **{mm_stop_to_en(p['from'])}** "
            f"to **{mm_stop_to_en(p['to'])}**. No transfer is needed."
        )
    else:
        p1, p2 = path_data[0], path_data[1]
        transfer_text = (
            f"From **{mm_stop_to_en(p1['from'])}**, take **{p1['route']}** and get off at "
            f"**{mm_stop_to_en(p1['to'])}**. Then transfer to **{p2['route']}** and continue to "
            f"**{mm_stop_to_en(p2['to'])}**."
        )

    return (
        f"Route found from **{start_stop}** to **{end_stop}**.\n\n"
        f"🚌 **Buses:** {routes_str}  "
        f"📍 **Board at:** {start_stop}  "
        f"🏁 **Get off at:** {end_stop}\n\n"
        f"🔄 **Info:** {transfer_text}"

    )



def check_greeting(text):
    text = text.lower().strip()

    # English Greetings
    en_greetings = ["hi", "hello", "hey", "good morning", "mingalabar","good night","good evening","good afternoon","sppl","chit lar"]
    # Myanmar Greetings
    mm_greetings = ["မင်္ဂလာပါ", "ဟိုင်း", "နေကောင်းလား", "ဟယ်လို"]

    if any(greet in text for greet in en_greetings):
        return "Hello! How can I help you today? Where would you like to go?"

    if any(greet in text for greet in mm_greetings):
        return "မင်္ဂလာပါရှင်။ ဒီနေ့ ဘယ်ကိုသွားဖို့ ကူညီပေးရမလဲဟင်?"

    return None


def get_general_knowledge(query):
    results = collection.query(
        query_embeddings=[model.encode(query).tolist()],
        n_results=5,  # ✅ 1 -> 5 (or 8)
        where={"type": "knowledge"}
    )

    cands = results["documents"][0] if results.get("documents") else []
    if not cands:
        return None

    # ✅ rerank using LLM (no keyword rules)
    prompt = (
        "You are answering a user's question using ONLY the given candidate passages.\n"
        "Pick the single best passage that directly answers the question.\n"
        "Return ONLY the chosen passage text, no explanations.\n\n"
        f"Question:\n{query}\n\n"
        "Candidates:\n" + "\n\n---\n\n".join(cands)
    )

    resp = client_ai.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()



def classify_intent_with_gemini(user_input):
    prompt = f"""
    You are an AI classifier for a Yangon Bus Chatbot. 
    Classify the following user input into one of these 4 categories:
    1. 'greeting' : If user says hi, hello, or asks how are you.
    2. 'bus_route' : If user asks how to go from one place to another (must have both source and destination).
    3. 'knowledge' : If user asks about YBS history, system rules, YPS card details, number of buses, or general facts.
    4. 'single_stop' : If user mentions only one bus stop name without asking for a route or knowledge.

    User input: "{user_input}"

    Return ONLY the category name in lowercase.
    """

    try:
        response = client_ai.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip().lower()
    except:
        return "knowledge"  # Error တက်ရင် default knowledge လို့ထားနိုင်ပါတယ်

# --- REPLACE THIS FUNCTION in final_chatbot.py ---


# def get_general_knowledge(query):
#     txt_path = "data/New Text Document.txt"  # index_data.py ထဲက path နဲ့တူအောင်ထားပါ  ✅
#
#     if not os.path.exists(txt_path):
#         return None
#
#     with open(txt_path, "r", encoding="utf-8") as f:
#         content = f.read()
#
#     # Paragraph-wise split (သင့် index_data.py နဲ့တူတဲ့ split pattern)
#     paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
#
#     q = query.replace(" ", "")
#     is_model_q = ("model" in query.lower()) or ("ဘတ်စ်ကား" in query) or ("မော်ဒယ်" in query)
#
#     if is_model_q:
#         model_terms = ["Yutong", "Ankai", "Foton", "Mini Bus", "မီနီဘတ်စ်", "တောင်ကိုရီးယား", "တစ်ပတ်ရစ်", "တရုတ်"]
#         for p in paragraphs:
#             if any(t in p for t in model_terms):
#                 return p
#
#     # (1) Count-type question (e.g., "စုစုပေါင်း ဘယ်နှစ်လိုင်း") → ၂၀၂၄/လိုင်း/ဒေတာ ပါတဲ့ paragraph ကို priority
#     # --- ADD inside get_general_knowledge() ---
#
#     q_lower = query.lower()
#
#     if "ybs card" in q_lower:
#         query = query + " YPS Card"
#         q_lower = query.lower()
#
#     is_total_lines_q_en = (
#             ("how many" in q_lower or "number of" in q_lower)
#             and ("line" in q_lower or "lines" in q_lower or "routes" in q_lower)
#             and ("ybs" in q_lower)
#     )
#
#     is_total_lines_q_mm = (
#             ("လိုင်း" in q or "ကားလိုင်း" in q or "ယာဉ်လိုင်း" in q)
#             and ("ဘယ်နှ" in q or "ဘယ္ႏွ" in q or "စုစုပေါင်း" in q)
#     )
#
#     if is_total_lines_q_en or is_total_lines_q_mm:
#         # 2024 + line paragraph priority
#         for p in paragraphs:
#             if ("၂၀၂၄" in p and "လိုင်း" in p and "ဒေတာ" in p):
#                 return p
#         for p in paragraphs:
#             if ("၂၀၂၄" in p and "လိုင်း" in p):
#                 return p
#         for p in paragraphs:
#             if ("၂၀၂၄" in p) and (("132" in p) or ("၁၃၂" in p)):
#                 return p
#
#     # (2) Normal knowledge fallback: keyword overlap နဲ့ best paragraph ယူ
#     keywords = [k for k in ["YBS", "YRTA", "YPS", "Card", "ကတ်", "ယာဉ်လိုင်း", "လိုင်း", "ဒေတာ", "၂၀၂၄",
#                             "how many", "number", "line", "lines", "route", "routes", "exist"] if k in query or k in q]
#     best_p, best_score = None, -1
#     for p in paragraphs:
#         score = sum(1 for k in keywords if k in p)
#         if score > best_score:
#             best_score = score
#             best_p = p
#
#     return best_p if best_score > 0 else None


