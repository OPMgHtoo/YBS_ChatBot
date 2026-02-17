import json
import os
import re
import random
from collections import deque

from openai import OpenAI

from index_data import collection, model
from utils import detect_lang


# ---------------------------
# Local LLM (Ollama / OpenAI-compatible)
# ---------------------------
OLLAMA_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3:latest")

# Ollama doesn't require an API key, but the OpenAI SDK expects one.
client_ai = OpenAI(base_url=OLLAMA_BASE_URL, api_key=os.getenv("OPENAI_API_KEY", "ollama"))


# ---------------------------
# Small helpers
# ---------------------------
def _safe_json_from_text(text: str):
    """Best-effort JSON object extraction.

    Local models sometimes return:
    - code fences ```json ... ```
    - extra text before/after the JSON
    """
    if not text:
        raise ValueError("Empty LLM output")

    # Strip code fences if present
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)

    # First try direct parse
    try:
        return json.loads(t)
    except Exception:
        pass

    # Try to extract the first JSON object { ... }
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        return json.loads(m.group(0))

    raise ValueError("Could not parse JSON")


# ---------------------------
# Greetings
# ---------------------------
def generate_random_greeting(lang: str = "en") -> str:
    en_replies = [
        "Hello there! Where would you like to go today?",
        "Hi! Tell me your starting point and destination.",
        "Hey! I’m ready to help you find a bus route.",
        "Good to see you! Where are you heading?",
        "Hi! Need help finding a bus?",
    ]

    mm_replies = [
        "မင်္ဂလာပါရှင်။ ဘယ်ကိုသွားချင်ပါသလဲ?",
        "ဟယ်လိုရှင်။ စမှတ်နဲ့ ဆုံးမှတ်ကို ပြောပေးပါဦး။",
        "မင်္ဂလာပါ။ ဘတ်စ်ကားလမ်းကြောင်း ရှာပေးရမလား?",
        "ဒီနေ့ ဘယ်ကိုသွားဖို့ စီစဉ်ထားပါသလဲရှင်?",
        "ကူညီပေးရမယ့် လမ်းကြောင်းရှိပါသလားရှင်?",
    ]

    return random.choice(en_replies if lang == "en" else mm_replies)


# ---------------------------
# Translation (local LLM)
# ---------------------------
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
            temperature=0,
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
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return text


# ---------------------------
# Routing data
# ---------------------------
def load_json_routing_data():
    stop_to_routes, route_to_stops = {}, {}
    json_path = "data/YBS_data.json"

    if not os.path.exists(json_path):
        return {}, {}, []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for line_data in data["ybs_bus_lines"]:
            route_id = f"YBS {line_data['line']}"
            stops = [
                " ".join(str(s).strip().replace("စျေး", "ဈေး").split())
                for s in line_data["stops"]
            ]
            route_to_stops[route_id] = stops
            for s in stops:
                if route_id not in stop_to_routes.setdefault(s, []):
                    stop_to_routes[s].append(route_id)
    return stop_to_routes, route_to_stops, list(stop_to_routes.keys())


S_TO_R, R_TO_S, MASTER_STOP_LIST = load_json_routing_data()


def _norm_en(s: str) -> str:
    # Han Tar Waddy  ==  HanTharWaddy  normalize
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


EN_ALIAS_MAP = {}  # norm_en -> list of {"mm":..., "en":...}


def _load_stop_tsv_map(tsv_path: str = "data/stops.tsv"):
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

            # Canonicalize Burmese so it matches MASTER_STOP_LIST
            mm = " ".join(
                str(mm).strip().replace("စျေး", "ဈေး").replace("၀", "ဝ").split()
            )

            k = _norm_en(en)
            EN_ALIAS_MAP.setdefault(k, []).append({"mm": mm, "en": en})


_load_stop_tsv_map()


# ---------------------------
# LLM-based entity extraction & intent
# ---------------------------
def extract_entities_with_gemini(query: str):
    """Extract {from,to}.

    Function name kept for compatibility with your interface code.
    """
    prompt = (
        "Extract the start and destination bus stop phrases from the input.\n"
        'Return ONLY JSON: {"from": "...", "to": "..."}\n'
        "Do NOT add extra keys.\n"
        "Do NOT wrap in markdown.\n"
    )

    try:
        completion = client_ai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            temperature=0,
        )
        return _safe_json_from_text(completion.choices[0].message.content)
    except Exception:
        return {"from": None, "to": None}


def classify_intent_with_gemini(user_input: str) -> str:
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
            temperature=0.1,
        )
        return response.choices[0].message.content.strip().lower()
    except Exception:
        return "knowledge"


def check_greeting(intent: str, user_input: str):
    if intent != "greeting":
        return None
    return generate_random_greeting(detect_lang(user_input))


# ---------------------------
# Stop matching
# ---------------------------
def get_best_stop_matches(name: str):
    if not name:
        return []

    clean_name = " ".join(
        str(name).replace("စျေး", "ဈေး").replace("၀", "ဝ").split()
    ).strip()

    is_en_input = not any("\u1000" <= ch <= "\u109F" for ch in clean_name)

    if is_en_input:
        qk = _norm_en(clean_name)

        hits = []
        seen = set()
        for k, lst in EN_ALIAS_MAP.items():
            if k.startswith(qk) or (qk in k):
                for item in lst:
                    mm = item["mm"]
                    if mm in MASTER_STOP_LIST and mm not in seen:
                        seen.add(mm)
                        hits.append({"name": mm, "label": item["en"]})

        def _rank(x):
            lk = _norm_en(x["label"])
            return (0 if lk.startswith(qk) else 1, len(lk))

        hits.sort(key=_rank)
        if hits:
            return hits[:10]

    # 1) Exact Burmese match
    if clean_name in MASTER_STOP_LIST:
        return [{"name": clean_name, "label": clean_name}]

    # 2) Vector search in stop.tsv (stop_alias type)
    results = collection.query(
        query_embeddings=[model.encode(clean_name).tolist()],
        n_results=20,
        where={"type": "stop_alias"},
    )

    # 2.5) Boost: substring match inside MASTER_STOP_LIST (fast + nice UX)
    q_norm = clean_name
    prefix_hits = [s for s in MASTER_STOP_LIST if s.startswith(q_norm)]
    contain_hits = [s for s in MASTER_STOP_LIST if (q_norm in s and s not in prefix_hits)]
    strict_hits = prefix_hits + contain_hits
    if strict_hits:
        return [{"name": s, "label": s} for s in strict_hits[:10]]

    metas = results.get("metadatas", [[]])[0]
    if not metas:
        return []

    seen = set()
    output = []
    for m in metas:
        mm_name = m.get("mm")
        if mm_name:
            mm_name = " ".join(
                str(mm_name).strip().replace("စျေး", "ဈေး").replace("၀", "ဝ").split()
            )
        if mm_name and mm_name in MASTER_STOP_LIST and mm_name not in seen:
            seen.add(mm_name)
            output.append({"name": mm_name, "label": m.get("en", mm_name)})

    def rank_key(item):
        nm = item.get("name", "")
        lb = item.get("label", "")
        starts_with = nm.startswith(q_norm) or lb.startswith(q_norm)
        contains = (q_norm in nm) or (q_norm in lb)
        return (0 if starts_with else 1 if contains else 2, len(nm))

    return sorted(output, key=rank_key)


def mm_stop_to_en(mm_name: str) -> str:
    if not mm_name:
        return mm_name

    q = " ".join(str(mm_name).replace("စျေး", "ဈေး").replace("၀", "ဝ").split()).strip()
    results = collection.query(
        query_embeddings=[model.encode(q).tolist()],
        n_results=1,
        where={"type": "stop_alias"},
    )
    metas = results.get("metadatas", [[]])[0]
    if metas and metas[0].get("en"):
        return metas[0]["en"]
    return mm_name


# ---------------------------
# Path finding
# ---------------------------
def find_simplest_path(start_name: str, end_name: str):
    def normalize_stop(name):
        return " ".join(
            str(name).strip().replace("စျေး", "ဈေး").replace("၀", "ဝ").split()
        )

    s_clean = normalize_stop(start_name)
    e_clean = normalize_stop(end_name)
    if s_clean == e_clean:
        return None

    start_routes = S_TO_R.get(s_clean, [])
    end_routes = S_TO_R.get(e_clean, [])
    common_routes = set(start_routes) & set(end_routes)

    # direct route
    if common_routes:
        best_path = None
        min_stops = float("inf")
        for r_id in common_routes:
            stops = R_TO_S[r_id]
            s_indices = [i for i, x in enumerate(stops) if x == s_clean]
            e_indices = [i for i, x in enumerate(stops) if x == e_clean]
            for s_idx in s_indices:
                for e_idx in e_indices:
                    if s_idx < e_idx:
                        seg = stops[s_idx : e_idx + 1]
                        if len(seg) < min_stops:
                            min_stops = len(seg)
                            best_path = [{"from": s_clean, "to": e_clean, "route": r_id, "segment": seg}]
        if best_path:
            return best_path

    # one transfer at most (path length <= 2)
    queue = deque([(s_clean, [])])
    visited = {s_clean}
    while queue:
        curr_stop, path = queue.popleft()
        if len(path) >= 2:
            continue

        for r_id in S_TO_R.get(curr_stop, []):
            stops = R_TO_S[r_id]
            curr_indices = [i for i, x in enumerate(stops) if x == curr_stop]
            for c_idx in curr_indices:
                for next_stop_idx in range(c_idx + 1, len(stops)):
                    next_stop = stops[next_stop_idx]
                    if next_stop == e_clean:
                        return path + [
                            {
                                "from": curr_stop,
                                "to": e_clean,
                                "route": r_id,
                                "segment": stops[c_idx : next_stop_idx + 1],
                            }
                        ]
                    if next_stop not in visited:
                        visited.add(next_stop)
                        queue.append(
                            (
                                next_stop,
                                path
                                + [
                                    {
                                        "from": curr_stop,
                                        "to": next_stop,
                                        "route": r_id,
                                        "segment": stops[c_idx : next_stop_idx + 1],
                                    }
                                ],
                            )
                        )
    return None


# ---------------------------
# Replies
# ---------------------------
def generate_smart_reply(user_query: str, path_data):
    if not path_data:
        return "စိတ်မရှိပါနဲ့ရှင်၊ လမ်းကြောင်း ရှာမတွေ့ပါဘူး။"

    try:
        routes_list = list(dict.fromkeys([p["route"] for p in path_data]))
        routes_str = " နှင့် ".join(routes_list)
        start_stop = path_data[0]["from"]
        end_stop = path_data[-1]["to"]

        if len(path_data) == 1:
            p = path_data[0]
            transfer_text = (
                f"**{p['from']}** တွင် **{p['route']}** ကို တိုက်ရိုက်စီးပြီး "
                f"**{p['to']}** သို့ သွားနိုင်ပါတယ်ရှင်။ ဘတ်စ်ကား ပြောင်းရန် မလိုပါ။"
            )
        else:
            p1 = path_data[0]
            p2 = path_data[1]
            transfer_text = (
                f"**{p1['from']}** တွင် **{p1['route']}** စီးပြီး "
                f"**{p1['to']}** မှတ်တိုင်တွင် **{p2['route']}** ကားတစ်ဆင့်ပြောင်းစီးကာ "
                f"**{p2['from']}** မှ **{p2['to']}** သို့ သွားရပါမည်။"
            )

        return (
            f"မင်္ဂလာပါရှင်။ **{start_stop}** မှ **{end_stop}** သို့ သွားရန် လမ်းကြောင်း တွေ့ရှိပါတယ်ရှင်။\n\n"
            f"🚌 **စီးနင်းရမည့်ကား:** {routes_str}  "
            f"📍 **စီးရမည့်မှတ်တိုင်:** {start_stop}  "
            f"🏁 **ဆင်းရမည့်မှတ်တိုင်:** {end_stop}\n\n"
            f"🔄 **အချက်အလက်:** {transfer_text}\n"
        )
    except Exception:
        return "လမ်းကြောင်း အချက်အလက် ထုတ်ပြန်ရာတွင် အမှားအယွင်း ရှိနေပါသည်ရှင်။"


def generate_smart_reply_en(path_data):
    if not path_data:
        return "Sorry, I couldn't find a route."

    routes_list = list(dict.fromkeys([p["route"] for p in path_data]))
    routes_str = " and ".join(routes_list)

    start_stop = mm_stop_to_en(path_data[0]["from"])
    end_stop = mm_stop_to_en(path_data[-1]["to"])

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


# ---------------------------
# Knowledge retrieval
# ---------------------------
def get_general_knowledge(query: str):
    results = collection.query(
        query_embeddings=[model.encode(query).tolist()],
        n_results=5,
        where={"type": "knowledge"},
    )
    cands = results["documents"][0] if results.get("documents") else []
    if not cands:
        return None

    prompt = (
        "You are answering a user's question using ONLY the given candidate passages.\n"
        "Pick the single best passage that directly answers the question.\n"
        "Return ONLY the chosen passage text, no explanations.\n\n"
        f"Question:\n{query}\n\n"
        "Candidates:\n" + "\n\n---\n\n".join(cands)
    )

    try:
        resp = client_ai.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # If reranking fails, fall back to the first candidate
        return cands[0].strip() if cands else None
