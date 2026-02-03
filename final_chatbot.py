# import ollama
# import json
# import os
# import chromadb
# from collections import deque
# from sentence_transformers import SentenceTransformer
#
# # =========================================================
# # ၁။ Setup & Load Data
# # =========================================================
# v_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# client = chromadb.PersistentClient(path="./ybs_vector_db")
# collection = client.get_collection(name="ybs_stops")
#
#
# def load_routing_data():
#     stop_to_routes, route_to_stops = {}, {}
#     ROUTE_DIR = "./data/routes"
#     if os.path.exists(ROUTE_DIR):
#         for file in os.listdir(ROUTE_DIR):
#             if file.endswith(".json"):
#                 with open(os.path.join(ROUTE_DIR, file), encoding="utf-8") as f:
#                     data = json.load(f)
#                     route_id = str(data.get("route_id"))
#                     stops = [str(s) for s in data.get("stops", [])]
#                     route_to_stops[route_id] = stops
#                     for s in stops:
#                         stop_to_routes.setdefault(s, []).append(route_id)
#     return stop_to_routes, route_to_stops
#
#
# S_TO_R, R_TO_S = load_routing_data()
# all_data = collection.get()
#
#
# # =========================================================
# # ၂။ Improved Stop Matching (With Semantic Fallback)
# # =========================================================
# def get_all_matching_stops(name):
#     if not name or name == "...": return []
#     name = name.strip()
#
#     # ပထမအဆင့် - Direct Keyword Match
#     matches = {}
#     for meta in all_data["metadatas"]:
#         mm = str(meta.get("name_mm", ""))
#         en = str(meta.get("name_en", "")).lower()
#         if name in mm or name.lower() in en:
#             sid = str(meta["id"])
#             matches[sid] = {"id": sid, "name": mm}
#
#     # ဒုတိယအဆင့် - ရှာမတွေ့ရင် Semantic Vector Search ကို သုံးမယ် (ဒါမှ စာလုံးပေါင်းမှားတာ ရှာတွေ့မှာပါ)
#     if not matches:
#         query_vector = v_model.encode(name).tolist()
#         results = collection.query(query_embeddings=[query_vector], n_results=3)
#         for i in range(len(results['ids'][0])):
#             sid = results['ids'][0][i]
#             mm_name = results['metadatas'][0][i]['name_mm']
#             matches[sid] = {"id": sid, "name": mm_name}
#
#     return list(matches.values())[:3]
#
#
# # =========================================================
# # ၃။ Logic For Extraction & Reply (Strict Temperature 0.1)
# # =========================================================
# def ask_ai_extract(user_input):
#     prompt = f"""Extract ONLY "from" and "to" bus stop names from the user input.
#     Input: "{user_input}"
#     Return JSON ONLY: {{"from": "name", "to": "name"}}"""
#     try:
#         res = ollama.chat(
#             model="qwen3:latest",
#             messages=[{"role": "user", "content": prompt}],
#             format="json",
#             options={"temperature": 0.1}
#         )
#         return json.loads(res["message"]["content"])
#     except:
#         return {"from": "...", "to": "..."}
#
#
# def generate_human_reply(user_query, found_routes):
#     # Language Rule ထည့်သွင်းထားသည်
#     system_instruction = """
#     မင်းက ရန်ကုန်မြို့ YBS Guide တစ်ယောက်ပါ။
#     ၁။ Found Routes ထဲမှာ အချက်အလက်ရှိရင် အဲဒါကိုပဲ သုံးပြီး ဖြေပါ။
#     ၂။ User က English လိုပဲ တိတိကျကျ မေးရင် English လို ဖြေပါ။ မြန်မာလို (သို့) Mixed မေးရင် မြန်မာလိုပဲ ရင်းရင်းနှီးနှီး ဖြေပါ။
#     ၃။ စကားလုံးတွေ ထပ်နေတာ၊ အဓိပ္ပာယ်မရှိတဲ့ စာသားတွေ (ဥပမာ- ကူညီရန် အပ်နှံပါတယ်) လုံးဝ မသုံးရပါ။
#     ၄။ လမ်းကြောင်းမရှိရင် 'စိတ်မရှိပါနဲ့ဗျ၊ အဲဒီကို တိုက်ရိုက်ကား မရှိပါဘူး' လို့ ယဉ်ကျေးစွာ ပြောပါ။
#     """
#
#     messages = [
#         {"role": "system", "content": system_instruction},
#         {"role": "user", "content": f"User: {user_query}\nData: {json.dumps(found_routes, ensure_ascii=False)}"}
#     ]
#
#     res = ollama.chat(model="qwen3:latest", messages=messages, options={"temperature": 0.1})
#     return res["message"]["content"]
#
#
# # =========================================================
# # ၄။ Pathfinding Logic (Unchanged)
# # =========================================================
# def find_simplest_path(start_id, end_id):
#     if start_id == end_id: return []
#     queue = deque([(start_id, [])])
#     visited_stops, visited_routes = {start_id}, set()
#     while queue:
#         current, path = queue.popleft()
#         for bus in S_TO_R.get(current, []):
#             if bus in visited_routes: continue
#             visited_routes.add(bus)
#             for next_stop in R_TO_S.get(bus, []):
#                 if next_stop in visited_stops: continue
#                 visited_stops.add(next_stop)
#                 new_path = path + [{"bus": bus, "from": current, "to": next_stop}]
#                 if next_stop == end_id: return new_path
#                 queue.append((next_stop, new_path))
#     return None
#
#
# # =========================================================
# # ၅။ Execution Loop
# # =========================================================
# def smart_ybs_helper():
#     print("\n🚌 မင်္ဂလာပါဗျ! YBS လမ်းကြောင်းတွေ မေးလို့ရပါပြီ")
#     while True:
#         query = input("\n💬 User: ").strip()
#         if not query: continue
#         if query.lower() in ["exit", "quit"]: break
#
#         entities = ask_ai_extract(query)
#         from_raw, to_raw = entities.get("from"), entities.get("to")
#         found_routes = []
#
#         if from_raw != "..." and to_raw != "...":
#             starts = get_all_matching_stops(from_raw)
#             ends = get_all_matching_stops(to_raw)
#
#             for s in starts:
#                 for e in ends:
#                     path = find_simplest_path(s["id"], e["id"])
#                     if path:
#                         buses = sorted(set(step["bus"] for step in path))
#                         found_routes.append({"from": s["name"], "to": e["name"], "buses": buses})
#
#         final_answer = generate_human_reply(query, found_routes)
#         print(f"\n🤖 Assistant: {final_answer}")
#
#
# if __name__ == "__main__":
#     smart_ybs_helper()


# import ollama
# import json
# import os
# import chromadb
# import difflib
# from collections import deque
# from sentence_transformers import SentenceTransformer
#
# # =========================================================
# # ၁။ Setup & Load Data (JSON Only for Routing)
# # =========================================================
# v_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# client = chromadb.PersistentClient(path="./ybs_vector_db")
# collection = client.get_collection(name="ybs_stops")
#
#
# def load_json_routing_data():
#     stop_to_routes = {}
#     route_to_stops = {}
#     json_path = "data/YBS_data.json"
#
#     if not os.path.exists(json_path):
#         print(f"❌ Error: {json_path} not found.")
#         return {}, {}, []
#
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#         for line_data in data["ybs_bus_lines"]:
#             route_id = f"YBS {line_data['line']}"
#             stops = [s.strip() for s in line_data["stops"]]
#             route_to_stops[route_id] = stops
#             for s in stops:
#                 stop_to_routes.setdefault(s, []).append(route_id)
#
#     print(f"✅ Data Loaded: {len(stop_to_routes)} stops, {len(route_to_stops)} routes.")
#     return stop_to_routes, route_to_stops, list(stop_to_routes.keys())
#
#
# S_TO_R, R_TO_S, MASTER_STOP_LIST = load_json_routing_data()
#
#
# # =========================================================
# # ၂။ Smart Stop Matching (Refined with Debug)
# # =========================================================
# def get_best_stop_matches(name):
#     if not name or name.lower() == "null" or len(name) < 2:
#         return []
#
#     print(f"🔍 [DEBUG] Matching Stop Name: '{name}'")
#
#     # Vector Search
#     query_vector = v_model.encode(name).tolist()
#     results = collection.query(query_embeddings=[query_vector], n_results=5)
#
#     matches = []
#     seen = set()
#
#     for i in range(len(results['metadatas'][0])):
#         meta = results['metadatas'][0][i]
#         vector_name = meta['name_mm']
#
#         # JSON ထဲက နာမည်အမှန်တွေနဲ့ Fuzzy Match လုပ်မယ်
#         best_match = difflib.get_close_matches(vector_name, MASTER_STOP_LIST, n=1, cutoff=0.5)
#
#         if best_match:
#             stop_name = best_match[0]
#             if stop_name not in seen:
#                 matches.append({"name": stop_name, "township": meta.get('township_mm', 'ရန်ကုန်')})
#                 seen.add(stop_name)
#
#     print(f"📍 [DEBUG] Found Matches: {[m['name'] for m in matches]}")
#     return matches
#
#
# # =========================================================
# # ၃။ Pathfinding (Name-based BFS)
# # =========================================================
# def find_simplest_path(start_name, end_name):
#     print(f"🛤️ [DEBUG] Finding path from '{start_name}' to '{end_name}'")
#     if start_name == end_name: return []
#
#     queue = deque([(start_name, [])])
#     visited = {start_name}
#
#     while queue:
#         current, path = queue.popleft()
#         for bus in S_TO_R.get(current, []):
#             for next_stop in R_TO_S.get(bus, []):
#                 if next_stop not in visited:
#                     visited.add(next_stop)
#                     new_path = path + [{"bus": bus, "from": current, "to": next_stop}]
#                     if next_stop == end_name:
#                         print(f"✅ [DEBUG] Path Found with {len(new_path)} steps.")
#                         return new_path
#                     if len(new_path) < 2:  # 1-transfer limit for speed
#                         queue.append((next_stop, new_path))
#
#     print("❌ [DEBUG] No path found in BFS.")
#     return None
#
#
# # =========================================================
# # ၄။ Final AI Response
# # =========================================================
# def generate_smart_reply(user_query, context):
#     system_instruction = """
#     You are a professional YBS Assistant.
#     - If context has 'found_routes', explain the bus numbers and stops clearly.
#     - If context has 'ambiguous_stops', ask the user to clarify which stop they mean.
#     - IMPORTANT: NEVER mention names that are NOT in the context data provided.
#     - If no data is found, politely say you don't have that route in your YBS records.
#     - Always reply in Burmese.
#     """
#
#     prompt_content = f"User Query: {user_query}\nContext: {json.dumps(context, ensure_ascii=False)}"
#
#     res = ollama.chat(
#         model="qwen3:latest",  # or your qwen3
#         messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": prompt_content}],
#         options={"temperature": 0.1}
#     )
#     return res["message"]["content"]
#
#
# # =========================================================
# # ၅။ Execution Loop with Improved Extraction
# # =========================================================
# def smart_ybs_helper():
#     print("\n🚌 YBS Helper Ready! (Type 'exit' to quit)")
#
#     while True:
#         query = input("\n💬 User: ").strip()
#         if query.lower() in ["exit", "quit"]: break
#
#         # Step 1: Strict Entity Extraction
#         extract_prompt = f"""
#         Extract bus stop names from the input string.
#         Rules:
#         - Use ONLY words from the input.
#         - DO NOT invent or assume stop names.
#         - 'from' is origin, 'to' is destination.
#         - 'from' is the starting stop, 'to' is the destination stop.
#         - Remove verbs like "သွားချင်တယ်", "သွားမယ်", "ပို့ပေးပါ".
#         - Remove particles like "ကနေ", "ကို", "သို့".
#         - If the user says "လှည်းကူးဈေးကနေဦးခွေးပုသွားချင်တယ်", then from="လှည်းကူးဈေး", to="ဦးခွေးပု".
#         Input: "{query}"
#         Return JSON: {{"from": "name", "to": "name"}}
#         """
#
#         try:
#             extract = ollama.chat(
#                 model="qwen3:latest",
#                 messages=[{"role": "user", "content": extract_prompt}],
#                 format="json",
#                 options={"temperature": 0}
#             )
#             entities = json.loads(extract["message"]["content"])
#             print(f"🤖 [DEBUG] Extraction: FROM={entities.get('from')}, TO={entities.get('to')}")
#         except Exception as e:
#             print(f"⚠️ [DEBUG] Extraction failed: {e}")
#             entities = {"from": None, "to": None}
#
#         # Step 2: Match Stops
#         starts = get_best_stop_matches(entities.get("from"))
#         ends = get_best_stop_matches(entities.get("to"))
#
#         context = {"found_routes": [], "ambiguous_stops": {}}
#
#         # Step 3: Business Logic (Refined)
#         if starts and ends:
#             # အကယ်၍ match တွေ အများကြီး ထွက်နေရင်တောင် ပထမဆုံးတစ်ခုကို ယူပြီး
#             # direct route ရှိ၊ မရှိ အရင် စမ်းစစ်မယ်
#             path = find_simplest_path(starts[0]["name"], ends[0]["name"])
#
#             if path:
#                 context["found_routes"] = path
#             else:
#                 # Direct မရှိမှသာ ရွေးချယ်စရာတွေ (Ambiguous) ပြမယ်
#                 if len(starts) > 1: context["ambiguous_stops"]["from"] = starts
#                 if len(ends) > 1: context["ambiguous_stops"]["to"] = ends
#         else:
#             print(f"⚠️ [DEBUG] No matches found for extraction. FROM: {len(starts)}, TO: {len(ends)}")
#         # Step 4: Final Reply
#         print(f"\n🤖 Assistant: {generate_smart_reply(query, context)}")
#
#
# if __name__ == "__main__":
#     smart_ybs_helper()


import ollama
import json
import os
import chromadb
import difflib
import time
from collections import deque
from sentence_transformers import SentenceTransformer

# =========================================================
# ၁။ Setup & Load Data
# =========================================================
v_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client = chromadb.PersistentClient(path="./ybs_vector_db")
collection = client.get_collection(name="ybs_stops")


def load_json_routing_data():
    stop_to_routes = {}
    route_to_stops = {}
    json_path = "data/YBS_data.json"

    if not os.path.exists(json_path):
        print(f"❌ Error: {json_path} not found.")
        return {}, {}, []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for line_data in data["ybs_bus_lines"]:
            route_id = f"YBS {line_data['line']}"
            stops = [s.strip() for s in line_data["stops"]]
            route_to_stops[route_id] = stops
            for s in stops:
                stop_to_routes.setdefault(s, []).append(route_id)

    print(f"✅ Data Loaded: {len(stop_to_routes)} stops, {len(route_to_stops)} routes.")
    return stop_to_routes, route_to_stops, list(stop_to_routes.keys())


S_TO_R, R_TO_S, MASTER_STOP_LIST = load_json_routing_data()


# =========================================================
# ၂။ Smart Matching Logic
# =========================================================
def get_best_stop_matches(name):
    if not name or name.lower() == "null" or len(name) < 2:
        return []

    # A. Exact Match
    if name in MASTER_STOP_LIST:
        print(f"🎯 [DEBUG] Exact Match Found: '{name}'")
        return [{"name": name}]

    # B. Fuzzy Match
    close_names = difflib.get_close_matches(name, MASTER_STOP_LIST, n=1, cutoff=0.8)
    if close_names:
        print(f"🔍 [DEBUG] Fuzzy Match Found: '{close_names[0]}'")
        return [{"name": close_names[0]}]

    # C. Vector Search
    print(f"🔍 [DEBUG] Vector Matching: '{name}'")
    query_vector = v_model.encode(name).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=3)

    matches = []
    seen = set()
    for i in range(len(results['metadatas'][0])):
        meta = results['metadatas'][0][i]
        vector_name = meta['name_mm']
        best_match = difflib.get_close_matches(vector_name, MASTER_STOP_LIST, n=1, cutoff=0.6)
        if best_match and best_match[0] not in seen:
            matches.append({"name": best_match[0]})
            seen.add(best_match[0])

    print(f"📍 [DEBUG] Best Matches: {[m['name'] for m in matches]}")
    return matches


# =========================================================
# ၃။ Pathfinding & Final Reply Functions
# =========================================================
# def find_simplest_path(start_name, end_name):
#     if start_name == end_name: return []
#     queue = deque([(start_name, [])])
#     visited = {start_name}
#
#     while queue:
#         current, path = queue.popleft()
#         for bus in S_TO_R.get(current, []):
#             for next_stop in R_TO_S.get(bus, []):
#                 if next_stop not in visited:
#                     visited.add(next_stop)
#                     new_path = path + [{"bus": bus, "from": current, "to": next_stop}]
#                     if next_stop == end_name: return new_path
#                     if len(new_path) < 2: queue.append((next_stop, new_path))
#     return None

def find_simplest_path(start_name, end_name):
    if start_name == end_name: return []

    # queue: [(current_stop, path_taken, total_stops_count)]
    queue = deque([(start_name, [], 0)])
    visited = {start_name: 0}  # stop_name: min_stops_to_reach

    best_path = None
    min_total_stops = float('inf')

    while queue:
        current, path, total_count = queue.popleft()

        if total_count >= min_total_stops:
            continue

        for bus in S_TO_R.get(current, []):
            stops_in_route = R_TO_S.get(bus, [])
            try:
                curr_idx = stops_in_route.index(current)

                for i, next_stop in enumerate(stops_in_route):
                    if next_stop == current: continue

                    travel_distance = abs(i - curr_idx)
                    new_total_count = total_count + travel_distance


                    if next_stop not in visited or new_total_count < visited[next_stop]:
                        visited[next_stop] = new_total_count
                        new_path = path + [{"bus": bus, "from": current, "to": next_stop, "stops": travel_distance}]

                        if next_stop == end_name:
                            if new_total_count < min_total_stops:
                                min_total_stops = new_total_count
                                best_path = new_path
                        elif len(path) < 1:
                            queue.append((next_stop, new_path, new_total_count))
            except ValueError:
                continue

    return best_path


def get_ybs_statistics():
    route_lengths = {route: len(stops) for route, stops in R_TO_S.items()}
    longest_bus = max(route_lengths, key=route_lengths.get)

    stats = {
        "total_stops": len(MASTER_STOP_LIST),
        "total_routes": len(R_TO_S),
        "longest_route": {"line": longest_bus, "count": route_lengths[longest_bus]},
        "most_connected_stop": max(S_TO_R, key=lambda x: len(S_TO_R[x]))
    }
    return stats


YBS_STATS = get_ybs_statistics()


# def generate_smart_reply(user_query, context):
#     system_instruction = """
#     ROLE: You are "YBS Chatbot", a polite, professional, and local Yangon Bus Expert.
#
#     TONE & STYLE:
#     - Use warm, natural Burmese like a helpful local person.
#     - ALWAYS use polite particles: "ပါရှင်/ပါခင်ဗျာ" and "မင်္ဂလာပါရှင်/ခင်ဗျာ".
#     - Avoid repeating the same sentence or advice twice.
#     - Avoid weird phrases like "စီစဉ်ပါ" or "ဝါဆို အပ်ပြီး". Use "စီးပေးပါ", "စောင့်စီးပါ", "ဆင်းပေးပါ".
#
#     RESPONSE GUIDELINES:
#     1. GREETING: Start with a polite greeting.
#     2. SUMMARY: Say "ဝါဆို ကနေ ကုန်သည်လမ်း ကို သွားဖို့ [YBS နံပါတ်] ကို စီးလို့ရပါတယ်".
#     3. DETAIL STEPS (Step-by-step):
#        - Step 1: Tell where to start and which bus to take.
#        - Step 2 (If transfer): Clearly tell where to get off and which bus to change to. Use the word "မှတ်တိုင်မှာ ကားပြောင်းစီးပါ".
#        - Step 3: Tell the final destination stop.
#     4. PRO-TIP: Give ONE simple advice (e.g., "ကားနံပါတ် သေချာကြည့်စီးပါ").
#     5. CLOSING: "ဘေးကင်းလုံခြုံစွာ သွားလာနိုင်ပါစေ".
#
#     CONSTRAINTS:
#     - ONLY Burmese language.
#     - DO NOT hallucinate. If the context says 2 stops, say "မှတ်တိုင် ၂ ခု စီးရပါမယ်".
#     - Keep it concise but helpful. No extra robotic words.
#     """
#     prompt_content = f"User Query: {user_query}\nContext: {json.dumps(context, ensure_ascii=False)}"


def generate_smart_reply(user_query, context):
    system_instruction = """
    ROLE: You are "YBS Guide AI", a polite and professional Yangon Bus Expert.

    TONE & STYLE:
    - Warm, helpful Burmese (local natural style).
    - Use polite endings: "ပါရှင်/ပါခင်ဗျာ", "မင်္ဂလာပါရှင်/ခင်ဗျာ".
    - DO NOT use weird phrases like "အရည်အချင်းရှိပါမည်", "ရပ်တွင်းများ", "စီစဉ်ပါ".
    - Use natural bus terms: "စီးပေးပါ", "ဆင်းပေးပါ", "မှတ်တိုင်", "ကားပြောင်းစီးပါ".

    FORMATTING RULES:
    1. Greeting: "မင်္ဂလာပါရှင်။ YBS ChatBot မှ ကူညီပေးပါရစေ။"
    2. Sectioning: Use clear numbers (၁၊ ၂၊ ၃) and bullet points.
    3. If DIRECT ROUTE: Tell which YBS numbers to take from start to finish.
    4. If TRANSFER REQUIRED:
       - Step 1: Tell where to start and which bus to take.
       - Step 2: Tell clearly at which stop to GET OFF and which bus to CHANGE to.
       - Step 3: Mention the destination.
    5. Closing: "ခရီးစဉ်တစ်လျှောက် ဘေးကင်းလုံခြုံစွာ သွားလာနိုင်ပါစေရှင်။"

    SPECIFIC INSTRUCTIONS:
    - If the user asks general questions (နေကောင်းလား), reply warmly.
    - If the user asks for stats (မှတ်တိုင်ဘယ်နှခုရှိလဲ), use 'stats' in context to answer clearly.
    - If info is not in context, say "တောင်းပန်ပါတယ်ရှင်။ ဒီလမ်းကြောင်းကိုတော့ ကျွန်မ ရှာမတွေ့သေးလို့ မှတ်တိုင်အမည်လေး ပြန်စစ်ပေးပါဦးနော်။"
    """

    prompt_content = f"User Query: {user_query}\nContext: {json.dumps(context, ensure_ascii=False)}"

    res = ollama.chat(
        model="qwen3:latest",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt_content}
        ],
        options={"temperature": 0.2}
    )
    return res["message"]["content"]
# =========================================================
# ၄။ Execution Loop (Two Calls Strategy)
# =========================================================
# def smart_ybs_helper():
#     print("\n🚌 YBS Helper Ready! (Two-Call Mode)")
#
#     while True:
#         query = input("\n💬 User: ").strip()
#         if not query: continue
#         if query.lower() in ["exit", "quit"]: break
#
#         # အရေးကြီးသည်: Loop အသစ်တိုင်းမှာ variable တွေကို clear လုပ်ပါ
#         from_name, to_name = None, None
#         starts, ends = [], []
#         context = {"found_routes": [], "ambiguous_stops": {}}
#
#         start_time = time.time()
#
#         # Call 1: Entity Extraction (Improved Prompt)
#         extract_prompt = f"""
#         Extract 'from' and 'to' bus stops from the text.
#         Text: "{query}"
#         Rules:
#         - If the user provides multiple sentences, look at the whole text.
#         - 'from' is where they are, 'to' is where they want to go.
#         - Return JSON ONLY: {{"from": "string or null", "to": "string or null"}}
#         """
#
#         try:
#             extract_res = ollama.chat(model="qwen3:latest", messages=[{"role": "user", "content": extract_prompt}],
#                                       format="json")
#             entities = json.loads(extract_res["message"]["content"])
#             from_name = entities.get("from")
#             to_name = entities.get("to")
#             print(f"🤖 [DEBUG] Extraction: FROM={from_name}, TO={to_name}")
#         except:
#             entities = {"from": None, "to": None}
#
#         # Step 2: Matching & Logic
#         starts = get_best_stop_matches(from_name)
#         ends = get_best_stop_matches(to_name)
#
#         context = {"found_routes": [], "ambiguous_stops": {}}
#
#         if starts and ends:
#             path = find_simplest_path(starts[0]["name"], ends[0]["name"])
#             if path:
#                 context["found_routes"] = path
#             else:
#                 if len(starts) > 1: context["ambiguous_stops"]["from"] = starts
#                 if len(ends) > 1: context["ambiguous_stops"]["to"] = ends
#
#
#         # Call 2: Final Response
#         response = generate_smart_reply(query, context)
#         print(f"\n🤖 Assistant: {response}")
#
#         print(f"⏱️ [Timer] Response time: {time.time() - start_time:.2f} seconds")


def smart_ybs_helper():
    print("\n🚌 YBS Helper Ready! (Full Analytics Mode)")

    while True:
        query = input("\n💬 User: ").strip()
        if not query: continue
        if query.lower() in ["exit", "quit"]: break

        start_time = time.time()


        context = {
            "stats": YBS_STATS,
            "found_routes": [],
            "specific_bus_route": None,
            "ambiguous_stops": {}
        }

        # ၁။ YBS နံပါတ်ရှာ
        import re
        bus_match = re.search(r"YBS\s*(\d+)", query, re.IGNORECASE)
        if bus_match:
            bus_no = f"YBS {bus_match.group(1)}"
            if bus_no in R_TO_S:
                context["specific_bus_route"] = {bus_no: R_TO_S[bus_no]}

        # ၂။ လမ်းကြောင်းရှာ
        if from_name and to_name and from_name != "null":
            starts = get_best_stop_matches(from_name)
            ends = get_best_stop_matches(to_name)
            if starts and ends:
                path = find_simplest_path(starts[0]["name"], ends[0]["name"])
                if path:
                    formatted_path = []
                    for step in path:
                        formatted_path.append({
                            "bus": step["bus"],
                            "get_on": step["from"],
                            "get_off": step["to"],
                            "total_stops": step.get("stops", 0)
                        })
                    context["found_routes"] = formatted_path


        # Call 1: Entity Extraction
        extract_prompt = f"""
                Extract 'from' and 'to' bus stops from the text.
                Text: "{query}"
                Rules:
                - 'from' is the starting point, 'to' is the destination.
                - Return JSON ONLY in this format: {{"from": "string or null", "to": "string or null"}}
                """

        try:

            extract_res = ollama.chat(
                model="qwen3:latest",
                messages=[{"role": "user", "content": extract_prompt}],
                format="json"
            )
            entities = json.loads(extract_res["message"]["content"])
            from_name = entities.get("from")
            to_name = entities.get("to")
            print(f"🤖 [DEBUG] Extraction: FROM={from_name}, TO={to_name}")
        except Exception as e:
            print(f"⚠️ [DEBUG] Extraction Error: {e}")
            from_name, to_name = None, None

        # ၃။ Specific Bus Line ပါသလား စစ်ဆေး
        import re
        bus_match = re.search(r"YBS\s*(\d+)", query, re.IGNORECASE)
        if bus_match:
            bus_no = f"YBS {bus_match.group(1)}"
            if bus_no in R_TO_S:
                context["specific_bus_route"] = {bus_no: R_TO_S[bus_no]}

        # ၄။ Route Finding Logic
        if from_name and to_name and from_name.lower() != "null" and to_name.lower() != "null":
            starts = get_best_stop_matches(from_name)
            ends = get_best_stop_matches(to_name)

            if starts and ends:
                path = find_simplest_path(starts[0]["name"], ends[0]["name"])
                if path:
                    context["found_routes"] = path
                else:
                    if len(starts) > 1: context["ambiguous_stops"]["from"] = starts
                    if len(ends) > 1: context["ambiguous_stops"]["to"] = ends

        # ၅။ Final Response Generation
        response = generate_smart_reply(query, context)

        print(f"\n🤖 Assistant: {response}")
        print(f"⏱️ [Timer] Response time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    smart_ybs_helper()