# import json
# import ollama
# import sys
# import threading
# import time
# from thefuzz import process
#
# MODEL_NAME = "qwen3:latest"
# DB_PATH = "../data/master_bus_db.json"
#
#
# def load_db():
#     with open(DB_PATH, 'r', encoding='utf-8') as f:
#         return json.load(f)
#
#
# DB = load_db()
#
#
# # Loading Spinner logic
# class Spinner:
#     def __init__(self, message="Thinking..."):
#         self.spinner = ['|', '/', '-', '\\']
#         self.idx = 0
#         self.busy = False
#         self.message = message
#
#     def spinning_cursor(self):
#         while self.busy:
#             sys.stdout.write(f'\r{self.message} {self.spinner[self.idx]}')
#             sys.stdout.flush()
#             self.idx = (self.idx + 1) % len(self.spinner)
#             time.sleep(0.1)
#         sys.stdout.write('\r' + ' ' * (len(self.message) + 10) + '\r')
#
#     def __enter__(self):
#         self.busy = True
#         threading.Thread(target=self.spinning_cursor).start()
#
#     def __exit__(self, exception_type, exception_value, traceback):
#         self.busy = False
#         time.sleep(0.2)
#
#
# def find_stops(name_input):
#     if not name_input: return []
#     stop_names = [info['name'] for info in DB['stops'].values()]
#     matches = process.extract(name_input, stop_names, limit=5)
#
#     unique_matches = []
#     seen = set()
#     for name, score in matches:
#         if score > 70:
#             for s_id, info in DB['stops'].items():
#                 if info['name'] == name:
#                     entry = {"id": s_id, "name": name, "township": info['township']}
#                     uid = f"{name}_{info['township']}"
#                     if uid not in seen:
#                         unique_matches.append(entry)
#                         seen.add(uid)
#     return unique_matches
#
#
# def find_routes(start_id, end_id):
#     s_routes = set(DB['stop_to_routes'].get(str(start_id), []))
#     e_routes = set(DB['stop_to_routes'].get(str(end_id), []))
#     results = []
#
#     # Direct
#     direct = s_routes.intersection(e_routes)
#     for r in direct:
#         results.append({"type": "direct", "bus": r})
#
#     # 1-Transfer
#     if len(results) < 4:
#         for r1 in s_routes:
#             for t_id in DB['route_to_stops'].get(r1, []):
#                 t_routes = set(DB['stop_to_routes'].get(str(t_id), []))
#                 common = t_routes.intersection(e_routes)
#                 if common:
#                     t_name = DB['stops'][str(t_id)]['name']
#                     results.append({"type": "transfer", "bus1": r1, "at": t_name, "bus2": list(common)[0]})
#                     if len(results) >= 5: break
#     return results[:5]
#
#
# def get_disambiguated_stop(stop_name, label="မှတ်တိုင်"):
#     candidates = find_stops(stop_name)
#     if not candidates:
#         return None
#     if len(candidates) == 1:
#         return candidates[0]['id']
#
#     print(f"\nAI: {label} '{stop_name}' အတွက် အောက်ပါမြို့နယ်များထဲမှ ရွေးချယ်ပေးပါ-")
#     for i, c in enumerate(candidates):
#         print(f"({i + 1}) {c['name']} ({c['township']})")
#
#     while True:
#         try:
#             choice = int(input("နံပါတ်ရွေးပါ (Choose #): ")) - 1
#             if 0 <= choice < len(candidates):
#                 return candidates[choice]['id']
#         except ValueError:
#             pass
#         print("မှန်ကန်သော နံပါတ်ကို ရွေးချယ်ပေးပါခင်ဗျာ။")
#
#
# def run_bot():
#     print("🤖 YBS AI Assistant: မင်္ဂလာပါခင်ဗျာ။ ဘယ်ကို သွားချင်ပါသလဲ?")
#
#     while True:
#         user_input = input("\nYou: ")
#         if user_input.lower() in ['exit', 'quit', 'bye']: break
#
#         with Spinner("AI စဉ်းစားနေပါသည်..."):
#             system_prompt = """
#             You are a polite YBS Expert.
#             Extract 'from' and 'to' from text.
#             Respond ONLY in JSON: {"from": "stop_name", "to": "stop_name"}
#             If greeting, respond with text.
#             """
#             response = ollama.chat(model=MODEL_NAME, messages=[
#                 {'role': 'system', 'content': system_prompt},
#                 {'role': 'user', 'content': user_input}
#             ])
#             ai_raw = response['message']['content']
#
#         try:
#             entities = json.loads(ai_raw)
#             start_id = get_disambiguated_stop(entities.get('from'), "စထွက်မည့်မှတ်တိုင်")
#             end_id = get_disambiguated_stop(entities.get('to'), "သွားမည့်မှတ်တိုင်")
#
#             if start_id and end_id:
#                 routes = find_routes(start_id, end_id)
#                 if routes:
#                     print(f"\nAI: လူကြီးမင်းအတွက် အဆင်ပြေမည့် လမ်းကြောင်းများမှာ-")
#                     for r in routes:
#                         if r['type'] == 'direct':
#                             print(f"✅ {r['bus']} ကို တိုက်ရိုက်စီးပါခင်ဗျာ။")
#                         else:
#                             print(f"🔄 {r['bus1']} စီးပြီး {r['at']} မှာ {r['bus2']} ကို ပြောင်းစီးပါ။")
#                 else:
#                     print("\nAI: စိတ်မကောင်းပါဘူး။ လမ်းကြောင်း ရှာမတွေ့ပါဘူးခင်ဗျာ။")
#             else:
#                 print("\nAI: မှတ်တိုင်အမည်များကို ရှာမတွေ့ပါဘူး။ ပြန်လည်စစ်ဆေးပေးပါ။")
#         except:
#             print(f"\nAI: {ai_raw}")
#
#
# if __name__ == "__main__":
#     run_bot()

import streamlit as st
import pandas as pd
import json
import os
import ollama
from collections import deque
from thefuzz import process


@st.cache_data
def load_data():
    stop_to_routes = {}
    route_to_stops = {}
    routes_path = './data/routes/'

    if os.path.exists(routes_path):
        for filename in os.listdir(routes_path):
            if filename.endswith('.json'):
                with open(os.path.join(routes_path, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    r_id = data.get('route_id')
                    stops = data.get('stops', [])
                    route_to_stops[r_id] = stops
                    for s_id in stops:
                        stop_to_routes.setdefault(str(s_id), []).append(r_id)

    # Unicode normalization helps with Myanmar font issues
    stops_df = pd.read_csv('./data/stops.tsv', sep='\t')
    id_to_name = {str(row['id']): f"{row['name_mm']} ({row['township_mm']})" for _, row in stops_df.iterrows()}
    name_to_id = {row['name_mm']: str(row['id']) for _, row in stops_df.iterrows()}

    return stop_to_routes, route_to_stops, name_to_id, id_to_name


def ai_extract_stops(user_input):
    # System prompt provides context to Qwen3
    prompt = f"Extract 'from' and 'to' bus stops from: '{user_input}'. Return ONLY JSON: {{\"from\": \"...\", \"to\": \"...\"}}"
    try:
        response = ollama.chat(model='qwen3:latest', messages=[{'role': 'user', 'content': prompt}])
        return json.loads(response['message']['content'])
    except:
        return None


def find_bus_route(start_id, end_id, s_to_r, r_to_s):
    queue = deque([(start_id, [])])
    visited = {start_id}
    while queue:
        curr_stop, path = queue.popleft()
        if curr_stop == end_id: return path
        for bus in s_to_r.get(curr_stop, []):
            for next_stop in r_to_s.get(bus, []):
                ns = str(next_stop)
                if ns not in visited:
                    visited.add(ns)
                    queue.append((ns, path + [{"bus": bus, "stop": ns}]))
    return None


# --- ၄။ Main UI Logic ---
def main():
    st.set_page_config(page_title="YBS ChatBot", page_icon="🚌")
    st.title("🚌 YBS Smart Route Finder")

    s_to_r, r_to_s, name_to_id, id_to_info = load_data()

    user_query = st.text_input("ဘယ်ကနေ ဘယ်ကို သွားမှာလဲ?", placeholder="ဥပမာ- အင်းစိန်ကနေ ဆူးလေ")

    if st.button("လမ်းကြောင်းရှာပါ"):
        if user_query:
            with st.status("AI အလုပ်လုပ်နေပါသည်...", expanded=True) as status:
                st.write("🧠 AI က မှတ်တိုင်အမည်များကို ခွဲခြားနေသည်...")
                extracted = ai_extract_stops(user_query)

                if not extracted:
                    status.update(label="AI Error!", state="error")
                    st.error("AI က မှတ်တိုင်ကို ခွဲခြားရတာ အခက်အခဲရှိနေပါတယ်။")
                    return

                st.write(f"🔍 ရှာဖွေတွေ့ရှိမှု: {extracted.get('from')} -> {extracted.get('to')}")
                st.write("📍 Database ထဲမှာ အနီးစပ်ဆုံး မှတ်တိုင်ကို ရှာနေသည်...")

                all_stops = list(name_to_id.keys())
                best_from = process.extractOne(extracted['from'], all_stops)
                best_to = process.extractOne(extracted['to'], all_stops)

                if best_from and best_from[1] > 60 and best_to and best_to[1] > 60:
                    sid = name_to_id[best_from[0]]
                    eid = name_to_id[best_to[0]]

                    st.write("🚌 အမြန်ဆုံး Bus လမ်းကြောင်းကို တွက်ချက်နေသည်...")
                    result_path = find_bus_route(sid, eid, s_to_r, r_to_s)

                    status.update(label="ရှာဖွေမှု ပြီးဆုံးပါပြီ!", state="complete", expanded=False)
                else:
                    status.update(label="ရှာမတွေ့ပါ!", state="error")
                    st.error("မှတ်တိုင်အမည်ကို Database ထဲမှာ ရှာမတွေ့ပါ။")
                    return

            if result_path:
                st.success(f"🚩 **{best_from[0]}** မှ **{best_to[0]}** သို့ လမ်းကြောင်း")
                current_bus = ""
                for step in result_path:
                    if step['bus'] != current_bus:
                        st.divider()
                        st.warning(f"🚌 **YBS {step['bus']}** ကို စီးပါ")
                        current_bus = step['bus']
                    st.write(f" ➔ {id_to_info.get(step['stop'])}")




if __name__ == "__main__":
    main()