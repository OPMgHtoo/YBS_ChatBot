# import json
# import pandas as pd
# import os
# import re
#
#
# def build_master_db():
#     print("--- 🛠️ DATA Files connecting together ---")
#
#     # 1. Stops Data ကို ဖတ်ခြင်း
#     stops_df = pd.read_csv('../data/stops.tsv', sep='\t')
#     stop_map = {str(row['id']): {"name": row['name_mm'], "township": row['township_mm']} for _, row in
#                 stops_df.iterrows()}
#
#     # 2. busdata.txt ထဲက Route Descriptions တွေကို Extract လုပ်ခြင်း
#     # (AI က နာမည်အပြည့်အစုံ သိဖို့အတွက်ပါ)
#     route_meta = {}
#     with open('../data/busdata.txt', 'r', encoding='utf-8') as f:
#         content = f.read()
#         # id: "..." နဲ့ color: "..." တွေကို regex နဲ့ ရှာမယ်
#         matches = re.findall(r'id:\s*"(.*?)",\s*color:\s*"(.*?)"', content)
#         for r_id, color in matches:
#             route_meta[r_id] = {"desc": r_id, "color": color}
#
#     # 3. Route Index နဲ့ Stops တွေကို ချိတ်ဆက်ခြင်း
#     with open('../data/routes-index.json', 'r') as f:
#         routes_index = json.load(f)
#
#     stop_to_routes = {}
#     route_to_stops = {}
#
#     routes_path = '../data/routes'
#     for ybs_no, file_list in routes_index.items():
#         for filename in file_list:
#             path = os.path.join(routes_path, filename)
#             if os.path.exists(path):
#                 with open(path, 'r') as f:
#                     data = json.load(f)
#                     stops = [str(s) for s in data.get('stops', [])]
#                     route_id = filename.replace('.json', '')
#
#                     route_to_stops[route_id] = stops
#                     for s_id in stops:
#                         if s_id not in stop_to_routes:
#                             stop_to_routes[s_id] = []
#                         stop_to_routes[s_id].append(route_id)
#
#     # 4. Master JSON ထုတ်ယူခြင်း
#     master_db = {
#         "stops": stop_map,  # ID -> {Name, Township}
#         "stop_to_routes": stop_to_routes,  # Stop ID -> [Route IDs]
#         "route_to_stops": route_to_stops,  # Route ID -> [Stop IDs]
#         "route_meta": route_meta  # Route ID -> {Description, Color}
#     }
#
#     with open('../data/master_bus_db.json', 'w', encoding='utf-8') as f:
#         json.dump(master_db, f, ensure_ascii=False, indent=4)
#
#     print(f"✅ Success! BusStops {len(stop_map)} are already indexed")
#
#
# if __name__ == "__main__":
#     build_master_db()

import json
import os
import pandas as pd


def process_ybs_data():
    stop_to_routes = {}  # { "stop_id": ["1", "2", "3A"] }
    route_to_stops = {}  # { "1": [2011, 2013, 2342] }

    routes_dir = './data/routes/'

    for filename in os.listdir(routes_dir):
        if filename.endswith('.json'):
            with open(os.path.join(routes_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                r_id = data['route_id']
                stops = data['stops']

                route_to_stops[r_id] = stops
                for s_id in stops:
                    s_id_str = str(s_id)
                    if s_id_str not in stop_to_routes:
                        stop_to_routes[s_id_str] = []
                    stop_to_routes[s_id_str].append(r_id)

    return stop_to_routes, route_to_stops


def get_stop_info():
    df = pd.read_csv('./data/stops.tsv', sep='\t')
    stop_names = {str(row['id']): row['name_mm'] for _, row in df.iterrows()}
    return stop_names