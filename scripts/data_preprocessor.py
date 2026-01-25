import pandas as pd
import json
import os


def build_simple_map():
    # 1. Load stops data
    # stops.tsv ထဲက နာမည်တွေကို Township နဲ့တွဲပြီး သိမ်းထားမယ် (နာမည်တူရင် ခွဲခြားဖို့)
    stops_df = pd.read_csv('../data/stops.tsv', sep='\t')
    stop_info = {}
    for _, row in stops_df.iterrows():
        stop_info[str(row['id'])] = {
            "name": row['name_mm'],
            "township": row['township_mm']
        }

    # 2. Bus route database တည်ဆောက်မယ်
    stop_to_routes = {}
    route_to_stops = {}

    routes_path = '../data/routes'  # သင့်ရဲ့ route json တွေရှိတဲ့နေရာ

    print("Reading routes and indexing...")
    for filename in os.listdir(routes_path):
        if filename.endswith('.json'):
            with open(os.path.join(routes_path, filename), 'r') as f:
                data = json.load(f)
                r_id = filename.replace('route', '').replace('.json', '')
                stops = data.get('stops', [])

                route_to_stops[r_id] = stops

                for s_id in stops:
                    s_id = str(s_id)
                    if s_id not in stop_to_routes:
                        stop_to_routes[s_id] = []
                    if r_id not in stop_to_routes[s_id]:
                        stop_to_routes[s_id].append(r_id)

    # 3. Final Database သိမ်းဆည်းခြင်း
    final_db = {
        "stop_info": stop_info,
        "stop_to_routes": stop_to_routes,
        "route_to_stops": route_to_stops
    }

    with open('../data/simple_route_db.json', 'w', encoding='utf-8') as f:
        json.dump(final_db, f, ensure_ascii=False, indent=4)

    print(f"Success! Database built with {len(stop_to_routes)} stops.")


if __name__ == "__main__":
    build_simple_map()