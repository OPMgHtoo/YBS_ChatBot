import streamlit as st
import json
import time
import ollama

from final_chatbot import (
    generate_smart_reply,
    find_simplest_path,
    get_best_stop_matches, YBS_STATS
)


context = {
    "stats": YBS_STATS,
    "found_routes": [],
    "ambiguous_stops": {}
}
st.set_page_config(page_title="YBS Chatbot", page_icon="🚌")

st.title("🚌 Yangon Bus Chatbot")
st.caption("ရန်ကုန်မြို့တွင်း ခရီးသွားလာရေးအတွက် လမ်းညွှန်ပေးမည့် AI လက်ထောက်")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# if prompt := st.chat_input("ဘယ်ကနေ ဘယ်ကို သွားချင်ပါသလဲ?"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)
#
#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         message_placeholder.markdown("🔍 အချက်အလက်များ စစ်ဆေးနေပါတယ်...")
#
#         try:
#             # ၂။ Entity Extraction (Call 1)
#             extract_prompt = f"""
#             Extract 'from' and 'to' bus stops from the text.
#             Text: "{prompt}"
#             Return JSON ONLY: {{"from": "string or null", "to": "string or null"}}
#             """
#             extract_res = ollama.chat(
#                 model="qwen3:latest",
#                 messages=[{"role": "user", "content": extract_prompt}],
#                 format="json"
#             )
#             entities = json.loads(extract_res["message"]["content"])
#             from_name = entities.get("from")
#             to_name = entities.get("to")
#
#             # ၃။ Backend Logic (Matching & Routing) ကို ချိတ်ဆက်ခြင်း
#             context = {"found_routes": [], "ambiguous_stops": {}}
#
#             # မှတ်တိုင်အမည်များကို ရှာဖွေခြင်း
#             starts = get_best_stop_matches(from_name)
#             ends = get_best_stop_matches(to_name)
#
#             if starts and ends:
#                 # find_simplest_path function ကို ခေါ်သုံးခြင်း
#                 path = find_simplest_path(starts[0]["name"], ends[0]["name"])
#                 if path:
#                     context["found_routes"] = path
#                 else:
#                     if len(starts) > 1: context["ambiguous_stops"]["from"] = starts
#                     if len(ends) > 1: context["ambiguous_stops"]["to"] = ends
#
#             # ၄။ Final Smart Reply (Call 2)
#             full_response = generate_smart_reply(prompt, context)
#
#             message_placeholder.markdown(full_response)
#             st.session_state.messages.append({"role": "assistant", "content": full_response})
#
#         except Exception as e:
#             message_placeholder.markdown("စနစ်အတွင်း အမှားအယွင်းတစ်ခု ဖြစ်ပေါ်ခဲ့ပါတယ်ရှင်။")
#             st.error(f"Error Details: {e}")

if prompt := st.chat_input("ဘယ်ကနေ ဘယ်ကို သွားချင်ပါသလဲ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_text = st.empty()  # Timer

        start_time = time.time()  # Timer

        try:
            # ၁။ Entity Extraction (Call 1)
            extract_prompt = f"""
            Extract 'from' and 'to' bus stops from the text.
            Text: "{prompt}"
            Return JSON ONLY: {{"from": "string or null", "to": "string or null"}}
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
            except Exception as e:
                from_name, to_name = None, None
                print(f"Error extracting entities: {e}")

            # ၂။ Logic & Routing


            context = {"found_routes": [], "ambiguous_stops": {}}
            starts = get_best_stop_matches(from_name)
            ends = get_best_stop_matches(to_name)

            if starts and ends:
                path = find_simplest_path(starts[0]["name"], ends[0]["name"])
                if path:
                    context["found_routes"] = path
                else:
                    if len(starts) > 1: context["ambiguous_stops"]["from"] = starts
                    if len(ends) > 1: context["ambiguous_stops"]["to"] = ends

            # ၃။ Final Reply Generation (Call 2)
            full_response = generate_smart_reply(prompt, context)

            # ၄။ Timer
            end_time = time.time()
            total_time = end_time - start_time

            message_placeholder.markdown(full_response)
            status_text.markdown(f"⏱️ Response time: {total_time:.2f} seconds")  # အောက်ခြေမှာ အချိန်ပြမယ်

            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            status_text.empty()
            message_placeholder.markdown("စနစ်အတွင်း အမှားအယွင်းတစ်ခု ဖြစ်ပေါ်ခဲ့ပါတယ်ရှင်။")
            st.error(f"Error Details: {e}")