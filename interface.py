import streamlit as st
import time
from final_chatbot import (
    generate_smart_reply, find_simplest_path,
    get_best_stop_matches, extract_entities_with_gemini,
    check_greeting, get_general_knowledge,
    classify_intent_with_gemini,
    translate_to_myanmar,
    translate_to_english, generate_smart_reply_en
)

TIMEOUT_SECONDS = 120

st.set_page_config(page_title="YBS Route Finding Assistant", page_icon="🚌", layout="centered")
st.title("🚌 YBS Route Finding Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "waiting_for" not in st.session_state:
    st.session_state.waiting_for = None
if "pending_info" not in st.session_state:
    st.session_state.pending_info = {}
if "last_active_time" not in st.session_state:
    st.session_state.last_active_time = time.time()

def bot_response(text, elapsed=None):
    with st.chat_message("assistant"):
        st.markdown(text)
        if elapsed is not None:
            st.caption(f"⏱ Response time: {elapsed:.2f} seconds")

    # ✅ elapsed ကို message ထဲမှာ သိမ်း
    msg = {"role": "assistant", "content": text}
    if elapsed is not None:
        msg["elapsed"] = float(elapsed)
    st.session_state.messages.append(msg)


def detect_lang(text: str) -> str:
    for ch in text:
        if "\u1000" <= ch <= "\u109F":
            return "mm"
    return "en"

# --- Timeout Logic ---
current_time = time.time()
if st.session_state.waiting_for:
    if (current_time - st.session_state.last_active_time) > TIMEOUT_SECONDS:
        st.session_state.waiting_for = None
        st.session_state.pending_info = {}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # ✅ bot reply (assistant) အောက်မှာပဲ time ပြ
        if msg["role"] == "assistant" and "elapsed" in msg:
            st.caption(f"⏱ Response time: {msg['elapsed']:.2f} seconds")


if prompt := st.chat_input("မင်္ဂလာပါ၊ ဘာကူညီပေးရမလဲရှင်?"):
    user_input = prompt.strip()
    st.session_state.last_active_time = time.time()
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    start_t = time.perf_counter()

    # ✅ Spinner ကို processing block အကုန်လုံးကို wrap လုပ်
    with st.spinner("Thinking..."):
        # ၁) Re-ask flow (ရွေးခိုင်းထားတာရှိရင်)
        if st.session_state.waiting_for:
            info = st.session_state.pending_info
            selected = None

            for opt in info["options"]:
                if opt.get("label") in user_input or opt.get("name") in user_input:
                    selected = opt
                    break

            if selected:
                chosen_mm = selected["name"]
                f_stop = chosen_mm if st.session_state.waiting_for == "from" else info["f_fixed"]
                t_stop = info["t_fixed"] if st.session_state.waiting_for == "from" else chosen_mm

                path = find_simplest_path(f_stop, t_stop)
                user_lang = detect_lang(user_input)

                elapsed = time.perf_counter() - start_t
                if user_lang == "en":
                    bot_response(generate_smart_reply_en(path), elapsed=elapsed)
                else:
                    bot_response(generate_smart_reply(user_input, path), elapsed=elapsed)

                st.session_state.waiting_for = None
                st.session_state.pending_info = {}
            else:
                elapsed = time.perf_counter() - start_t
                bot_response("ကျေးဇူးပြု၍ စာရင်းထဲမှ မှတ်တိုင်အမည်ကို ရွေးချယ်ပေးပါရှင်။", elapsed=elapsed)

        # ၂) Normal flow
        else:
            intent = classify_intent_with_gemini(user_input)

            # (A) Greeting
            if intent == "greeting":
                reply = check_greeting(user_input)
                elapsed = time.perf_counter() - start_t
                bot_response(reply, elapsed=elapsed)

            # (B) Knowledge
            elif intent == "knowledge":
                user_lang = detect_lang(user_input)
                if user_lang == "en":
                    mm_query = translate_to_myanmar(user_input)
                    knowledge = get_general_knowledge(mm_query)
                    reply = translate_to_english(knowledge) if knowledge else "Sorry, I don't know about that yet."
                else:
                    knowledge = get_general_knowledge(user_input)
                    reply = knowledge if knowledge else "တောင်းပန်ပါတယ်၊ အဲဒီအကြောင်းအရာကို ကျွန်မ မသိသေးပါဘူးရှင်။"

                elapsed = time.perf_counter() - start_t
                bot_response(reply, elapsed=elapsed)

            # (C) Single stop
            elif intent == "single_stop":
                entities = extract_entities_with_gemini(user_input)
                found = entities.get("from") or entities.get("to") or user_input
                reply = (
                    f"'{found}' မှတ်တိုင်ကို တွေ့ရှိပါတယ်ရှင်။ "
                    f"ခရီးစဉ်ရှာလိုပါက စမှတ်ရော၊ ဆုံးမှတ်ပါ အပြည့်အစုံ "
                    f"(ဥပမာ - {found} ကနေ ဆူးလေကို) ပြောပေးပါဦးရှင်။"
                )
                elapsed = time.perf_counter() - start_t
                bot_response(reply, elapsed=elapsed)

            # (D) Bus route
            elif intent == "bus_route":
                user_lang = detect_lang(user_input)
                entities = extract_entities_with_gemini(user_input)
                f_name, t_name = entities.get("from"), entities.get("to")

                if f_name and t_name:
                    starts = get_best_stop_matches(f_name)
                    ends = get_best_stop_matches(t_name)

                    if starts and ends:
                        is_f_exact = any(s["name"] == f_name for s in starts)
                        is_t_exact = any(e["name"] == t_name for e in ends)

                        if not is_f_exact:
                            st.session_state.waiting_for = "from"
                            st.session_state.pending_info = {
                                "options": starts,
                                "t_fixed": ends[0]["name"],
                                "lang": user_lang,
                            }

                            if user_lang == "en":
                                reply = (
                                    f"Please choose the correct start stop for '{f_name}':\n" +
                                    "\n".join([f"- {s.get('label', s['name'])}" for s in starts])
                                )
                            else:
                                reply = (
                                    f"စမှတ် '*{f_name}*' အတွက် မှတ်တိုင်ရွေးပေးပါဦးရှင် -\n" +
                                    "\n".join([f"- {s['name']}" for s in starts])
                                )

                            elapsed = time.perf_counter() - start_t
                            bot_response(reply, elapsed=elapsed)

                        elif not is_t_exact:
                            st.session_state.waiting_for = "to"
                            st.session_state.pending_info = {
                                "options": ends,
                                "f_fixed": starts[0]["name"],
                                "lang": user_lang,
                            }

                            if user_lang == "en":
                                reply = (
                                    f"Please choose the correct destination stop for '{t_name}':\n" +
                                    "\n".join([f"- {e.get('label', e['name'])}" for e in ends])
                                )
                            else:
                                reply = (
                                    f"သွားမည့်နေရာ '*{t_name}*' အတွက် မှတ်တိုင်ရွေးပေးပါဦးရှင် -\n" +
                                    "\n".join([f"- {e['name']}" for e in ends])
                                )

                            elapsed = time.perf_counter() - start_t
                            bot_response(reply, elapsed=elapsed)

                        else:
                            # exact from/to -> calculate route now
                            f_stop = starts[0]["name"]
                            t_stop = ends[0]["name"]
                            path = find_simplest_path(f_stop, t_stop)

                            elapsed = time.perf_counter() - start_t
                            if user_lang == "en":
                                bot_response(generate_smart_reply_en(path), elapsed=elapsed)
                            else:
                                bot_response(generate_smart_reply(user_input, path), elapsed=elapsed)

                    else:
                        elapsed = time.perf_counter() - start_t
                        bot_response("တောင်းပန်ပါတယ်၊ မှတ်တိုင်အမည်များကို ရှာမတွေ့ပါဘူးရှင်။", elapsed=elapsed)
                else:
                    elapsed = time.perf_counter() - start_t
                    bot_response("ဘယ်ကနေ ဘယ်ကိုသွားမှာလဲဆိုတာလေး အပြည့်အစုံပြောပေးပါဦးရှင်။", elapsed=elapsed)

            # (E) Fallback
            else:
                knowledge = get_general_knowledge(user_input)
                reply = knowledge if knowledge else ("How can I help you?" if detect_lang(user_input) == "en" else "ဘာကူညီပေးရမလဲရှင်?")
                elapsed = time.perf_counter() - start_t
                bot_response(reply, elapsed=elapsed)
