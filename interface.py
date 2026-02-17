import re
import time
from typing import Optional

import streamlit as st

from final_chatbot import (
    classify_intent_with_gemini,
    extract_entities_with_gemini,
    find_simplest_path,
    generate_random_greeting,
    generate_smart_reply,
    generate_smart_reply_en,
    get_best_stop_matches,
    get_general_knowledge,
    translate_to_english,
    translate_to_myanmar,
)
from utils import detect_lang


# --- Configuration ---
TIMEOUT_SECONDS = 120

st.set_page_config(page_title="YBS Route Finding Assistant", page_icon="🚌")
st.title("🚌 YBS Route Finding Assistant")


# --- Session state init ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "waiting_for" not in st.session_state:
    st.session_state.waiting_for = None
if "pending_info" not in st.session_state:
    st.session_state.pending_info = {}
if "last_active_time" not in st.session_state:
    st.session_state.last_active_time = time.time()


def bot_response(text: str, elapsed_s: Optional[float] = None):
    """Render assistant message + optional response time. Also persist into session."""
    with st.chat_message("assistant"):
        st.markdown(text)
        if elapsed_s is not None:
            st.caption(f"⏱️ Response time: {elapsed_s:.2f}s")

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "elapsed_s": elapsed_s}
    )


# --- Timeout Logic ---
current_time = time.time()
if st.session_state.waiting_for:
    if (current_time - st.session_state.last_active_time) > TIMEOUT_SECONDS:
        st.session_state.waiting_for = None
        st.session_state.pending_info = {}


# --- Replay chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("elapsed_s") is not None:
            st.caption(f"⏱️ Response time: {msg['elapsed_s']:.2f}s")


# --- Main input ---
if prompt := st.chat_input("မင်္ဂလာပါ၊ ဘာကူညီပေးရမလဲရှင်?"):
    user_input = prompt.strip()
    st.session_state.last_active_time = time.time()

    # Store and render user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Spinner should keep spinning until we produce the assistant reply
    spinner_text = (
        "I am thinking..." if detect_lang(user_input) == "en" else "စဉ်းစားနေပါသည်..."
    )
    start_t = time.perf_counter()

    with st.spinner(spinner_text):
        # 1) Re-ask selection flow (highest priority)
        if st.session_state.waiting_for:
            info = st.session_state.pending_info
            selected = None

            for opt in info.get("options", []):  # opt: {name,label}
                label = opt.get("label") or ""
                name = opt.get("name") or ""
                if (label and label in user_input) or (name and name in user_input):
                    selected = opt
                    break

            if selected:
                chosen_mm = selected["name"]
                f_stop = chosen_mm if st.session_state.waiting_for == "from" else info["f_fixed"]
                t_stop = info["t_fixed"] if st.session_state.waiting_for == "from" else chosen_mm

                path = find_simplest_path(f_stop, t_stop)
                user_lang = detect_lang(user_input)

                reply_text = (
                    generate_smart_reply_en(path)
                    if user_lang == "en"
                    else generate_smart_reply(user_input, path)
                )

                st.session_state.waiting_for = None
                st.session_state.pending_info = {}

                elapsed = time.perf_counter() - start_t
                bot_response(reply_text, elapsed)

            else:
                reply_text = (
                    "Please select a stop name from the list."
                    if detect_lang(user_input) == "en"
                    else "ကျေးဇူးပြု၍ စာရင်းထဲမှ မှတ်တိုင်အမည်ကို ရွေးချယ်ပေးပါရှင်။"
                )
                elapsed = time.perf_counter() - start_t
                bot_response(reply_text, elapsed)

        # 2) Normal flow
        else:
            intent = classify_intent_with_gemini(user_input)

            # (A) Greeting
            if intent == "greeting":
                reply_text = generate_random_greeting(detect_lang(user_input))
                elapsed = time.perf_counter() - start_t
                bot_response(reply_text, elapsed)

            # (B) Knowledge
            elif intent == "knowledge":
                user_lang = detect_lang(user_input)
                if user_lang == "en":
                    mm_query = translate_to_myanmar(user_input)
                    knowledge = get_general_knowledge(mm_query)
                    reply_text = (
                        translate_to_english(knowledge)
                        if knowledge
                        else "Sorry, I don't know about that yet."
                    )
                else:
                    knowledge = get_general_knowledge(user_input)
                    reply_text = (
                        knowledge
                        if knowledge
                        else "တောင်းပန်ပါတယ်၊ အဲဒီအကြောင်းအရာကို ကျွန်မ မသိသေးပါဘူးရှင်။"
                    )

                elapsed = time.perf_counter() - start_t
                bot_response(reply_text, elapsed)

            # (C) Single stop
            elif intent == "single_stop":
                entities = extract_entities_with_gemini(user_input)
                found = entities.get("from") or entities.get("to") or user_input

                if detect_lang(user_input) == "en":
                    reply_text = (
                        f"I found the '{found}' stop. "
                        f"If you want to find a route, please provide the full starting and ending points "
                        f"(for example, I want to go from {found} to your destination)."
                    )
                else:
                    reply_text = (
                        f"'{found}' မှတ်တိုင်ကို တွေ့ရှိပါတယ်ရှင်။ "
                        "ခရီးစဉ်ရှာလိုပါက စမှတ်ရော၊ ဆုံးမှတ်ပါ အပြည့်အစုံ "
                        f"(ဥပမာ - {found} ကနေ ဆူးလေကို) ပြောပေးပါဦးရှင်။"
                    )

                elapsed = time.perf_counter() - start_t
                bot_response(reply_text, elapsed)

            # (D) Bus route
            elif intent == "bus_route":
                user_lang = detect_lang(user_input)

                def extract_from_to_en(text: str):
                    """Rule-based extractor for English/Myanglish."""
                    t = text.strip()
                    patterns = [
                        r"\bfrom\s+(?P<from>.+?)\s+\bto\s+(?P<to>.+?)(?:\?|\.|$)",
                        r"\bgo\s+from\s+(?P<from>.+?)\s+\bto\s+(?P<to>.+?)(?:\?|\.|$)",
                        r"\broute\s+from\s+(?P<from>.+?)\s+\bto\s+(?P<to>.+?)(?:\?|\.|$)",
                        r"^(?P<from>.+?)\s+\bto\s+(?P<to>.+?)(?:\?|\.|$)",
                    ]
                    for p in patterns:
                        m = re.search(p, t, flags=re.IGNORECASE)
                        if m:
                            f = m.group("from").strip(" -,:")
                            to = m.group("to").strip(" -,:")
                            return f, to
                    return None, None

                if user_lang == "en":
                    f_guess, t_guess = extract_from_to_en(user_input)
                    if f_guess and t_guess:
                        f_name, t_name = f_guess, t_guess
                    else:
                        entities = extract_entities_with_gemini(user_input)
                        f_name, t_name = entities.get("from"), entities.get("to")
                else:
                    entities = extract_entities_with_gemini(user_input)
                    f_name, t_name = entities.get("from"), entities.get("to")

                def clean_stop_phrase(s: str):
                    if not s:
                        return s
                    s = s.strip()
                    for bad in ["ကနေ", "မှ", "ကို", "သို့"]:
                        s = s.replace(bad, "")
                    return s.strip()

                def norm_mm_stop(s: str):
                    if not s:
                        return s
                    return " ".join(
                        str(s).strip().replace("စျေး", "ဈေး").replace("၀", "ဝ").split()
                    )

                if f_name and t_name:
                    f_name = norm_mm_stop(clean_stop_phrase(f_name))
                    t_name = norm_mm_stop(clean_stop_phrase(t_name))

                    starts = get_best_stop_matches(f_name)
                    ends = get_best_stop_matches(t_name)

                    if starts and ends:
                        is_f_exact = any(s.get("name") == f_name for s in starts)
                        is_t_exact = any(e.get("name") == t_name for e in ends)

                        if not is_f_exact:
                            st.session_state.waiting_for = "from"
                            st.session_state.pending_info = {
                                "options": starts,
                                "t_fixed": ends[0]["name"],
                                "lang": user_lang,
                            }
                            if user_lang == "en":
                                reply_text = (
                                    f"Please choose the correct start stop for '{f_name}':\n"
                                    + "\n".join(
                                        [f"- {s.get('label', s['name'])}" for s in starts]
                                    )
                                )
                            else:
                                reply_text = (
                                    f"စမှတ် '*{f_name}*' အတွက် မှတ်တိုင်ရွေးပေးပါဦးရှင် -\n"
                                    + "\n".join([f"- {s['name']}" for s in starts])
                                )

                            elapsed = time.perf_counter() - start_t
                            bot_response(reply_text, elapsed)

                        elif not is_t_exact:
                            st.session_state.waiting_for = "to"
                            st.session_state.pending_info = {
                                "options": ends,
                                "f_fixed": starts[0]["name"],
                                "lang": user_lang,
                            }
                            if user_lang == "en":
                                reply_text = (
                                    f"Please choose the correct destination stop for '{t_name}':\n"
                                    + "\n".join(
                                        [f"- {e.get('label', e['name'])}" for e in ends]
                                    )
                                )
                            else:
                                reply_text = (
                                    f"သွားမည့်နေရာ '*{t_name}*' အတွက် မှတ်တိုင်ရွေးပေးပါဦးရှင် -\n"
                                    + "\n".join([f"- {e['name']}" for e in ends])
                                )

                            elapsed = time.perf_counter() - start_t
                            bot_response(reply_text, elapsed)

                        else:
                            f_stop = starts[0]["name"]
                            t_stop = ends[0]["name"]
                            path = find_simplest_path(f_stop, t_stop)

                            reply_text = (
                                generate_smart_reply_en(path)
                                if user_lang == "en"
                                else generate_smart_reply(user_input, path)
                            )

                            elapsed = time.perf_counter() - start_t
                            bot_response(reply_text, elapsed)

                    else:
                        reply_text = (
                            "Sorry, I can't find bus stop names."
                            if user_lang == "en"
                            else "တောင်းပန်ပါတယ်၊ မှတ်တိုင်အမည်များကို ရှာမတွေ့ပါဘူးရှင်။"
                        )
                        elapsed = time.perf_counter() - start_t
                        bot_response(reply_text, elapsed)

                else:
                    reply_text = (
                        "Please tell me clearly: where are you going from and to?"
                        if user_lang == "en"
                        else "ဘယ်ကနေ ဘယ်ကိုသွားမှာလဲဆိုတာလေး အပြည့်အစုံပြောပေးပါဦးရှင်။"
                    )
                    elapsed = time.perf_counter() - start_t
                    bot_response(reply_text, elapsed)

            # (E) Fallback
            else:
                knowledge = get_general_knowledge(user_input)
                if knowledge:
                    reply_text = knowledge
                else:
                    reply_text = (
                        "How can I help you?"
                        if detect_lang(user_input) == "en"
                        else "ဘာကူညီပေးရမလဲရှင်?"
                    )

                elapsed = time.perf_counter() - start_t
                bot_response(reply_text, elapsed)
