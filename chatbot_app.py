import json
import random
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------- Caching ----------
@st.cache_data(show_spinner=False)
def load_knowledge_base(file_path="chatbot_qa.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chatbot_knowledge"] if "chatbot_knowledge" in data else data

@st.cache_resource(show_spinner=False)
def setup_sentence_transformer():
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def setup_dialo():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    return tokenizer, model

# ---------- Bot Response Logic ----------
def fallback_response(user_input):
    tokenizer, model = setup_dialo()
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    output = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return "🤖 Hmm... I 'm not too sure, but I try to say: " + output

def get_best_response(user_input, kb_data, model, threshold=0.6):
    all_patterns, response_map, tag_map = [], [], []
    for intent in kb_data:
        for pattern in intent["patterns"]:
            all_patterns.append(pattern)
            response_map.append(intent["responses"])
            tag_map.append(intent["tag"])
    embeddings = model.encode(all_patterns, convert_to_tensor=True)
    query_embedding = model.encode(user_input, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, embeddings)[0]
    best_score = torch.max(cosine_scores).item()
    best_idx = torch.argmax(cosine_scores).item()
    if best_score >= threshold:
        return random.choice(response_map[best_idx]), tag_map[best_idx]
    else:
        return fallback_response(user_input), None

# ---------- Chat Message Renderer ----------
def render_message(message, is_user):
    style = """
        background-color:{bg}; padding:10px 15px; border-radius:15px;
        max-width:80%; margin-bottom:10px; {align};
    """
    if is_user:
        st.markdown(
            f"""
            <div style="{style.format(bg="#331C1C", align='margin-left:auto')}">
                <p style="margin:0; font-weight:600;">👤 You</p>
                <p style="margin:0;">{message}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            f"""
            <div style="{style.format(bg="#515457", align='margin-right:auto')}">
                <p style="margin:0; font-weight:600;">🤖 Bot</p>
                <p style="margin:0;">{message}</p>
            </div>
            """, unsafe_allow_html=True)

# ---------- Main App ----------
def main():
    st.set_page_config(page_title="CS Chatbot Advisor KWASU", page_icon="🤖", layout="wide")

    # Setup columns for central layout
    left_col, center_col, right_col = st.columns([1, 2, 1])
    with center_col:

        st.title("🤖 Computer Science Chatbot Advisor (KWASU)")
        st.markdown("Ask me anything about computer science education, your courses, projects, or careers.")

        # Load models and data
        kb_data = load_knowledge_base()
        model = setup_sentence_transformer()

        # Session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "last_intent" not in st.session_state:
            st.session_state.last_intent = None

        # Suggested buttons
        # suggested_questions = [
        #     "How do I register my courses?",
        #     "What programming languages should I learn?",
        #     "Give me final year project ideas.",
        #     "How do I prepare for exams?",
        #     "What career options do I have after CS?",
        #     "Tell me about cybersecurity.",
        #     "What tools should I install?",
        #     "Who is the HOD of computer science?"
        # ]

        # Display chat history
        for chat in st.session_state.chat_history:
            render_message(chat["message"], is_user=chat["user"])

        # Suggested buttons in 2 rows
        # st.markdown("### 💡 Suggested Questions:")
        # rows = [suggested_questions[:4], suggested_questions[4:]]
        # for row in rows:
        #     cols = st.columns(len(row))
        #     for i, question in enumerate(row):
        #         if cols[i].button(question):
        #             bot_reply, intent_tag = get_best_response(question, kb_data, model)
        #             st.session_state.chat_history.append({"user": True, "message": question})
        #             st.session_state.chat_history.append({"user": False, "message": bot_reply})
        #             st.session_state.last_intent = intent_tag
        #             st.rerun()

        # Text input submission
        def submit():
            user_text = st.session_state.user_input
            if user_text.strip():
                bot_reply, intent_tag = get_best_response(user_text, kb_data, model)
                st.session_state.chat_history.append({"user": True, "message": user_text})
                st.session_state.chat_history.append({"user": False, "message": bot_reply})
                st.session_state.last_intent = intent_tag
            st.session_state.user_input = ""

        st.text_input("Type your question and press Enter:", key="user_input", on_change=submit)

        # Optional: Clear chat
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()
