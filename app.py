import json
import random
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data(show_spinner=False)
def load_knowledge_base(file_path="chatbot_qa.json"):
    """Load and preprocess knowledge base with error handling"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both direct array and nested structure
        if isinstance(data, dict) and "chatbot_knowledge" in data:
            data = data["chatbot_knowledge"]
        
        logger.info(f"Loaded {len(data)} intents from knowledge base")
        return data
    except Exception as e:
        logger.error(f"Error loading knowledge base: {e}")
        st.error("Failed to load knowledge base. Please check the file.")
        return []

@st.cache_resource(show_spinner=False)
def setup_sentence_transformer():
    """Initialize sentence transformer with error handling"""
    try:
        model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        logger.info("Sentence transformer loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading sentence transformer: {e}")
        st.error("Failed to load AI model. Please check your internet connection.")
        return None

@st.cache_resource(show_spinner=False)
def setup_dialo():
    """Initialize DialoGPT with error handling and optimization"""
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info("DialoGPT loaded successfully")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading DialoGPT: {e}")
        return None, None

def fallback_response(user_input):
    """Generate fallback response with improved error handling"""
    try:
        tokenizer, model = setup_dialo()
        if tokenizer is None or model is None:
            return "🤖 I'm having trouble accessing my backup AI. Please try rephrasing your question or ask about CS topics I know well."
        
        # Limit input length to prevent memory issues
        max_input_length = 100
        if len(user_input) > max_input_length:
            user_input = user_input[:max_input_length]
        
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            chat_history_ids = model.generate(
                input_ids, 
                max_length=min(1000, input_ids.shape[-1] + 50),
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                no_repeat_ngram_size=2
            )
        
        output = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        if not output.strip():
            return "🤖 I'm not sure about that. Could you ask about CS courses, programming, or career advice?"
        
        return f"🤖 Let me try to help: {output.strip()}"
        
    except Exception as e:
        logger.error(f"Error in fallback response: {e}")
        return "🤖 I'm having technical difficulties. Please ask about CS topics like programming languages, courses, or career advice."

@st.cache_data(show_spinner=False)
def preprocess_knowledge_base(kb_data):
    """Preprocess knowledge base for faster similarity search"""
    all_patterns = []
    response_map = []
    tag_map = []
    
    for intent in kb_data:
        for pattern in intent["patterns"]:
            all_patterns.append(pattern.lower().strip())
            response_map.append(intent["responses"])
            tag_map.append(intent["tag"])
    
    return all_patterns, response_map, tag_map

def get_best_response(user_input, kb_data, model, threshold=0.65):
    """Optimized response matching with better preprocessing"""
    if model is None:
        return "🤖 AI model not available. Please refresh the page.", None
    
    try:
        # Use cached preprocessing
        all_patterns, response_map, tag_map = preprocess_knowledge_base(kb_data)
        
        if not all_patterns:
            return fallback_response(user_input), None
        
        # Normalize user input
        user_input_normalized = user_input.lower().strip()
        
        # Encode patterns and query
        embeddings = model.encode(all_patterns, convert_to_tensor=True, show_progress_bar=False)
        query_embedding = model.encode(user_input_normalized, convert_to_tensor=True, show_progress_bar=False)
        
        # Calculate similarity
        cosine_scores = util.cos_sim(query_embedding, embeddings)[0]
        best_score = torch.max(cosine_scores).item()
        best_idx = torch.argmax(cosine_scores).item()
        
        logger.info(f"Best match score: {best_score:.3f} for query: '{user_input[:50]}...'")
        
        if best_score >= threshold:
            selected_response = random.choice(response_map[best_idx])
            return selected_response, tag_map[best_idx]
        else:
            return fallback_response(user_input), None
            
    except Exception as e:
        logger.error(f"Error in get_best_response: {e}")
        return "🤖 I encountered an error processing your question. Please try again.", None

def render_message(message, is_user):
    """Optimized message rendering with better styling"""
    if is_user:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           padding: 12px 16px; border-radius: 18px 18px 4px 18px; 
                           max-width: 75%; color: white; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <div style="font-weight: 600; font-size: 0.85em; opacity: 0.9; margin-bottom: 4px;">👤 You</div>
                    <div style="line-height: 1.4;">{message}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                           padding: 12px 16px; border-radius: 18px 18px 18px 4px; 
                           max-width: 75%; color: white; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <div style="font-weight: 600; font-size: 0.85em; opacity: 0.9; margin-bottom: 4px;">🤖 CS Assistant</div>
                    <div style="line-height: 1.4;">{message}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="CS Chatbot Advisor KWASU", 
        page_icon="🤖", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        border: none;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Computer Science Chatbot Advisor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your AI companion for CS education at KWASU • Ask about courses, programming, careers & more</p>', unsafe_allow_html=True)

    # Load models and data with error handling
    kb_data = load_knowledge_base()
    if not kb_data:
        st.error("Unable to load knowledge base. Please refresh the page.")
        return
        
    model = setup_sentence_transformer()
    if model is None:
        st.error("Unable to load AI model. Please check your internet connection and refresh.")
        return

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        # Add welcome message
        st.session_state.chat_history.append({
            "user": False, 
            "message": "Hello! I'm your CS assistant. I can help with courses, programming languages, career advice, exam prep, and more. What would you like to know?"
        })
    
    if "last_intent" not in st.session_state:
        st.session_state.last_intent = None

    # Improved suggested questions
    suggested_questions = [
        "What programming languages should I learn first?",
        "How do I prepare effectively for CS exams?", 
        "Give me some final year project ideas",
        "What career paths are available in CS?",
        "How can I get a good internship?",
        "What development tools should I install?",
        "Tell me about cybersecurity careers",
        "How do I register for courses?"
    ]

    # Chat container
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for chat in st.session_state.chat_history:
            render_message(chat["message"], is_user=chat["user"])
        st.markdown('</div>', unsafe_allow_html=True)

    # Suggested questions in a more organized layout
    st.markdown("### 💡 Quick Start Questions")
    cols = st.columns(4)
    for idx, question in enumerate(suggested_questions):
        col_idx = idx % 4
        if cols[col_idx].button(question, key=f"btn_{idx}"):
            with st.spinner("Thinking..."):
                bot_reply, intent_tag = get_best_response(question, kb_data, model)
                st.session_state.chat_history.append({"user": True, "message": question})
                st.session_state.chat_history.append({"user": False, "message": bot_reply})
                st.session_state.last_intent = intent_tag
            st.rerun()

    # Input handling with better UX
    def submit():
        user_text = st.session_state.user_input.strip()
        if user_text:
            with st.spinner("Processing your question..."):
                bot_reply, intent_tag = get_best_response(user_text, kb_data, model)
                st.session_state.chat_history.append({"user": True, "message": user_text})
                st.session_state.chat_history.append({"user": False, "message": bot_reply})
                st.session_state.last_intent = intent_tag
        st.session_state.user_input = ""

    # Input section
    col1, col2 = st.columns([4, 1])
    with col1:
        st.text_input(
            "Ask me anything about Computer Science:", 
            key="user_input", 
            on_change=submit,
            placeholder="Type your question here and press Enter..."
        )
    with col2:
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = [{
                "user": False, 
                "message": "Chat cleared! How can I help you with your CS studies today?"
            }]
            st.rerun()

    # Footer with stats
    if st.session_state.chat_history:
        st.markdown(f"---")
        st.markdown(f"💬 **Chat Stats:** {len([m for m in st.session_state.chat_history if m['user']])} questions asked • Last intent: {st.session_state.last_intent or 'None'}")

if __name__ == "__main__":
    main()
