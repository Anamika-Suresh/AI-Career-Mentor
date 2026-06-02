import os
import streamlit as st
from dotenv import load_dotenv
from rag_backend import CareerMentorRAG

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Career Mentor - RAG Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    /* Font definitions */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, .title-font {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
    }
    
    /* Header Styling */
    .main-title {
        background: linear-gradient(135deg, #FF6B6B 0%, #4D96FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #718096;
        margin-bottom: 2rem;
    }
    
    /* Chat Bubble custom styling */
    .chat-bubble {
        padding: 1rem 1.5rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
    }
    
    .user-bubble {
        background-color: #F0F4FF;
        border-left: 5px solid #4D96FF;
        color: #2D3748;
    }
    
    .mentor-bubble {
        background-color: #FDF2F2;
        border-left: 5px solid #FF6B6B;
        color: #2D3748;
    }
    
    /* Source card custom styling */
    .source-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
    }
    
    .source-badge {
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
        background-color: #4D96FF;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        margin-right: 0.5rem;
        display: inline-block;
    }
    
    .source-category {
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
        background-color: #4A5568;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        display: inline-block;
    }
    
    /* Dynamic styling for statuses */
    .status-ok {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
    }
    
    .status-warn {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Session State Initialization
# ----------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "selected_role" not in st.session_state:
    st.session_state.selected_role = "Data Analyst"

# Load default API keys from environment
env_gemini_key = os.getenv("GEMINI_API_KEY", "")
env_openai_key = os.getenv("OPENAI_API_KEY", "")

# ----------------------------------------------------
# Sidebar Configuration
# ----------------------------------------------------
st.sidebar.markdown("<h2 class='title-font' style='margin-bottom:0;'>⚙️ Settings</h2>", unsafe_allow_html=True)
st.sidebar.write("Configure your LLM model and target career path.")

# Provider selection
llm_provider = st.sidebar.selectbox(
    "1. LLM Provider",
    ["Google Gemini", "OpenAI"],
    help="Select the AI service provider. We recommend Gemini for this DeepMind-designed agent!"
)

# API Key input
api_key = ""
if llm_provider == "Google Gemini":
    api_key = st.sidebar.text_input(
        "2. Gemini API Key",
        value=env_gemini_key if env_gemini_key else st.session_state.get("gemini_key", ""),
        type="password",
        help="Input your Gemini API key. Stored only in-memory."
    )
    if api_key:
        st.session_state.gemini_key = api_key
else:
    api_key = st.sidebar.text_input(
        "2. OpenAI API Key",
        value=env_openai_key if env_openai_key else st.session_state.get("openai_key", ""),
        type="password",
        help="Input your OpenAI API key. Stored only in-memory."
    )
    if api_key:
        st.session_state.openai_key = api_key

# Target role selection
target_role = st.sidebar.selectbox(
    "3. Target Career Role",
    ["Data Analyst", "Data Scientist", "ML Engineer", "AI Engineer"],
    help="Selecting a role customizes the system prompt and recommended questions."
)
st.session_state.selected_role = target_role

# Divider
st.sidebar.markdown("---")

# RAG Instantiation & Status check
rag = None
if api_key:
    try:
        # Cache or instantiate RAG
        rag = CareerMentorRAG(llm_provider, api_key)
        st.sidebar.markdown(f"<div class='status-ok'>✓ RAG Connected ({llm_provider})</div>", unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.markdown(f"<div class='status-warn'>⚠️ Connection Failed</div>", unsafe_allow_html=True)
        st.sidebar.caption(f"Error: {e}")
else:
    st.sidebar.markdown("<div class='status-warn'>⚠️ API Key Required</div>", unsafe_allow_html=True)
    st.sidebar.caption("Please input an API Key above or define it in a `.env` file to start.")

# Reset Vector Database
st.sidebar.markdown("<h3 class='title-font'>🧹 Maintenance</h3>", unsafe_allow_html=True)
if st.sidebar.button("Reset Vector Database", help="This will delete all documents and clear the current database."):
    if rag:
        success = rag.clear_database()
        if success:
            st.sidebar.success("Database successfully cleared!")
            st.session_state.chat_history = []
            st.rerun()
        else:
            st.sidebar.error("Failed to clear database.")
    else:
        st.sidebar.warning("RAG connection required to perform reset.")

# ----------------------------------------------------
# Main UI Layout
# ----------------------------------------------------
st.markdown("<h1 class='main-title'>🎓 Personal Career Mentor RAG</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Get hyper-personalized career, resume, and interview prep advice tailored to your target technical path.</div>", unsafe_allow_html=True)

# Define Tabs
tab_chat, tab_docs = st.tabs(["💬 Career Advisor Chat", "📂 Document Manager"])

# ----------------------------------------------------
# Tab 1: Chat Interface
# ----------------------------------------------------
with tab_chat:
    # Role-specific greetings and recommendations
    greetings = {
        "Data Analyst": "📊 Hello! I am your Data Analyst Career Mentor. I'll help you master SQL, dashboarding, Excel, and descriptive stats so you can ace your interviews and polish your metrics-driven resume.",
        "Data Scientist": "🧪 Hello! I am your Data Scientist Career Mentor. Let's practice modeling, probability, experimental designs (A/B testing), and machine learning trade-offs.",
        "ML Engineer": "🤖 Hello! I am your Machine Learning Engineering Mentor. I'll guide you through ML system designs, training pipelines, Docker containerization, FastAPI serving, and latency optimizations.",
        "AI Engineer": "✨ Hello! I am your AI / Generative AI Engineering Mentor. Let's design state-of-the-art RAG pipelines, fine-tune LLMs, write robust prompts, and evaluate multi-agent workflows."
    }
    
    role_questions = {
        "Data Analyst": [
            "What technical tools do I need to learn for this role?",
            "How do I frame SQL project experience on my resume?",
            "How should I prepare for a case study investigation on dropping active users?"
        ],
        "Data Scientist": [
            "What statistics and experimental design questions are common in interviews?",
            "How do I design a robust A/B test for recommender systems?",
            "What metrics should I focus on for an imbalanced classification model like churn?"
        ],
        "ML Engineer": [
            "What are the core components of a production ML recommendation system?",
            "How do I optimize PyTorch model serving latency for production?",
            "What are the key MLOps practices I should display on my resume?"
        ],
        "AI Engineer": [
            "What are the differences between RAG and Fine-Tuning, and when do I use which?",
            "How do I choose the best chunking and index strategy for a dense document corpus?",
            "How do I evaluate a conversational agent for hallucinations?"
        ]
    }
    
    st.markdown(f"<div class='chat-bubble mentor-bubble'><strong>Mentor Persona:</strong> {greetings[target_role]}</div>", unsafe_allow_html=True)
    
    # Check Database Status
    doc_count = 0
    indexed_files = []
    if rag:
        indexed_files = rag.get_indexed_documents()
        doc_count = len(indexed_files)
        
    if doc_count == 0:
        st.warning("⚠️ The knowledge base is currently empty! Click on the **'Document Manager'** tab to upload resumes, interview sheets, or load the pre-made sample documents to get full RAG-driven answers.")
    else:
        st.info(f"📚 RAG Knowledge Base active with **{doc_count}** indexed document(s). Direct citations will be generated.")
        
    # Suggested Questions Section
    st.write("**💡 Suggested Questions (click to ask):**")
    cols = st.columns(3)
    suggested_q = ""
    for idx, q in enumerate(role_questions[target_role]):
        with cols[idx % 3]:
            if st.button(q, key=f"btn_q_{idx}", use_container_width=True):
                suggested_q = q
                
    # Filter selection
    categories_available = ["All Categories", "Job Description", "Resume/Tips", "Interview Prep/Experience", "Company Hiring Patterns"]
    selected_filter = st.selectbox("🎯 Retrieval Filter (Limit search context)", categories_available)
    
    # Input Area
    user_query = st.chat_input("Ask your mentor a question...")
    
    # Process input (either from text input or clicked suggestion)
    final_query = user_query if user_query else suggested_q
    
    if final_query:
        if not rag:
            st.error("Please provide a valid API Key in the settings sidebar to ask questions.")
        else:
            # Append user message
            st.session_state.chat_history.append({"role": "user", "content": final_query})
            
            # Show spinner and run query
            with st.spinner("Retrieving facts and formulating mentor response..."):
                response_dict = rag.query(final_query, target_role, selected_filter)
                
            # Append assistant response
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response_dict["answer"],
                "sources": response_dict["sources"]
            })
            
    # Render Chat History (in reverse or normal order - normal chronological is standard)
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-bubble user-bubble'><strong>You:</strong><br>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble mentor-bubble'><strong>AI Mentor:</strong><br>{msg['content']}</div>", unsafe_allow_html=True)
            
            # If there are sources, render them in an expander
            if msg.get("sources"):
                with st.expander("🔍 Citations & Retrieved Sources", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(f"""
                        <div class='source-card'>
                            <div>
                                <span class='source-badge'>📁 File: {src['source']}</span>
                                <span class='source-category'>🏷️ Category: {src['category']}</span>
                            </div>
                            <div style='margin-top:0.5rem; font-style:italic; font-size:0.9rem; color:#4A5568;'>
                                "{src['content']}"
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
    # Clear chat history button
    if len(st.session_state.chat_history) > 0:
        if st.button("Clear Chat Window"):
            st.session_state.chat_history = []
            st.rerun()

# ----------------------------------------------------
# Tab 2: Document Manager
# ----------------------------------------------------
with tab_docs:
    st.markdown("<h3 class='title-font'>📁 Upload Documents</h3>", unsafe_allow_html=True)
    st.write("Feed the vector database with target resumes, career guides, job postings, or company notes.")
    
    col_uploader, col_sample = st.columns([2, 1])
    
    with col_uploader:
        # File selector details
        doc_category = st.selectbox(
            "Document Category",
            ["Job Description", "Resume/Tips", "Interview Prep/Experience", "Company Hiring Patterns"]
        )
        
        uploaded_files = st.file_uploader(
            "Choose files (.txt, .pdf, .docx)",
            type=["txt", "pdf", "docx"],
            accept_multiple_files=True
        )
        
        if st.button("Index Uploaded Documents"):
            if not rag:
                st.error("RAG Engine not connected. Please supply a valid API Key in the sidebar.")
            elif not uploaded_files:
                st.warning("Please choose one or more files first.")
            else:
                progress_bar = st.progress(0)
                success_count = 0
                
                for idx, file in enumerate(uploaded_files):
                    file_bytes = file.read()
                    try:
                        success = rag.add_document(
                            file_content=file_bytes,
                            filename=file.name,
                            file_type=file.type,
                            category=doc_category
                        )
                        if success:
                            success_count += 1
                    except Exception as e:
                        st.error(f"Error loading {file.name}: {e}")
                        
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                st.success(f"Successfully processed and indexed {success_count} file(s) into Chroma DB!")
                st.rerun()
                
    with col_sample:
        st.markdown("<div style='background-color:#F8FAFC; border:1px dashed #CBD5E0; border-radius:12px; padding:1.5rem; text-align:center;'>", unsafe_allow_html=True)
        st.markdown("<h4 class='title-font' style='margin-top:0;'>💡 Ready-to-use Sample Packs</h4>", unsafe_allow_html=True)
        st.write("Instantly test the platform using our built-in JDs, interview experience reviews, and resume checklists for all roles.")
        
        if st.button("Load Pre-made Career Samples", use_container_width=True):
            if not rag:
                st.error("Please supply an API Key in the sidebar first.")
            else:
                samples = [
                    ("data_analyst_jd.txt", "Job Description"),
                    ("data_scientist_jd.txt", "Job Description"),
                    ("ml_engineer_jd.txt", "Job Description"),
                    ("ai_engineer_jd.txt", "Job Description"),
                    ("data_roles_experience.txt", "Interview Prep/Experience"),
                    ("resume_tips_general.txt", "Resume/Tips")
                ]
                
                sample_dir = "sample_documents"
                success_count = 0
                
                for filename, cat in samples:
                    path = os.path.join(sample_dir, filename)
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            content = f.read()
                        try:
                            # Assume text files
                            success = rag.add_document(
                                file_content=content,
                                filename=filename,
                                file_type="text/plain",
                                category=cat
                            )
                            if success:
                                success_count += 1
                        except Exception as e:
                            st.error(f"Error loading sample {filename}: {e}")
                    else:
                        st.error(f"Sample file not found at {path}")
                        
                if success_count > 0:
                    st.success(f"Indexed {success_count} pre-made career documents!")
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<h3 class='title-font'>📋 Indexed Documents in Vector Store</h3>", unsafe_allow_html=True)
    
    if rag:
        docs = rag.get_indexed_documents()
        if not docs:
            st.info("No documents are currently indexed in the active database. Upload a document or load samples to start.")
        else:
            # Display database table
            import pandas as pd
            df = pd.DataFrame(docs)
            df.columns = ["Filename", "Category", "Database Chunks", "Uploaded Timestamp"]
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Input your API key in the sidebar to see the list of indexed documents.")
