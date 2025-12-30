import streamlit as st
import os
import base64
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

# --- 1. CONFIGURATION (Dark Mode & Wide Layout) ---
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. PROFESSIONAL DARK UI STYLING ---
st.markdown("""
<style>
    /* Force Dark Theme Colors */
    .stApp {background-color: #0E1117;}
    
    /* Header Gradient */
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #4A90E2, #9013FE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        color: #A0A0A0;
        font-size: 1.1rem;
        margin-top: -10px;
    }

    /* Chat Bubbles (Distinct Colors) */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E2329; /* User (Dark Grey) */
        border: 1px solid #333;
        border-radius: 12px;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #262730; /* AI (Lighter Grey) */
        border: 1px solid #444;
        border-radius: 12px;
    }
    
    /* Input Box Polish */
    .stChatInput {
        padding-bottom: 15px;
    }
    
    /* Hide Streamlit Footer */
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 3. CACHED RESOURCES (SPEED & STABILITY) ---

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_llm(api_key):
    os.environ["GOOGLE_API_KEY"] = api_key
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.6)

# --- 4. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=50)
    st.markdown("### ⚙️ Control Panel")
    
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = st.text_input("🔑 Google API Key", type="password")
    
    uploaded_files = st.file_uploader("📂 Upload PDF(s)", type="pdf", accept_multiple_files=True)
    
    st.divider()
    
    # Custom Persona Setting
    st.markdown("### 🧠 AI Persona")
    system_instruction = st.text_area(
        "System Instructions", 
        value="You are a friendly expert tutor. Explain things clearly.",
        height=70
    )
    
    if st.button("🗑️ Reset Chat", type="primary"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# --- 5. MAIN HEADER ---
col1, col2 = st.columns([8, 2])
with col1:
    st.markdown('<p class="main-header">DocuMind AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Document RAG Intelligence System</p>', unsafe_allow_html=True)

st.divider()

# --- 6. MAIN APP LOGIC ---
if uploaded_files and api_key:
    
    # A. Indexing (The "Brain" Building Phase)
    if "vector_store" not in st.session_state:
        with st.spinner("🧠 Reading & Indexing Documents..."):
            all_chunks = []
            
            # Process each file
            for uploaded_file in uploaded_files:
                # Save temp file
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Load PDF
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                
                # Tag chunks with source filename
                for doc in docs:
                    doc.metadata["source_file"] = uploaded_file.name
                
                # Split Text
                splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            # Create DB (In-Memory for Stability)
            embeddings = get_embeddings()
            st.session_state.vector_store = FAISS.from_documents(all_chunks, embeddings)
            st.toast("Knowledge Base Ready!", icon="✅")

    # B. Split Screen Layout
    col_pdf, col_chat = st.columns([1, 1])
    
    # --- LEFT: PDF VIEWER ---
    with col_pdf:
        st.markdown("##### 📄 Document Viewer")
        file_map = {f.name: f for f in uploaded_files}
        selected_filename = st.selectbox("Select File", list(file_map.keys()), label_visibility="collapsed")
        
        if selected_filename:
            # Display PDF
            base64_pdf = base64.b64encode(file_map[selected_filename].getvalue()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

    # --- RIGHT: CHATBOT ---
    with col_chat:
        st.markdown("##### 💬 AI Chat")
        
        # Init History
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "I've read your documents. Ask me anything!"}]
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Chat Container
        chat_container = st.container(height=530)
        with chat_container:
            for msg in st.session_state.messages:
                avatar = "🤖" if msg["role"] == "assistant" else "👤"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])

        # Input & Processing
        if user_input := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with chat_container:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(user_input)

            with chat_container:
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Thinking..."):
                        llm = get_llm(api_key)
                        
                        # --- THE SMART HYBRID PROMPT ---
                        template = f"""
                        System: {system_instruction}
                        
                        Guidelines:
                        1. GREETINGS: If user says "hi", "thanks", etc., reply naturally.
                        2. MAIN SOURCE: Answer based ONLY on the Context below.
                        3. FALLBACK: If the answer is not in the context, you may use General Knowledge, but explicitly say: "This isn't in the document, but generally..."
                        4. SOURCES: Cite the specific file name if possible.

                        Context: {{context}}
                        Chat History: {{chat_history}}
                        Question: {{question}}
                        Answer:
                        """
                        QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "chat_history", "question"])
                        
                        chain = ConversationalRetrievalChain.from_llm(
                            llm=llm,
                            retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 3}),
                            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
                            return_source_documents=True
                        )
                        
                        result = chain.invoke({"question": user_input, "chat_history": st.session_state.chat_history})
                        answer = result['answer']
                        
                        # Smart Citations
                        sources = result.get('source_documents', [])
                        if sources and "isn't in the document" not in answer.lower():
                            unique_refs = sorted(list(set([f"{doc.metadata.get('source_file')} (p.{doc.metadata.get('page', 0)+1})" for doc in sources])))
                            answer += f"\n\n**Refs:** {', '.join(unique_refs)}"
                        
                        st.markdown(answer)
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.chat_history.append((user_input, answer))

else:
    # Empty State
    st.markdown("""
    <div style="text-align: center; margin-top: 50px;">
        <h3>👋 Welcome to DocuMind AI</h3>
        <p style="color: #888;">Upload your PDFs on the left to start analyzing.</p>
    </div>
    """, unsafe_allow_html=True)