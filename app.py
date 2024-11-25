# streamlit run [FILE_NAME] [ARGUMENTS]

from .src.rag_chatbot import *
import streamlit as st
    
# Load environment variables
openai_api_key = load_env_variables(".env")
os.environ['OPENAI_API_KEY'] = openai_api_key

# Configs
UPDATE_VECTOR_STORE = False
PERSIST_DIRECTORY = "./chroma_langchain_db"
OPEN_AI_MODEL = "gpt-4o-mini"
OPEN_AI_EMBEDDING_MODEL = "text-embedding-3-small"

if UPDATE_VECTOR_STORE == True:   

    # Download files from Google Drive
    download_files_from_google_drive("googledrive_folders.json")

    # Collect and load HTML files
    html_files = collect_downloaded_html_files_path("tmp/")
    docs = load_html_documents(html_files)
    docs = docs[3:7]  # Subset for testing

    # Build vectorstore
    vectorstore = build_vectorstore(docs, PERSIST_DIRECTORY, OPEN_AI_EMBEDDING_MODEL)

# Load vectorstore
vectorstore = get_vectorstore(PERSIST_DIRECTORY, OPEN_AI_EMBEDDING_MODEL)

# Create RAG chain
rag_chain = create_rag_chain(vectorstore, model = OPEN_AI_MODEL)

# Streamlit app
st.title("Chatbot")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""


if st.button("Clear Memory"):
    st.session_state["chat_history"] = []
    st.session_state["user_input"] = ""

user_input = st.text_input("Enter your query:", value=st.session_state["user_input"], key="user_input")

if user_input:
    chat_history = st.session_state.get("chat_history", [])
    answer = get_answer_and_update_history(rag_chain, user_input, chat_history)
    st.text_area("Bot:", answer)
    st.session_state["chat_history"] = chat_history
