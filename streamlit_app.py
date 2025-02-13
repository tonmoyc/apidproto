import os
import streamlit as st
from pathlib import Path
from getpass import getpass
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.components.converters import MarkdownToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import OpenAIChatGenerator


# --------------------------- Initialize Streamlit ---------------------------
st.set_page_config(page_title="Haystack Chatbot", page_icon="🤖")
st.title("📝 API Discovery Assistant (prototype)")

# --------------------------- Setup Paths ---------------------------
script_dir = Path(__file__).parent
output_dir = script_dir / 'products'

# Ensure the output directory exists
if not output_dir.exists() or not output_dir.is_dir():
    st.error(f"❌ Output directory '{output_dir}' does not exist.")
    st.stop()

# Get all files in the output directory
files = list(output_dir.glob("**/*"))

# Ensure files exist
if not files:
    st.error(f"❌ No files found in '{output_dir}'. Please add markdown files and restart.")
    st.stop()


# --------------------------- Initialize Components ---------------------------
document_store = InMemoryDocumentStore()

file_type_router = FileTypeRouter(mime_types=["text/markdown"])
markdown_converter = MarkdownToDocument()
document_joiner = DocumentJoiner()
document_cleaner = DocumentCleaner()
document_splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=50)
document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
document_writer = DocumentWriter(document_store)

# --------------------------- Build Preprocessing Pipeline ---------------------------
preprocessing_pipeline = Pipeline()
preprocessing_pipeline.add_component("file_type_router", file_type_router)
preprocessing_pipeline.add_component("markdown_converter", markdown_converter)
preprocessing_pipeline.add_component("document_joiner", document_joiner)
preprocessing_pipeline.add_component("document_cleaner", document_cleaner)
preprocessing_pipeline.add_component("document_splitter", document_splitter)
preprocessing_pipeline.add_component("document_embedder", document_embedder)
preprocessing_pipeline.add_component("document_writer", document_writer)

preprocessing_pipeline.connect("file_type_router.text/markdown", "markdown_converter.sources")
preprocessing_pipeline.connect("markdown_converter", "document_joiner")
preprocessing_pipeline.connect("document_joiner", "document_cleaner")
preprocessing_pipeline.connect("document_cleaner", "document_splitter")
preprocessing_pipeline.connect("document_splitter", "document_embedder")
preprocessing_pipeline.connect("document_embedder", "document_writer")

# Run preprocessing
with st.spinner("🔄 Processing markdown files..."):
    preprocessing_pipeline.run({"file_type_router": {"sources": files}})
st.success("✅ Documents processed successfully!")

# --------------------------- Set Up OpenAI API Key ---------------------------

if "OPENAI_API_KEY" not in os.environ:
    api_key = st.text_input("🔑 Enter your OpenAI API key:", type="password")
    if st.button("Save API Key"):
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("API Key Saved! 🎉")

if "OPENAI_API_KEY" not in os.environ:
    st.error("❌ OpenAI API key is required to proceed.")
    st.stop()


# --------------------------- Define LLM Query Pipeline ---------------------------
template = [
    ChatMessage.from_user(
        """
Answer the questions based on the given context.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

In your answers always mention the recommended API Product name, such as Lead Logistics Booking API, Customer Order/Purchase Order API, Cargo Receipt API, etc.
If there are more than 1 API product, mention those as well. In that case, give a short message explaining why?

Question: {{ question }}
Answer:
"""
    )
]

qa_pipeline = Pipeline()
qa_pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
qa_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
qa_pipeline.add_component("chat_prompt_builder", ChatPromptBuilder(template=template))
qa_pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini"))

qa_pipeline.connect("embedder.embedding", "retriever.query_embedding")
qa_pipeline.connect("retriever", "chat_prompt_builder.documents")
qa_pipeline.connect("chat_prompt_builder.prompt", "llm.messages")


# --------------------------- Streamlit Chat UI ---------------------------
st.subheader("💬 Chat with the Bot")

# Store chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# Input box for user question
user_input = st.chat_input("Type your question here...")

if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run the pipeline
    with st.spinner("🔍 Searching for an answer..."):
        response = qa_pipeline.run({
            "embedder": {"text": user_input},
            "chat_prompt_builder": {"question": user_input}
        })

    bot_response = response["llm"]["replies"][0].text

    # Add bot response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)
