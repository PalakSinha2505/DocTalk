import os
import re
import certifi
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# =========================================================
# SSL FIX
# =========================================================

os.environ["SSL_CERT_FILE"] = certifi.where()

# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="DocTalk - AI PDF Assistant",
    layout="wide"
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("DocTalk")

st.sidebar.markdown("""
AI-powered PDF Question Answering System

Upload any PDF and ask:
- Summaries
- Skills
- Technologies
- Key points
- Explanations
""")

# =========================================================
# MAIN TITLE
# =========================================================

st.title("DocTalk - Your PDFs Talk Back")

st.markdown(
    "Upload a PDF and ask questions in natural language."
)

# =========================================================
# FILE UPLOAD
# =========================================================

pdf = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)

# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    text = text.replace("•", "\n• ")

    return text.strip()

# =========================================================
# PDF PROCESSING
# =========================================================

@st.cache_resource
def process_pdf(pdf_file):

    pdf_reader = PdfReader(pdf_file)

    text = ""

    for page in pdf_reader.pages:

        extracted = page.extract_text()

        if extracted:
            text += extracted + "\n"

    text = clean_text(text)

    if not text.strip():
        return None, None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        chunks,
        embeddings
    )

    return vector_store, text

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model

# =========================================================
# GENERATE RESPONSE
# =========================================================

def generate_answer(query, docs=None, full_text=None):

    tokenizer, model = load_model()

    summarize_keywords = [
        "summarize",
        "summary",
        "overview",
        "brief",
        "explain this document"
    ]

    is_summary = any(
        word in query.lower()
        for word in summarize_keywords
    )

    # =====================================================
    # SUMMARY MODE
    # =====================================================

    if is_summary:

        context = full_text[:6000]

        prompt = f"""
You are an AI document analysis assistant.

Read the document carefully and generate a detailed professional summary.

Instructions:
- Write detailed bullet points.
- Explain the purpose of the document.
- Mention important modules, technologies, architecture, features, and implementation details.
- Explain concepts properly instead of copying raw lines.
- Keep the response descriptive and meaningful.
- Generate at least 8-10 points.

DOCUMENT:
{context}

DETAILED SUMMARY:
"""

    # =====================================================
    # QUESTION ANSWERING MODE
    # =====================================================

    else:

        context = "\n\n".join([
            doc.page_content for doc in docs
        ])

        prompt = f"""
You are an intelligent AI assistant.

Answer the question using ONLY the provided context.

Instructions:
- Give detailed and explanatory answers.
- Explain features and technologies clearly.
- Use proper sentences and formatting.
- Do not copy raw chunks directly.
- If information is missing, say:
"I could not find this information in the document."

CONTEXT:
{context}

QUESTION:
{query}

DETAILED ANSWER:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    outputs = model.generate(
        **inputs,

        max_new_tokens=300,

        temperature=0.2,

        do_sample=True,

        top_p=0.9,

        repetition_penalty=1.2
    )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # =====================================================
    # CLEAN RESPONSE
    # =====================================================

    remove_phrases = [
        "DETAILED ANSWER:",
        "ANSWER:",
        "SUMMARY:",
        "DETAILED SUMMARY:"
    ]

    for phrase in remove_phrases:

        if phrase in response:
            response = response.split(phrase)[-1]

    return response.strip()

# =========================================================
# MAIN LOGIC
# =========================================================

if pdf is not None:

    with st.spinner("Processing PDF..."):

        knowledge_base, full_text = process_pdf(pdf)

    if knowledge_base is None:

        st.error(
            "Could not extract readable text from the PDF."
        )

    else:

        st.success("PDF processed successfully.")

        query = st.text_input(
            "Ask a question about your PDF"
        )

        example_questions = [
            "Summarize this document",
            "What technologies are mentioned?",
            "Explain the main project",
            "What skills are listed?",
            "Highlight key features"
        ]

        with st.expander("Example Questions"):

            for q in example_questions:
                st.write(f"- {q}")

        if query:

            with st.spinner("Generating response..."):

                summarize_keywords = [
                    "summarize",
                    "summary",
                    "overview",
                    "brief"
                ]

                is_summary = any(
                    word in query.lower()
                    for word in summarize_keywords
                )

                docs = []

                if not is_summary:

                    retriever = knowledge_base.as_retriever(
                        search_kwargs={"k": 5}
                    )

                    docs = retriever.get_relevant_documents(
                        query
                    )

                response = generate_answer(
                    query=query,
                    docs=docs,
                    full_text=full_text
                )

            # =================================================
            # RESPONSE
            # =================================================

            st.subheader("Answer")

            st.write(response)

            # =================================================
            # SOURCE CONTEXT
            # =================================================

            with st.expander("Retrieved Context"):

                for i, doc in enumerate(docs):

                    st.markdown(f"### Chunk {i+1}")

                    st.write(doc.page_content)

                    st.markdown("---")