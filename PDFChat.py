import os
import certifi
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from PIL import Image

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# SSL Fix
os.environ['SSL_CERT_FILE'] = certifi.where()

# Load env
load_dotenv()

# UI Setup
img = Image.open(r"C:\Users\palak\OneDrive\Desktop\DocTalk\Artifacts\Image Resources\images.jpeg")
st.set_page_config(page_title="DocTalk: Your PDFs talk back!", page_icon=img)
st.header("Ask Your PDF 📄")

pdf = st.file_uploader("Upload your PDF", type="pdf")

# -------------------- PROMPT --------------------
prompt_template = """
Summarize the following content in 3-4 clear bullet points.

Context:
{context}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context"]
)

# -------------------- VECTOR DB --------------------
@st.cache_resource
def process_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""

    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=400,
        chunk_overlap=100,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base

# -------------------- LLM --------------------
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text-generation",   
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0.3
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# -------------------- MAIN --------------------
if pdf is not None:
    knowledge_base = process_pdf(pdf)

    query = st.text_input("Ask your question (e.g., 'Summarize this resume')")

    if query:
        llm = load_llm()

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=knowledge_base.as_retriever(search_kwargs={"k": 2}),
            chain_type_kwargs={"prompt": PROMPT}
        )

        response = qa_chain.run(query)

        st.success(response.strip())