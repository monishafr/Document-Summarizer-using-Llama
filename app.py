########################
# app.py
########################
import streamlit as st
import pdfplumber
import docx
import re
import os
import requests
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter

###############################
# 1. SENSITIVE CONTENT (REGEX)
###############################
SENSITIVE_PATTERNS = {
    "PHONE": r"(\+?\d[\d\-\(\) ]{7,}\d)",
    "EMAIL": r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)",
}

def detect_and_redact(text: str) -> str:
    for label, pattern in SENSITIVE_PATTERNS.items():
        text = re.sub(pattern, f"<REDACTED:{label}>", text)
    return text

###############################
# 2. FILE READERS
###############################
def read_pdf_file(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def read_docx_file(file_path: str) -> str:
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

def read_txt_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

###############################
# 3. CALL GROQCLOUD Llama 3.1
###############################
def call_groqcloud_chat(
    user_prompt: str, 
    system_prompt: str = "", 
    max_tokens=300, 
    temperature=0.7
):
    """
    Calls GroqCloud's Llama 3.1 (8B) via their Chat Completions endpoint.
    system_prompt is optional context or instructions.
    user_prompt is the user content to summarize or Q&A with.
    """
    api_key = os.getenv("GROQCLOUD_API_KEY")
    if not api_key:
        raise ValueError("GROQCLOUD_API_KEY not set in environment or .env file!")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Prepare messages in Chat format
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    # Build payload
    payload = {
        "model": "llama-3.1-8b-instant",  # The production model ID from GroqCloud
        "messages": messages,
        "max_completion_tokens": max_tokens,  # Use 'max_tokens' as requested
        "temperature": temperature
    }

    # Make the request
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        st.error(f"GroqCloud call failed: status={response.status_code}, body={response.text}")
        print("GroqCloud error details:", response.status_code, response.text)
        raise RuntimeError(f"GroqCloud API call failed with code {response.status_code}")

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return "[Error: No text returned from GroqCloud response]"

###############################
# 4. SUMMARIZATION WORKFLOW
###############################
def summarize_document(text: str):
    """
    1) Redact sensitive content
    2) Chunk with LangChain
    3) Summarize each chunk in parallel
    4) Combine partial summaries
    5) Summarize final
    """
    redacted = detect_and_redact(text)

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(redacted)

    def summarize_chunk(chunk: str):
        system_prompt = "You are a helpful assistant. Summarize the user content briefly."
        user_prompt = f"Text:\n{chunk}\n\nSummary:"
        return call_groqcloud_chat(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=300,
            temperature=0.7
        )

    partial_summaries = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for result in executor.map(summarize_chunk, chunks):
            partial_summaries.append(result)

    combined_text = " ".join(partial_summaries)
    final_user_prompt = (
        f"Combine and refine these partial summaries:\n\n{combined_text}\n\nFinal Summary:"
    )
    final_result = call_groqcloud_chat(
        user_prompt=final_user_prompt,
        system_prompt="You are a helpful assistant. Please produce a concise final summary.",
        max_tokens=300,
        temperature=0.7
    )

    return final_result

###############################
# 5. Q&A WORKFLOW
###############################
def answer_question(document_text: str, question: str):
    """
    Provide a simple Q&A approach by feeding entire doc as context.
    For huge docs, you'd do retrieval chunking or RAG.
    """
    redacted = detect_and_redact(document_text)
    system_prompt = "You are a helpful assistant. Use the user's text to answer their question."
    user_prompt = f"Document:\n{redacted}\n\nQuestion: {question}\nAnswer:"
    answer = call_groqcloud_chat(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=300,
        temperature=0.7
    )
    return answer

###############################
# 6. STREAMLIT WEB APP
###############################
def main():
    st.set_page_config(page_title="Llama 3.1 Summarization & QnA", layout="centered")
    st.title("Llama 3.1 (8B) Summarization & Q&A via GroqCloud")

    uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        temp_filename = uploaded_file.name
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Read the file
        if temp_filename.endswith(".pdf"):
            doc_text = read_pdf_file(temp_filename)
        elif temp_filename.endswith(".docx"):
            doc_text = read_docx_file(temp_filename)
        elif temp_filename.endswith(".txt"):
            doc_text = read_txt_file(temp_filename)
        else:
            st.error("Unsupported file type.")
            return

        os.remove(temp_filename)  # Cleanup

        st.subheader("Document Preview:")
        st.write(doc_text[:500] + "..." if len(doc_text) > 500 else doc_text)

        if st.button("Summarize Document"):
            with st.spinner("Summarizing..."):
                summary = summarize_document(doc_text)
            st.success(summary)

        st.subheader("Ask a Question")
        user_question = st.text_input("Your question about this document:")
        if user_question:
            with st.spinner("Generating answer..."):
                answer_text = answer_question(doc_text, user_question)
            st.info(answer_text)

if __name__ == "__main__":
    main()





