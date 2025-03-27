# Intelligent Multiformat Document Summarization & Q&A  
**Using Llama 3.1 (8B) via GroqCloud**

## Overview  
This repository demonstrates how to upload and summarize large documents (PDF, DOCX, TXT) and ask questions about them using Llama 3.1 (8B) hosted on GroqCloud. The application is built with **Streamlit** for a user-friendly interface and **LangChain** for chunk-based text splitting, while **regex-based redaction** ensures sensitive data is masked. The entire solution can be deployed to **Streamlit Cloud** to share with others.

> 🌐 [Try the App Live](https://document-summarizer-llama.streamlit.app/)

## Features  
- **Multi-Format Support**: Seamlessly handle PDF, DOCX, and TXT.  
- **Regex Redaction**: Automatic masking of phone numbers and emails.  
- **Chunk-Based Summarization**: Faster processing using parallel requests.  
- **Simple Q&A**: Pose questions about your uploaded document context.  
- **GroqCloud Integration**: Outsource heavy LLM inference to a production-ready platform.

## Quick Start  

### Clone this repo:
```bash
git clone https://github.com/username/this-repo.git
cd this-repo
```

### Install dependencies (in a virtual environment):
```bash
pip install -r requirements.txt
```

### Set API Key:
Create a `.env` file or environment variable named `GROQCLOUD_API_KEY`.

If deploying on Streamlit Cloud, store `GROQCLOUD_API_KEY` in your app’s Secrets.

### Run Locally:
```bash
streamlit run app.py
```

## Usage:
- Open the localhost link.
- Upload a PDF, DOCX, or TXT.
- Click **Summarize Document** to see a concise summary.
- Ask a question in the input box for Q&A.

---

## Deploying to Streamlit Cloud
1. Push your code to GitHub.  
2. On Streamlit Cloud, create a new app linked to your repo’s `app.py`.  
3. Add your `GROQCLOUD_API_KEY` in **“Secrets.”**  
4. After deployment, share the resulting URL with colleagues or friends.  
   > ✅ Example: [https://document-summarizer-llama.streamlit.app/](https://document-summarizer-llama.streamlit.app/)

---

## Future Enhancements  
- **RAG (Retrieval-Augmented Generation)** for better Q&A on extremely large documents.  
- **Additional Regex Patterns** for broader sensitive data detection (SSNs, addresses, etc.).  
- **Advanced UI** with multi-file support or conversation memory.
