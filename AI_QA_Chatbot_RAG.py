
# Step 0: Install Required Libraries (Run these in a Colab notebook or command line)
# !pip install pypdf transformers sentence-transformers chromadb langchain PyPDF2 gradio beautifulsoup4 nltk torch accelerate

# Step 1: Import Libraries
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re, json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import chromadb
import gradio as gr

# Step 2: Download NLTK Resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Step 3: Initialize Models and Vector Store (ChromaDB)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
phi_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float16, device_map="auto")

chroma_client = chromadb.PersistentClient(path="chroma_gradio_db")
collection = chroma_client.get_or_create_collection(name="documents")

# Step 4: Helper Functions

# Extract text from uploaded PDF files (starting from page 3)
def extract_text_from_pdfs(pdf_files):
    all_text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages[2:]:  # Skip first 2 pages
            text = page.extract_text()
            if text:
                all_text += text + "\n"
    return all_text

# Extract text from one or more URLs
def extract_text_from_urls(url_input):
    urls = [url.strip() for url in url_input.split(",")]
    all_text = ""
    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            paragraphs = soup.find_all("p")
            for para in paragraphs:
                text = para.get_text().strip()
                if text:
                    all_text += text + "\n"
        except Exception as e:
            all_text += f"\n❌ Failed to extract from {url}: {e}\n"
    return all_text

# Clean and preprocess the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\W+", " ", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Store text chunks in ChromaDB after embedding
def store_embeddings(cleaned_text):
    chunk_size = 500
    chunks = [cleaned_text[i:i+chunk_size] for i in range(0, len(cleaned_text), chunk_size)]
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        collection.add(ids=[str(i)], embeddings=[embedding], metadatas=[{"text": chunk}])
    return f"{len(chunks)} chunks stored in Vector DB!"

# Retrieve relevant chunks and use LLM to answer the query
def answer_query(query):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    retrieved_chunks = [res["text"] for res in results["metadatas"][0]]
    context = "\n".join(retrieved_chunks)

    prompt = f"""You are a helpful assistant. Based only on the context below, provide a concise and accurate answer.

Context:
{context}

Question:
{query}

Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(phi_model.device)
    outputs = phi_model.generate(**inputs, max_new_tokens=350, do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    answer = response.split("Answer:")[-1].strip() if "Answer:" in response else response.strip()
    return answer

# Step 5: Gradio App Logic

extracted_text_global = ""

def upload_and_extract(pdf_files, urls, show_preview):
    global extracted_text_global
    pdf_text = extract_text_from_pdfs(pdf_files) if pdf_files else ""
    url_text = extract_text_from_urls(urls) if urls else ""
    combined_text = pdf_text + url_text
    cleaned = clean_text(combined_text)
    extracted_text_global = cleaned
    store_result = store_embeddings(cleaned)
    preview = cleaned[:1000] if show_preview else "Preview not requested."
    return preview, store_result

def handle_query(user_query):
    answer = answer_query(user_query)
    return answer

# Step 6: Gradio UI

with gr.Blocks() as demo:
    gr.Markdown("## 📚 AI Chatbot with Upload PDFs or URLs and ask context-aware questions using ChromaDB & Phi-2 LLM")

    with gr.Row():
        pdf_input = gr.File(file_types=[".pdf"], file_count="multiple", label="Upload PDFs")
        url_input = gr.Textbox(label="Paste URLs (comma-separated)")
        show_preview = gr.Checkbox(label="Show Cleaned Text Preview", value=True)

    extract_btn = gr.Button("Extract and Embed")
    output_preview = gr.Textbox(label="🧼 Cleaned Text Preview (optional)", lines=10)
    embed_status = gr.Textbox(label="📥 Embedding Status")

    with gr.Row():
        query_input = gr.Textbox(label="💬 Ask your question")
        query_btn = gr.Button("Get Answer")
        answer_output = gr.Textbox(label="🤖 Chatbot Answer", lines=5)

    extract_btn.click(upload_and_extract, inputs=[pdf_input, url_input, show_preview], outputs=[output_preview, embed_status])
    query_btn.click(handle_query, inputs=query_input, outputs=answer_output)

demo.launch()
