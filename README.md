# AI-Chatbot-PDF-URLs-RAG-powered-QA-System-with-ChromaDB-Phi-2-LLM-

![URLs rag chatbot](https://github.com/user-attachments/assets/9b242302-b4e7-422a-a325-c8b6c2e04b4c)

---

* PDF/URL ingestion
* Text cleaning + preprocessing (NLP)
* Embedding storage in **ChromaDB**
* Query-answering using **retrieval + Phi-2 LLM**
* A user interface using **Gradio**

---

### ✅ **Complete Workflow Explanation (Step-by-Step)**

---

##  **STEP 0: Library Installation**

### installed the required libraries:

```python
!pip install pypdf transformers sentence-transformers chromadb langchain PyPDF2 gradio
```

### These packages help with:

* `pypdf`/`PyPDF2`: PDF reading
* `transformers`, `sentence-transformers`: Text embeddings + language modeling
* `chromadb`: Storing/querying vector embeddings
* `gradio`: Building an interactive UI

---

##  **STEP 1: INPUT TEXT SOURCE (PDFs / URLs)**

### You created options to:

* Upload PDFs using `files.upload()` or `gr.File`
* Extract web content from multiple URLs using `requests + BeautifulSoup`

```python
def extract_text_from_uploaded_pdfs()
def extract_text_from_urls()
```

### You allow **flexibility** in how users provide data — PDFs or online articles. You **start from Page 3** to skip cover/title/index pages.

---

## **STEP 2: TEXT CLEANING + NLP PREPROCESSING**

### You cleaned the raw text using:

* Lowercasing
* Removing special characters
* Tokenization
* Stopword removal
* Lemmatization

```python
def clean_text(text)
```

### To standardize and simplify the text before embedding.
Clean text ensures:

* Better quality embeddings
* Reduced noise
* Faster similarity matching

---

##  **STEP 3: TEXT CHUNKING + EMBEDDING STORAGE (ChromaDB)**

### What did:

* You split cleaned text into chunks of 500 characters.
* Encoded them using SentenceTransformer: `all-MiniLM-L6-v2`
* Stored in **ChromaDB** for retrieval using vector search.

```python
def store_embeddings(cleaned_text)
```

###  Why:

* ChromaDB lets you store + retrieve similar text chunks using **semantic similarity**
* Splitting is essential to avoid long input limits and make retrieval efficient.

---

## **STEP 4: RETRIEVAL AUGMENTED GENERATION (RAG)**

###  What did:

* Query is embedded using the same model.
* Top 3 relevant chunks are retrieved using vector similarity.
* Combined into context.
* Passed to the **Phi-2 LLM** for final answer generation.

```python
def retrieve_relevant_chunks(query)
def answer_query(query)
```

### Why:

* RAG improves accuracy by grounding LLM answers in retrieved facts.
* You prevent hallucination by instructing the model to answer *based only on retrieved context*.

---

##  **STEP 5: UI WITH GRADIO**

### What did:

You built a front-end:

* File & URL inputs
* Extract and embed button
* Query input and chatbot output
* Checkbox for previewing cleaned text

```python
with gr.Blocks() as demo:
    ...
```

###  Why:

* Enables **non-technical users** to interact with your tool
* Real-time document Q\&A from PDFs or websites — super useful for education, research, and enterprise!

---

## ✅ FULL PIPELINE SUMMARY:

| Stage         | What Happens                                 | Tools Used                    |
| ------------- | -------------------------------------------- | ----------------------------- |
| 1. Input      | Upload PDFs / enter URLs                     | Gradio, BeautifulSoup, PyPDF2 |
| 2. Extraction | Text is extracted                            | BeautifulSoup, PyPDF2         |
| 3. Cleaning   | Lowercasing, remove stopwords, lemmatization | NLTK, regex                   |
| 4. Embedding  | Text chunks embedded & stored                | SentenceTransformer, ChromaDB |
| 5. Query      | Embed query & retrieve similar chunks        | ChromaDB                      |
| 6. LLM Answer | Use context + LLM to answer                  | Phi-2, Transformers           |
| 7. UI         | Gradio interface                             | Gradio                        |

---

### 🚀 Project Use-Cases:

* 📚 **Study assistant**: Upload notes/articles → ask questions
* 🧾 **Document QA**: Company policies, legal docs, research papers
* 🌐 **Web scraper chatbot**: Ask questions from multiple web articles

---

##  To Suggest Improvement (Future Work):

1. **Add deletion/reset button** to clear stored vectors in ChromaDB.
2. **Chunk smarter**: Split based on sentences or paragraphs (e.g., using `nltk.sent_tokenize`)
3. **Add summarization mode** using LLM
4. **Support DOCX/HTML** formats too
5. **UI polishing**: Add download options, loading spinners, etc.

---
