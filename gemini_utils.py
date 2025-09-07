import os
import re
import numpy as np
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from groq import Groq

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

class Word2VecEmbeddings(Embeddings):
    def _init_(self):
        self.model = None

    def train_model(self, texts):
        tokenized = [re.findall(r'\b\w+\b', doc.lower()) for doc in texts]
        self.model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4)

    def embed_query(self, text):
        tokens = re.findall(r'\b\w+\b', text.lower())
        vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]
        return np.mean(vectors, axis=0).tolist() if vectors else [0]*100

    def embed_documents(self, texts):
        self.train_model(texts)
        return [self.embed_query(text) for text in texts]

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def chunk_text(text, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents([text])

def top_chunks_tfidf(chunks, top_k=3):
    texts = [doc.page_content for doc in chunks]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()

    print("\n📊 TF-IDF Chunk Scores:")
    for i, (score, text) in enumerate(zip(scores, texts)):
        print(f"\n🧩 Chunk {i+1} | Score: {score:.4f}")
        print(text.strip()[:300] + "...\n")

    top_indices = scores.argsort()[::-1][:top_k]
    top_chunks = [chunks[i] for i in top_indices]

    print("\n✅ Top TF-IDF Chunks:")
    for i, doc in enumerate(top_chunks):
        print(f"\n🔥 Chunk {i+1}:\n{doc.page_content.strip()[:300]}...\n")

    return top_chunks

def get_summary(full_text, length='medium', style='concise'):
    chunks = chunk_text(full_text)
    top_chunks = top_chunks_tfidf(chunks, top_k=3)
    combined_text = "\n".join([doc.page_content for doc in top_chunks])

    print("\n📦 Combined Text Sent to Groq:\n")
    print(combined_text.strip()[:1000] + "...\n")

    prompt = (
        f"Summarize the following document in {length} length and {style} style.\n"
        "Use only the content below.\n"
        "Ensure grammar is correct and writing is polished.\n"
        "Output clean plain text only — no markdown, no bullets, no emojis.\n"
        "Use numbered points or paragraphs.\n\n"
        f"Document:\n{combined_text}"
    )

    chat_completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that produces grammatically correct, clean text summaries."},
            {"role": "user", "content": prompt}
        ]
    )

    return chat_completion.choices[0].message.content.strip()

def create_faiss_index(chunks, embedding_model):
    return FAISS.from_documents(chunks, embedding_model)

def get_chat_answer(full_text, user_question):
    chunks = chunk_text(full_text)
    texts = [doc.page_content for doc in chunks]

    embedding_model = Word2VecEmbeddings()
    embedding_model.train_model(texts)

    faiss_index = create_faiss_index(chunks, embedding_model)
    retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(user_question)

    question_words = re.findall(r'\b\w+\b', user_question.lower())
    print("\n🔎 Extracted Keywords from Question:")
    print(", ".join(question_words))

    print("\n📚 Top Chunks Retrieved for Q&A:")
    for i, doc in enumerate(docs):
        print(f"\n🧩 Chunk {i+1}:\n{doc.page_content.strip()[:300]}...\n")

    context = "\n".join([doc.page_content for doc in docs])

    prompt = (
        "Answer the following question based only on the document.\n"
        "Ensure the answer is clear, grammatically correct, and concise.\n"
        "Avoid any markdown, emojis, or special formatting.\n\n"
        f"Document:\n{context}\n\n"
        f"Question: {user_question}\n\nAnswer:"
    )

    chat_completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You answer clearly using only the provided document."},
            {"role": "user", "content": prompt}
        ]
    )

    return chat_completion.choices[0].message.content.strip()
