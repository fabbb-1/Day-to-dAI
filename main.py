from dotenv import load_dotenv
import os
from openai import OpenAI
import requests
from pathlib import Path
import fitz # PyMuPDF
from tqdm.auto import tqdm # for progress bars, requires !pip install tqdm
import spacy
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import numpy as np
from pymongo import UpdateOne, ASCENDING
from pymongo.operations import SearchIndexModel
import time

# Load spaCy's English model for sentence splitting
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

# Constants
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
BULK_WRITE_SIZE = 100 
EMBEDDING_DIM = 768

def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip()
    cleaned_text = cleaned_text.replace("  ", " ")
    return cleaned_text

def open_and_read_pdf_by_outline(pdf_path: str) -> list[str]:
    """
    Opens a PDF file, reads its text content based on the outline (chapters).

    Parameters:
        pdf_path (str): The file path to the PDF document to be opened and read.

    Returns:
        list[str]: A list of strings, where each string contains the text of a chapter.
    """
    doc = fitz.open(pdf_path)
    doc_list = []

    outline = doc.get_toc()
    for current_index, (lev, title, page_n) in enumerate(outline):
        start_page_number = page_n - 1

        # Find the end page of the current chapter
        end_page_number = outline[current_index+1][-1] if current_index+1 < len(outline)\
                          else len(doc) - 1 # Last page of the document

        chapter_text = ""
        for page_num in range(start_page_number, end_page_number + 1):
            if page_num < len(doc):
                 chapter_text += doc[page_num].get_text()

        doc_list.append(text_formatter(chapter_text))
    return doc_list

def chunk_text_with_padding(text: str, chunk_size: int, padding: int) -> list[str]:
    """
    Splits text into chunks of approximate size, preserving sentences and adding padding.

    Args:
        text (str): The input text to chunk.
        chunk_size (int): The approximate size of each text chunk in characters.
        padding (int): The number of sentences to overlap between adjacent chunks.

    Returns:
        list[str]: A list of text chunks with sentence-based padding.
    """
    doc = nlp(text)
    sentences = [str(sent) for sent in doc.sents]
    chunks = []
    current_chunk_sentences = []
    current_chunk_len = 0

    chunk_id = 0
    for sentence in sentences:
        sentence_len = len(sentence) + 1 # +1 for space

        if current_chunk_len + sentence_len > chunk_size and current_chunk_sentences:
            chunk = " ".join(current_chunk_sentences)
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk,
                "chunk_char_count": len(chunk),
                "chunk_word_count": len(chunk.split(" ")),
                "chunk_token_count": len(chunk) / 4,  # 1 token = ~4 chars
                "chunk_sentence_count": len(current_chunk_sentences)
            })
            # Start the next chunk with padding sentences from the end of the current chunk
            current_chunk_sentences = current_chunk_sentences[-padding:] if padding > 0 else []
            current_chunk_len = sum(len(s) + 1 for s in current_chunk_sentences)
            chunk_id += 1

        current_chunk_sentences.append(sentence)
        current_chunk_len += sentence_len

    if current_chunk_sentences:
        chunk = " ".join(current_chunk_sentences)
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk,
            "chunk_char_count": len(chunk),
            "chunk_word_count": len(chunk.split(" ")),
            "chunk_token_count": len(chunk) / 4,  # 1 token = ~4 chars
            "chunk_sentence_count": len(current_chunk_sentences)
        })
    return chunks

def connect_to_mongo(uri: str, db_name: str):
    """Connects to MongoDB and returns the client and the specified database object."""
    client_mongo = MongoClient(uri, server_api=ServerApi('1'),
                        tlsAllowInvalidCertificates=True)
    try:
        client_mongo.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        # Correzione: Sollevare un ConnectionError per una migliore gestione degli errori
        raise ConnectionError(f"Failed to connect to MongoDB: {e}")
        
    db = client_mongo[db_name]
    return client_mongo, db

def build_collection_chunks(db, chunked_text):
    """Resetta la collection 'employee_reviews' e inserisce i dati."""
    COLLECTION_NAME = 'employee_reviews'
    
    # 1. Pulizia e Preparazione (Elimina la vecchia collezione per inserimento pulito)
    if COLLECTION_NAME in db.list_collection_names():
        print(f"La collezione '{COLLECTION_NAME}' esiste. Eliminazione per inserimento pulito...")
        db[COLLECTION_NAME].drop()
        
    collection = db[COLLECTION_NAME]
    
    # 2. Inserimento
    collection.insert_many(chunked_text)
    assert collection.count_documents({}) == len(chunked_text)
    print(f"Inserted {len(chunked_text)} chunks into MongoDB.")
    
    # 3. Creazione di un indice
    collection.create_index([("chunk_id", ASCENDING)])
    
    return collection

def embed_texts(collection):
    """Genera gli embedding e li salva in batch in MongoDB."""
    
    # ATTENZIONE: Se non hai una GPU con CUDA, imposta device="cpu" nel tuo .env
    device = "cuda" if "cuda" in os.getenv("EMBEDDING_DEVICE", "cpu").lower() and np.any([1]) else "cpu"
    print(f"Loading SentenceTransformer on device: {device}")
    
    embedding_model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL_NAME,
                                      device=device)

    # Prepara il cursore per processare solo i documenti che non hanno ancora l'embedding
    documents_cursor = collection.find({"embedding": {"$exists": False}}) 
    total_docs_to_update = collection.count_documents({"embedding": {"$exists": False}})
    
    print(f"Found {total_docs_to_update} documents without embeddings to process.")

    bulk_updates = []
    
    for chunk in tqdm(documents_cursor,
                    total=total_docs_to_update,
                    desc='Generating and Updating Embeddings'):
        
        # Generazione dell'Embedding
        embedding = embedding_model.encode(chunk['text'], convert_to_numpy=True) 

        # Preparazione dell'Aggiornamento Batch
        # Filtra per l'ID del documento e aggiunge/setta il campo "embedding"
        bulk_updates.append(
            UpdateOne(
                {"_id": chunk['_id']}, # Correzione: Filtro per ID
                {"$set": {"embedding": embedding.tolist()}} # Correzione: Salvataggio come lista Python
            )
        )

        # CORREZIONE CRITICA: Esecuzione dell'Aggiornamento se la dimensione del batch è raggiunta
        if len(bulk_updates) >= BULK_WRITE_SIZE:
            collection.bulk_write(bulk_updates)
            bulk_updates = [] 

    # Esecuzione degli aggiornamenti rimanenti
    if bulk_updates:
        collection.bulk_write(bulk_updates)

    # Verifica finale
    updated_count = collection.count_documents({"embedding": {"$exists": True}})
    print(f"Finished generating and updating embeddings. Total documents with embeddings: {updated_count}.")
    
    if updated_count != collection.count_documents({}):
        print("ATTENZIONE: Alcuni documenti potrebbero non aver ricevuto l'embedding. Controlla gli errori.")

def build_vector_search_index(collection):
    """Crea l'indice di ricerca vettoriale per la collection."""
    
    index_name="vector_index1"
    
    # Uso delle costanti definite globalmente (Correzione dei placeholder)
    embedding_dim = EMBEDDING_DIM 
    embeddings_path = 'embedding' 

    # Definizione dell'indice Atlas Vector Search (metodo a dizionario)
    search_index_model = {
        "definition": {
            "fields": [
                {
                    "type": "vector",
                    "numDimensions": embedding_dim,
                    "path": embeddings_path,
                    "similarity": "cosine"
                }
            ]
        },
        "name": index_name,
        "type": "vectorSearch"
    }

    print(f"Tentativo di creare l'indice Atlas Vector Search '{index_name}'...")
    try:
        # Correzione: Metodo Collection per creare l'indice di ricerca
        collection.create_search_index(model=search_index_model)
        print(f"Indice '{index_name}' creato con successo.")
    except Exception as e:
        if "Index with name" in str(e):
             print(f"L'indice '{index_name}' esiste già. Saltando la creazione.")
        else:
             print(f"ERRORE durante la creazione dell'indice: {e}")

    """Formats the query and retrieved chunks into a structured prompt for the LLM."""

    context = "\n- ".join(context_texts)

    base_prompt = f"""
        Answer the following question based ONLY on the provided context.
        If you cannot answer from the context, state "I don't have enough information."

        Context:
        - {context}

        Question: {query}

        Answer:
    """
    return base_prompt

def main():
    # Load environment variables and initialize OpenAI client
    load_dotenv()
    client_AI = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

    # Define PDF path
    pdf_path = Path("One-time_review.pdf")

    # Extract chapter texts from PDF
    chapter_texts = open_and_read_pdf_by_outline(pdf_path=pdf_path)
    chunked_text = sum([chunk_text_with_padding(text, 500, 1) for text in tqdm(chapter_texts, desc="Chunking texts")], [])
    print("Total extracted chuncks: ", len(chunked_text))

    # Connect to MongoDB
    client_mongo, db = connect_to_mongo(uri=os.getenv("MONGO_URI"), db_name="Day-to-dAI")
    
    # Build or get the collection for chunks
    collection = build_collection_chunks(db, chunked_text)

    # Generate and store embeddings in MongoDB
    embed_texts(collection)

    # Build vector search index
    build_vector_search_index(collection)
    

if __name__ == "__main__":
    main()