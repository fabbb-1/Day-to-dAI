from dotenv import load_dotenv
import os
from openai import OpenAI
import requests
from pathlib import Path
import fitz # PyMuPDF
from tqdm.auto import tqdm 
import spacy
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import numpy as np
from pymongo import UpdateOne, ASCENDING
from pymongo.operations import SearchIndexModel
from pymongo.collection import Collection
from typing import List
import time

# Load spaCy's English model for sentence splitting
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

# Constants
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
BULK_WRITE_SIZE = 100 
EMBEDDING_DIM = 768 # Dimensione dell'embedding per il modello SentenceTransformer

def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip()
    cleaned_text = cleaned_text.replace("  ", " ")
    return cleaned_text

def open_and_read_pdf_by_outline(pdf_path: str) -> list[str]:
    """
    Opens a PDF file, reads its text content based on the outline (chapters).
    If no outline is found, the entire document is read as a single chunk.
    """
    doc = fitz.open(pdf_path)
    doc_list = []

    outline = doc.get_toc()

    if not outline:
        # Fallback: Read the entire PDF as one chunk if no outline is found
        full_text = ""
        for page_num in range(len(doc)):
            full_text += doc[page_num].get_text()
        
        # Aggiungi un'intestazione per il chunk completo per aiutare l'AI
        full_text = "Full Document Text:\n\n" + full_text
        doc_list.append(text_formatter(full_text))
        print("AVVISO: Nessun indice trovato. Letto l'intero documento come un unico chunk.")
        return doc_list

    # Logica originale se l'outline ESISTE:
    for current_index, (lev, title, page_n) in enumerate(outline):
        start_page_number = page_n - 1

        # Uso la logica più robusta per determinare la pagina finale basata sull'indice successivo
        next_page_n = outline[current_index+1][2] if current_index+1 < len(outline) else len(doc) + 1
        end_page_number = next_page_n - 2 if next_page_n > 1 else len(doc) - 1

        chapter_text = ""
        for page_num in range(start_page_number, end_page_number + 1):
            if page_num < len(doc):
                 chapter_text += doc[page_num].get_text()

        doc_list.append(text_formatter(chapter_text))
    return doc_list

def chunk_text_with_padding(text: str, chunk_size: int, padding: int) -> list[str]:
    """
    Splits text into chunks of approximate size, preserving sentences and adding padding.
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
        raise ConnectionError(f"Failed to connect to MongoDB: {e}")
        
    db = client_mongo[db_name]
    return client_mongo, db

def build_collection_chunks(db, chunked_text):
    """Resetta la collection 'employee_reviews' e inserisce i dati."""
    COLLECTION_NAME = 'employee_reviews'
    
    if COLLECTION_NAME in db.list_collection_names():
        print(f"La collezione '{COLLECTION_NAME}' esiste. Eliminazione per inserimento pulito...")
        db[COLLECTION_NAME].drop()
        
    collection = db[COLLECTION_NAME]
    
    collection.insert_many(chunked_text)
    assert collection.count_documents({}) == len(chunked_text)
    print(f"Inserted {len(chunked_text)} chunks into MongoDB.")
    
    collection.create_index([("chunk_id", ASCENDING)])
    
    return collection

def embed_texts(collection):
    """Genera gli embedding e li salva in batch in MongoDB."""
    
    device = "cuda" if "cuda" in os.getenv("EMBEDDING_DEVICE", "cpu").lower() and np.any([1]) else "cpu"
    print(f"Loading SentenceTransformer on device: {device}")
    
    embedding_model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL_NAME,
                                      device=device)

    documents_cursor = collection.find({"embedding": {"$exists": False}}) 
    total_docs_to_update = collection.count_documents({"embedding": {"$exists": False}})
    
    print(f"Found {total_docs_to_update} documents without embeddings to process.")

    bulk_updates = []
    
    for chunk in tqdm(documents_cursor,
                    total=total_docs_to_update,
                    desc='Generating and Updating Embeddings'):
        
        embedding = embedding_model.encode(chunk['text'], convert_to_numpy=True) 

        bulk_updates.append(
            UpdateOne(
                {"_id": chunk['_id']}, 
                {"$set": {"embedding": embedding.tolist()}} 
            )
        )

        if len(bulk_updates) >= BULK_WRITE_SIZE:
            collection.bulk_write(bulk_updates)
            bulk_updates = [] 

    if bulk_updates:
        collection.bulk_write(bulk_updates)

    updated_count = collection.count_documents({"embedding": {"$exists": True}})
    print(f"Finished generating and updating embeddings. Total documents with embeddings: {updated_count}.")
    
    if updated_count != collection.count_documents({}):
        print("ATTENZIONE: Alcuni documenti potrebbero non aver ricevuto l'embedding. Controlla gli errori.")

def build_vector_search_index(collection):
    """Crea l'indice di ricerca vettoriale per la collection e attende che sia pronto."""
    
    index_name="vector_index1"
    embedding_dim = EMBEDDING_DIM 
    embeddings_path = 'embedding' 

    search_index_model = SearchIndexModel(
        definition = {
            "fields": [
            {
                "type": "vector",
                "numDimensions": embedding_dim,
                "path": embeddings_path,
                "similarity": "cosine"
            }
            ]
        },
        name = index_name,
        type = "vectorSearch"
    )

    print(f"Tentativo di creare l'indice Atlas Vector Search '{index_name}'...")
    try:
        collection.create_search_index(model=search_index_model)
        print(f"Indice '{index_name}' creato con successo.")
    except Exception as e:
        # 1. Gestione Indice Esistente (OK)
        if "Index with name" in str(e):
             print(f"L'indice '{index_name}' esiste già. Saltando la creazione.")
        # 2. Gestione Limite FTS Raggiunto (CORREZIONE - Permetti di proseguire)
        elif "The maximum number of FTS indexes has been reached" in str(e):
             print("AVVISO: Limite massimo di indici Atlas raggiunto. Assumiamo che 'vector_index1' sia già attivo e proseguiamo.")
        else:
             raise RuntimeError(f"ERRORE durante la creazione dell'indice: {e}")

    # Logica di Attesa (Polling)
    print("Polling per verificare se l'indice è pronto. Potrebbe richiedere tempo.")
    
    def is_index_ready(index):
        return index.get("queryable") is True and index.get("status") == "READY"

    max_attempts = 12 
    attempts = 0

    while attempts < max_attempts:
        try:
            indices = list(collection.list_search_indexes(name=index_name))
            
            if len(indices) and is_index_ready(indices[0]):
                print(f"Indice '{index_name}' è pronto per la query. Procedo.")
                return 
            
            time.sleep(5)
            attempts += 1
            print(f"Attesa... Tentativo {attempts}/{max_attempts}")

        except Exception as e:
            print(f"Errore di polling temporaneo: {e}. Continuo ad attendere.")
            time.sleep(5)
            attempts += 1
            
    raise TimeoutError(f"Timeout: L'indice '{index_name}' non è diventato queryable dopo {max_attempts * 5} secondi. Controllare lo stato su Atlas.")
    """Crea l'indice di ricerca vettoriale per la collection e attende che sia pronto."""
    
    index_name="vector_index1"
    embedding_dim = EMBEDDING_DIM 
    embeddings_path = 'embedding' 

    # 1. Definizione dell'indice usando SearchIndexModel (come richiesto)
    search_index_model = SearchIndexModel(
        definition = {
            "fields": [
            {
                "type": "vector",
                "numDimensions": embedding_dim,
                "path": embeddings_path,
                "similarity": "cosine"
            }
            ]
        },
        name = index_name,
        type = "vectorSearch"
    )

    print(f"Tentativo di creare l'indice Atlas Vector Search '{index_name}'...")
    try:
        collection.create_search_index(model=search_index_model)
        print(f"Indice '{index_name}' creato con successo.")
    except Exception as e:
        if "Index with name" in str(e):
             print(f"L'indice '{index_name}' esiste già. Saltando la creazione.")
        else:
             # Se la creazione fallisce per altri motivi, lo script si ferma qui
             raise RuntimeError(f"ERRORE durante la creazione dell'indice: {e}")

    # 2. Attesa (Polling) che l'indice diventi queryable (CORREZIONE CRITICA)
    print("Polling per verificare se l'indice è pronto. Potrebbe richiedere fino a un minuto.")
    
    # La funzione predicate verifica se lo stato dell'indice è 'queryable'
    def is_index_ready(index):
        return index.get("queryable") is True and index.get("status") == "READY"

    max_attempts = 12 
    attempts = 0

    while attempts < max_attempts:
        try:
            # list_search_indexes restituisce un cursore, list() lo materializza
            indices = list(collection.list_search_indexes(name=index_name))
            
            if len(indices) and is_index_ready(indices[0]):
                print(f"Indice '{index_name}' è pronto per la query. Procedo.")
                return # Esci dalla funzione
            
            # Se non è pronto, aspetta e riprova
            time.sleep(5)
            attempts += 1
            print(f"Attesa... Tentativo {attempts}/{max_attempts}")

        except Exception as e:
            # Gestione di eventuali timeout o errori temporanei durante il polling
            print(f"Errore di polling temporaneo: {e}. Continuo ad attendere.")
            time.sleep(5)
            attempts += 1
            
    # Se il loop finisce senza successo
    raise TimeoutError(f"Timeout: L'indice '{index_name}' non è diventato queryable dopo {max_attempts * 5} secondi.")

def retrieve_context(collection: Collection, query: str, top_k: int = 5) -> List[str]:
    """
    Genera l'embedding della query e lo usa per cercare i chunk più rilevanti 
    nel database MongoDB tramite Atlas Vector Search.
    """
    # 1. Inizializzazione del Modello di Embedding (su CPU per non usare CUDA)
    embedding_model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL_NAME, device="cpu")
    
    # 2. Embedding della Query
    query_embedding = embedding_model.encode(query, convert_to_numpy=True).tolist()

    # 3. Pipeline di Aggregazione per Vector Search
    vector_search_pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index1",  # Nome dell'indice
                "queryVector": query_embedding,
                "path": "embedding",       # Campo del vettore
                "numCandidates": 100,      
                "limit": top_k             
            }
        },
        {"$project": {"text": 1, "_id": 0, "score": {"$meta": "vectorSearchScore"}}}
    ]

    # 4. Esecuzione della Ricerca
    context_results = list(collection.aggregate(vector_search_pipeline))
    context_texts = [result['text'] for result in context_results]
    
    print(f"Recuperati {len(context_texts)} chunk di contesto.")
    return context_texts

def create_rag_prompt(query: str, context_texts: List[str]) -> str:
    """
    Formatta la query e i chunk recuperati in un prompt strutturato per l'LLM.
    """
    
    context = "\n- ".join(context_texts)

    base_prompt = f"""
        Sei un assistente specializzato in risorse umane e benessere aziendale.
        Rispondi alla domanda seguente basandoti ESCLUSIVAMENTE sul contesto fornito (recuperato dalle recensioni dei dipendenti).
        Se non è possibile rispondere dalla sezione 'Contesto', rispondi "Non ho informazioni sufficienti basate sulle recensioni del team."

        Contesto:
        ---
        - {context}
        ---

        Domanda: {query}

        Risposta:
    """
    return base_prompt

def generate_answer(client_AI: OpenAI, prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Effettua la chiamata all'API di OpenAI con il prompt RAG, forzando la risposta in inglese.
    """
    
    # Messaggio di sistema modificato per richiedere la risposta in inglese
    system_message = (
        "You are a specialized corporate assistant focused on mental health and productivity "
        "within the IT consulting domain. Your objective is to provide concise, professional, "
        "and data-backed answers. **ALWAYS answer in English.**"
    )
    
    try:
        response = client_AI.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message}, # Uso del messaggio di sistema modificato
                {"role": "user", "content": prompt}
            ],
            temperature=0.0  
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"ERRORE during OpenAI API call: {e}" # Ho modificato anche l'errore in inglese per coerenza

def main():
    # 1. Setup
    client_mongo = None 
    load_dotenv()
    
    openai_api_key = os.getenv("OPEN_API_KEY")
    # Nota: Rimosso il controllo di esistenza del file prompt.txt qui per semplicità, 
    # ma la lettura deve avvenire nel blocco try.
    
    if not openai_api_key:
        print("AVVISO: OPEN_API_KEY non trovata. Le funzioni di generazione (RAG) falliranno.")
    client_AI = OpenAI(api_key=openai_api_key)

    # 2. PDF Processing
    pdf_path = Path("5workers50tasks.pdf") # Nome del file di input
    if not pdf_path.exists():
        print(f"ERRORE: File PDF non trovato al percorso: {pdf_path}. Assicurati di averlo in locale.")
        return

    # 3. MongoDB Connection and Insertion (Blocco Try/Except per la Robustezza)
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("MONGO_URI non trovata. Assicurati che sia nel file .env.")

    try:
        # Estrazione e Chunking
        chapter_texts = open_and_read_pdf_by_outline(pdf_path=pdf_path)
        
        if not chapter_texts:
            print("ERRORE: Il file PDF non ha restituito alcun testo. Interruzione.")
            return

        chunked_text = sum([chunk_text_with_padding(text, 500, 1) for text in tqdm(chapter_texts, desc="Chunking texts")], [])
        print(f"Total extracted chunks: {len(chunked_text)}")
        
        for chunk in chunked_text:
            chunk['doc_source'] = "review.pdf" 

        # Connessione al DB
        client_mongo, db = connect_to_mongo(uri=mongo_uri, db_name="Day-to-dAI")
        
        # Inserimento dati
        collection = build_collection_chunks(db, chunked_text)

        # Generazione Embedding
        embed_texts(collection)

        # Creazione Indice Vettoriale
        build_vector_search_index(collection)
        
        # 4. FASE DI QUERY RAG & REPORT GENERATION
        print("\n--- INIZIO FASE RAG: INTERROGAZIONE DEL DATABASE ---")
        
        # Legge l'intero prompt complesso dal file
        with open("prompt.txt", "r") as f:
            full_prompt_text = f.read()

        # Il prompt RAG necessita solo della query finale per il retrieval, 
        # ma l'LLM (generate_answer) riceverà l'intero full_prompt_text (il prompt complesso)
        # Qui usiamo una query semplice per l'embedding, dato che il prompt complesso è troppo lungo.
        file_input=open("prompt.txt","r")
        sample_query_for_retrieval = file_input.read()
        
        # Retrieval (Ricerca dei chunk rilevanti)
        context_texts = retrieve_context(collection, sample_query_for_retrieval, top_k=5)
        
        # Costruisce il prompt RAG combinando il prompt complesso e il contesto recuperato
        rag_prompt = create_rag_prompt(full_prompt_text, context_texts)
        print(f"Prompt RAG costruito (Contesto: {len(context_texts)} chunk)")

        # Generazione (Chiamata all'LLM)
        answer = generate_answer(client_AI, rag_prompt)
        
        file_output=open("output.txt","w")
        print("\n--- RISPOSTA GENERATA DALL'AI ---")
        print(f"Output in output.txt")
        file_output.write(answer)
        print("-----------------------------------")
        
    except Exception as e:
        # Gestione degli errori
        print(f"Un errore critico è avvenuto durante l'esecuzione: {e}")
        
    finally:
        # Blocco finally per chiudere la connessione
        if client_mongo:
            client_mongo.close()
            print("Connessione MongoDB chiusa.")

if __name__ == "__main__":
    main()