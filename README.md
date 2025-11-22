# Day-to-dAI
Reply clash 2025

Install requirements: pip install PyMuPDF tqdm transformers sentence-transformers accelerate bitsandbytes huggingface-hub pymongo spacy openai python-dotenv

Day to dAI (D2D): AI-Powered Wellness Scheduler - Technical Documentation

1. Overview

Day to dAI (D2D) is an autonomous AI agent prototype designed to solve the "Stress Minimization Scheduling Problem." By leveraging Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs), D2D transforms unstructured company data (PDFs) into a highly optimized, day-to-day work schedule that prioritizes employee mental health and prevents burnout.

2. Architecture & Interaction Flow

The system operates as a sequential pipeline:

Ingestion: Reads a raw company PDF (review.pdf) containing worker profiles (skills, stress tolerance) and unallocated tasks.

Vector Memory: Chunks the text, generates semantic embeddings, and stores them in a vector database (MongoDB Atlas) to create a queryable Knowledge Base.

RAG Retrieval: When a schedule is requested, the system retrieves the specific tables of workers and tasks from the database using targeted semantic search.

AI Reasoning: An LLM (OpenAI GPT-3.5/4) receives the retrieved data along with a strict "Zero-Burnout" optimization prompt. It performs Chain-of-Thought reasoning to assign tasks, calculate cumulative stress scores (WSS), and insert mandatory recovery breaks (Micro-Breaks/UFT).

Output Generation: The system produces two artifacts:

A output.txt file containing the raw reasoning and schedule.

A professional scheduling_report.tex (LaTeX) file ready for PDF compilation.

How to Run the Prototype

Prerequisites:

Python 3.8+

MongoDB Atlas Account (Cluster deployed)

OpenAI API Key

Environment Setup:
Create a .env file in the root directory with the following keys:

OPEN_API_KEY="sk-..."
MONGO_URI="mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/?appName=Lab4"


Execution:
Run the main script from your terminal:

python main.py


Interaction:

The script will automatically parse the PDF, connect to the DB, and run the optimization.

Console Output: Watch the progress bars for chunking and embedding generation.

Final Report: Once finished, open the generated scheduling_report.tex (compile with pdflatex) or view output.txt to see the optimal schedule.

3. Open-Source Libraries & Models

This prototype relies on state-of-the-art open-source components:

PyMuPDF (fitz): For high-fidelity extraction of text and structural outlines from PDF documents.

Sentence-Transformers (all-mpnet-base-v2): A top-tier open-source embedding model (via Hugging Face) used to convert text into 768-dimensional vectors. It ensures high semantic accuracy in retrieval.

PyMongo: The official driver for interacting with MongoDB Atlas Vector Search.

SpaCy: Used for advanced natural language processing (NLP) tasks like sentence splitting and text cleaning.

TQDM: For progress visualization during data processing.

4. Custom Logic Implementation

While standard libraries handle the "heavy lifting," the core intelligence of D2D lies in the custom orchestration logic:

Robust PDF Parsing with Fallback: The open_and_read_pdf_by_outline function includes a custom fallback mechanism. It attempts to read by chapter (Outline) for structure but automatically switches to a full-document scan if no metadata is found, ensuring robustness across different PDF types.

Dynamic Vector Search Construction: We implemented a custom retrieval function (retrieve_context) that dynamically builds MongoDB aggregation pipelines. It uses specific keywords ("WorkerID", "TaskID") to force the retrieval of tabular data, overcoming generic search limitations.

Wait-and-Retry Indexing: The build_vector_search_index function includes a custom polling mechanism (while loop with time.sleep). It monitors the asynchronous creation of the vector index on Atlas, blocking execution until the index is "READY" to prevent retrieval failures during the first run.

LaTeX Report Generation: A dedicated module (generate_latex_report) programmatically constructs a LaTeX source file, injecting the AI's output into a pre-defined professional template, automating the creation of executive-level documentation.