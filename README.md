# Career Advisory API (LLM Agent, ChromaDB, Ollama RAG)

A modern FastAPI backend for career advisory, using Retrieval-Augmented Generation (RAG) with Ollama LLMs (e.g., gemma3:4b), ChromaDB vector storage, web search, and persistent user/session memory. Designed to help students select subject streams, degree programs, and colleges, providing actionable, grounded, and concise advice.

---

## Features

- **FastAPI**: High-performance asynchronous REST API.
- **Ollama LLM Integration**: Run local LLMs (default: `gemma3:4b`) for generation and memory extraction.
- **ChromaDB Vector Store**: Stores and retrieves relevant documents for grounding answers (RAG).
- **Sentence Transformers**: All-MiniLM-L6-v2 embeddings for document and query vectors.
- **MongoDB**: User, session, and document metadata persistence.
- **Web Search**: Augment answers with fresh info via Google search snippets.
- **Session Context Window**: Supports long conversations (up to 128k characters per session).
- **In-Memory Fallback**: Automatically uses in-memory vector search if ChromaDB is unavailable.
- **User Memory Extraction**: Extracts user facts (age, interests, education, etc.) to personalize advice.

---

## API Endpoints

### User Management

- `POST /user`  
  Create or update a user.

- `GET /user/{user_id}`  
  Retrieve user details.

- `DELETE /user/{user_id}`  
  Delete a user and all sessions.

### Session Management

- `POST /session`  
  Create a new chat session.

- `GET /session/{session_id}`  
  Retrieve session details.

- `DELETE /session/{session_id}`  
  Delete a session.

- `POST /session/{session_id}/message`  
  Send a user message, run RAG, and get assistant response.

### Document Ingestion

- `POST /docs`  
  Ingest a new document (title, content, optional source) for retrieval.

### Health Check

- `GET /health`  
  Check the status of Ollama, ChromaDB, and the API.

---

## How It Works

1. **User starts a session** and sends a message/question.
2. **Relevant documents** are retrieved from ChromaDB (or in-memory fallback) using vector search.
3. **Web search** results are fetched for up-to-date information.
4. **Session and user memory** are loaded and included in the prompt.
5. **Prompt is constructed** and sent to Ollama LLM for response.
6. **Assistant response** is returned, logged in the session, and user facts are extracted for memory.

---

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com/) running locally (or update `OLLAMA_URL` for remote)
- MongoDB (local or remote)
- ChromaDB (persisted or in-memory)
- Google Search Python library (`googlesearch-python`)
- Sentence Transformers

### Install Dependencies

```sh
pip install fastapi uvicorn motor python-dotenv requests beautifulsoup4 googlesearch-python sentence-transformers chromadb
```

*You may need additional dependencies for ChromaDB and Ollama LLM model.*

---

## Configuration

Environment variables (see `.env` or set as system variables):

- `MONGO_URI` — default: `mongodb://localhost:27017`
- `DB_NAME` — default: `career_advisor`
- `OLLAMA_URL` — default: `http://localhost:11434`
- `OLLAMA_MODEL` — default: `gemma3:4b`
- `CHROMA_PERSIST_DIR` — default: `./chromadb_persist`
- `EMBEDDING_MODEL` — default: `sentence-transformers/all-MiniLM-L6-v2`
- `SESSION_CONTEXT_MAX_CHARS` — default: `128000`

---

## Running the API

Start Ollama and load your preferred LLM (e.g., `gemma3:4b`):

```sh
ollama run gemma3:4b
```

Start the API server:

```sh
uvicorn app.main:app --reload
```

---

## Example Usage

**Create user:**

```bash
curl -X POST http://localhost:8000/user -H "Content-Type: application/json" -d '{"user_id": "user123", "name": "Alice"}'
```

**Create session:**

```bash
curl -X POST http://localhost:8000/session -H "Content-Type: application/json" -d '{"user_id": "user123"}'
```

**Send message:**

```bash
curl -X POST http://localhost:8000/session/<session_id>/message -H "Content-Type: application/json" -d '{"user_id": "user123", "text": "What are good government colleges for engineering in Delhi?"}'
```

---

## Project Structure

```
app/
  main.py         # FastAPI app with all endpoints and logic
.env              # (optional) Environment variables
README.md
```

---

## License

MIT License

---

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [Ollama](https://ollama.com/)
- [ChromaDB](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Google Search Python](https://pypi.org/project/googlesearch-python/)

---

## Author

[PUSHPAK-JAISWAL](https://github.com/PUSHPAK-JAISWAL)
