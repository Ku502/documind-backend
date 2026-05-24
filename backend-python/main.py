from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, shutil, tempfile
import google.generativeai as genai
import pypdf

app = FastAPI(title="DocuMind AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# In-memory document store
document_store = {}

class QuestionRequest(BaseModel):
    session_id: str
    question: str

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

@app.get("/")
def root():
    return {"status": "DocuMind AI is running"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        text = extract_text_from_pdf(tmp_path)
        chunks = chunk_text(text)
        session_id = file.filename.replace(" ", "_").replace(".pdf", "")
        document_store[session_id] = {
            "text": text,
            "chunks": chunks,
            "filename": file.filename
        }
        os.unlink(tmp_path)

        return {
            "message": "PDF uploaded and processed successfully!",
            "session_id": session_id,
            "chunks": len(chunks)
        }

    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_question(req: QuestionRequest):
    if req.session_id not in document_store:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a PDF first.")

    doc = document_store[req.session_id]
    
    # Use first 6000 chars as context (fits in Gemini context window)
    context = doc["text"][:6000]

    prompt = f"""You are DocuMind, an intelligent document assistant.
Use the following document content to answer the user's question accurately and concisely.
If the answer is not in the document, say "I couldn't find this in the uploaded document."

Document Content:
{context}

Question: {req.question}

Answer:"""

    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        answer = response.text

        return {
            "question": req.question,
            "answer": answer,
            "session_id": req.session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
