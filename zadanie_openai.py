import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import ollama
import asyncio
import json
import logging
from datetime import datetime
import os
load_dotenv()
# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicjalizacja FastAPI
app = FastAPI(
    title="Ollama AI Service",
    description="Serwis AI wykorzystujący modele Ollama",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncOpenAI()


# === MODELE DANYCH ===

class SummarizeRequestOpenAI(BaseModel):
    text: str

class SummarizeResponseOpenAI(BaseModel):
    summary: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="Wiadomość użytkownika")
    model: str = Field(default="llama3.2", description="Model Ollama")
    system_prompt: Optional[str] = Field(None, description="Prompt systemowy")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)


class ChatResponse(BaseModel):
    response: str
    model: str
    tokens_used: int
    processing_time: float
    timestamp: datetime


class EmbeddingRequest(BaseModel):
    text: str = Field(..., description="Tekst do embeddingu")
    model: str = Field(default="nomic-embed-text", description="Model embedding")


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    model: str
    dimensions: int


class ModelInfo(BaseModel):
    name: str
    size: str
    digest: str
    modified_at: datetime


class SummarizeRequest(BaseModel):
    text: str = Field(..., description="Tekst do podsumowania")
    model: str = Field(default="llama3.2")
    max_length: int = Field(default=200, description="Maksymalna długość podsumowania")


# === SERWISY ===

class OllamaService:
    """Serwis do zarządzania modelami Ollama"""

    def __init__(self):
        self.client = ollama.Client()
        self.available_models = set()
        self.refresh_models()

    def refresh_models(self):
        """Odświeża listę dostępnych modeli"""
        try:
            models = self.client.list()
            self.available_models = {model['name'] for model in models['models']}
            logger.info(f"Dostępne modele: {self.available_models}")
        except Exception as e:
            logger.error(f"Błąd pobierania modeli: {e}")
            self.available_models = set()

    def ensure_model(self, model_name: str):
        """Zapewnia, że model jest dostępny (pobiera jeśli nie ma)"""
        if model_name not in self.available_models:
            logger.info(f"Pobieranie modelu: {model_name}")
            try:
                self.client.pull(model_name)
                self.refresh_models()
                return True
            except Exception as e:
                logger.error(f"Błąd pobierania modelu {model_name}: {e}")
                raise HTTPException(status_code=400, detail=f"Nie można pobrać modelu: {model_name}")
        return True

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Generuje odpowiedź czatu"""
        start_time = datetime.now()

        self.ensure_model(request.model)

        try:
            # Przygotowanie wiadomości
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.message})

            # Wywołanie Ollama
            response = self.client.chat(
                model=request.model,
                messages=messages,
                options={
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                }
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            return ChatResponse(
                response=response['message']['content'],
                model=request.model,
                tokens_used=response.get('eval_count', 0),
                processing_time=processing_time,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Błąd generowania odpowiedzi: {e}")
            raise HTTPException(status_code=500, detail=f"Błąd AI: {str(e)}")

    async def generate_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generuje embedding dla tekstu"""
        self.ensure_model(request.model)

        try:
            response = self.client.embeddings(
                model=request.model,
                prompt=request.text
            )

            embedding = response['embedding']

            return EmbeddingResponse(
                embedding=embedding,
                model=request.model,
                dimensions=len(embedding)
            )

        except Exception as e:
            logger.error(f"Błąd generowania embeddingu: {e}")
            raise HTTPException(status_code=500, detail=f"Błąd embeddingu: {str(e)}")

    async def summarize_text(self, request: SummarizeRequest) -> ChatResponse:
        """Podsumowuje tekst"""
        start_time = datetime.now()

        self.ensure_model(request.model)

        system_prompt = f"""Jesteś ekspertem w tworzeniu zwięzłych podsumowań. 
        Stwórz podsumowanie poniższego tekstu w maksymalnie {request.max_length} słowach.
        Zachowaj najważniejsze informacje i główne punkty."""

        try:
            response = self.client.chat(
                model=request.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Podsumuj ten tekst:\n\n{request.text}"}
                ]
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            return ChatResponse(
                response=response['message']['content'],
                model=request.model,
                tokens_used=response.get('eval_count', 0),
                processing_time=processing_time,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Błąd podsumowywania: {e}")
            raise HTTPException(status_code=500, detail=f"Błąd podsumowania: {str(e)}")


# Inicjalizacja serwisu
ollama_service = OllamaService()


# === ENDPOINTY ===

@app.get("/")
async def root():
    """Endpoint główny"""
    return {
        "message": "Ollama AI Service",
        "version": "1.0.0",
        "models": list(ollama_service.available_models)
    }


@app.get("/models", response_model=List[ModelInfo])
async def get_models():
    """Pobiera listę dostępnych modeli"""
    try:
        models_data = ollama_service.client.list()
        models = []

        for model in models_data['models']:
            models.append(ModelInfo(
                name=model['name'],
                size=model.get('size', 'Unknown'),
                digest=model.get('digest', 'Unknown'),
                modified_at=datetime.fromisoformat(model['modified_at'].replace('Z', '+00:00'))
            ))

        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd pobierania modeli: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """Endpoint do czatu z AI"""
    return await ollama_service.chat(request)


@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    """Tworzy embedding dla tekstu"""
    return await ollama_service.generate_embedding(request)


@app.post("/summarize", response_model=ChatResponse)
async def summarize_text(request: SummarizeRequest):
    """Podsumowuje podany tekst"""
    return await ollama_service.summarize_text(request)


@app.post("/pull-model")
async def pull_model(model_name: str, background_tasks: BackgroundTasks):
    """Pobiera nowy model w tle"""

    def download_model():
        try:
            ollama_service.client.pull(model_name)
            ollama_service.refresh_models()
            logger.info(f"Model {model_name} pobrany pomyślnie")
        except Exception as e:
            logger.error(f"Błąd pobierania modelu {model_name}: {e}")

    background_tasks.add_task(download_model)
    return {"message": f"Pobieranie modelu {model_name} rozpoczęte w tle"}


@app.post("/analyze-document")
async def analyze_document(
        file: UploadFile = File(...),
        model: str = "llama3.2",
        analysis_type: str = "summary"
):
    """Analizuje przesłany dokument"""
    if not file.filename.endswith(('.txt', '.md')):
        raise HTTPException(status_code=400, detail="Obsługiwane formaty: .txt, .md")

    try:
        content = await file.read()
        text = content.decode('utf-8')

        if analysis_type == "summary":
            request = SummarizeRequest(text=text, model=model)
            return await ollama_service.summarize_text(request)

        elif analysis_type == "keywords":
            chat_request = ChatRequest(
                message=f"Wyodrębnij 10 najważniejszych słów kluczowych z tekstu:\n\n{text}",
                model=model,
                system_prompt="Jesteś ekspertem w analizie tekstu. Wyodrębnij słowa kluczowe w formie listy."
            )
            return await ollama_service.chat(chat_request)

        elif analysis_type == "sentiment":
            chat_request = ChatRequest(
                message=f"Przeanalizuj sentyment tego tekstu (pozytywny/negatywny/neutralny):\n\n{text}",
                model=model,
                system_prompt="Jesteś ekspertem w analizie sentymentu. Określ wydźwięk emocjonalny tekstu."
            )
            return await ollama_service.chat(chat_request)

        else:
            raise HTTPException(status_code=400, detail="Dostępne typy: summary, keywords, sentiment")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd analizy dokumentu: {str(e)}")


@app.get("/health")
async def health_check():
    """Sprawdza status zdrowia serwisu"""
    try:
        # Test połączenia z Ollama
        models = ollama_service.client.list()
        return {
            "status": "healthy",
            "ollama_connected": True,
            "models_count": len(models['models']),
            "timestamp": datetime.now()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "ollama_connected": False,
            "error": str(e),
            "timestamp": datetime.now()
        }


# === STARTUP/SHUTDOWN ===

@app.on_event("startup")
async def startup_event():
    """Inicjalizacja przy starcie"""
    logger.info("🚀 Ollama AI Service startuje...")
    ollama_service.refresh_models()
    logger.info(f"📊 Załadowano {len(ollama_service.available_models)} modeli")


@app.on_event("shutdown")
async def shutdown_event():
    """Czyszczenie przy wyłączaniu"""
    logger.info("🔄 Ollama AI Service zatrzymywany...")


@app.post("/openai/summarize", response_model=SummarizeResponseOpenAI)
async def summarize(request: SummarizeRequestOpenAI):
    text = request.text.strip()
    if not text or len(text) < 20:
        raise HTTPException(status_code=400, detail="Tekst do podsumowania jest za krótki.")

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",  # możesz użyć "gpt-4-turbo" lub "gpt-3.5-turbo" (tańszy)
            messages=[
                {"role": "system", "content": "Jesteś asystentem podsumowującym tekst."},
                {"role": "user", "content": f"Streść ten tekst w kilku zdaniach: {text}"}
            ],
            max_tokens=150,
            temperature=0.6,
        )
        summary = response.choices[0].message.content.strip()
        return SummarizeResponseOpenAI(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd serwera: {str(e)}")

