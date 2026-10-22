from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api.routes import router
from app.config import get_settings
from app.services.errors import IngestionError

settings = get_settings()
app = FastAPI(title="Multilingual News Intelligence API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=settings.origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

@app.exception_handler(IngestionError)
async def ingestion_error(_: Request, exc: IngestionError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

app.include_router(router)
