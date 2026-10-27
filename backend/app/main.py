from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
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

@app.exception_handler(RequestValidationError)
async def request_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    if request.url.path == "/api/reports/pdf":
        return JSONResponse(status_code=422, content={"detail": "Invalid report request."})
    return await request_validation_exception_handler(request, exc)

app.include_router(router)
