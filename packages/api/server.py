from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from packages.api.routes import router
from packages.api.config import settings

app = FastAPI(
    title="Risk Assessment API",
    version=settings.api_version,
    description="Multi-miner risk assessment and validation API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)