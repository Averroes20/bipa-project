from fastapi import FastAPI
from app.routes import analyze
from app.routes import auth

app = FastAPI(title="BIPA Intonation Analysis API")

app.include_router(analyze.router)

app.include_router(auth.router)