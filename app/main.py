from fastapi import FastAPI
from app.routes import analyze

app = FastAPI(title="BIPA Intonation Analysis API")

app.include_router(analyze.router)