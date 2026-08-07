from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.database import engine, Base
import app.models.learning_models
import app.models.dataset_models
from app.routes import analyze, analytics, auth, tasks
from app.routes import dataset as dataset_router
from app.core.exceptions import global_exception_handler, bipa_exception_handler, BIPAException
from app.core.logger import logger

# Ensure ORM tables are created on startup if missing
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        Base.metadata.create_all(bind=engine)
        
        # Create HNSW index for pgvector
        with engine.connect() as conn:
            from sqlalchemy import text
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            try:
                conn.execute(text("ALTER TABLE dataset_feature ALTER COLUMN embedding_vector TYPE vector USING embedding_vector::vector"))
            except Exception:
                pass
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_dataset_feature ON dataset_feature USING hnsw (embedding_vector vector_cosine_ops)"))
            conn.commit()
            
        logger.info("Database startup initialization completed.")
    except Exception as e:
        logger.error(f"Database startup initialization notice: {e}")
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(analyze.router)
app.include_router(analytics.router)
app.include_router(auth.router)
app.include_router(tasks.router)
app.include_router(dataset_router.router)

# Register Exception Handlers
app.add_exception_handler(Exception, global_exception_handler)
app.add_exception_handler(BIPAException, bipa_exception_handler)

@app.get("/")
def root():
    return {"message": "BIPA Intonation Analysis API is online"}