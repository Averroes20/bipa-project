import os
from dotenv import load_dotenv

load_dotenv()

try:
    from pydantic_settings import BaseSettings
    class Settings(BaseSettings):
        PROJECT_NAME: str = "BIPA Audio Analysis API"
        DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
        SECRET_KEY: str = os.getenv("SECRET_KEY", "U2FsdGVkX1/0u6UyU1uRiBf/BC/DEhXj+pGhm4YdDJ/z61u5/iZ7LcqFD6VsY3Ew")
        ALGORITHM: str = "HS256"
        ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

        class Config:
            env_file = ".env"
            extra = "ignore"
    settings = Settings()

except (ImportError, ModuleNotFoundError):
    class Settings:
        PROJECT_NAME: str = "BIPA Audio Analysis API"
        DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
        SECRET_KEY: str = os.getenv("SECRET_KEY", "U2FsdGVkX1/0u6UyU1uRiBf/BC/DEhXj+pGhm4YdDJ/z61u5/iZ7LcqFD6VsY3Ew")
        ALGORITHM: str = "HS256"
        ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    settings = Settings()
