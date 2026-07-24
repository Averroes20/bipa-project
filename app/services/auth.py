from sqlalchemy.orm import Session
from app.repositories.user_repository import UserRepository
from app.models.user import User

def create_user(db: Session, email: str, password: str) -> User | None:
    repo = UserRepository(db)
    return repo.create(email, password)

def authenticate_user(db: Session, email: str, password: str) -> User | None:
    repo = UserRepository(db)
    return repo.authenticate(email, password)