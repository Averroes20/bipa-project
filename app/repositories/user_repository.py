from typing import Optional
from sqlalchemy.orm import Session
from app.models.user import User
from app.core.security import hash_password, verify_password

class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_email(self, email: str) -> Optional[User]:
        return self.db.query(User).filter(User.email == email).first()

    def create(self, email: str, password: str) -> Optional[User]:
        if self.get_by_email(email):
            return None

        hashed_pwd = hash_password(password)
        user = User(email=email, password=hashed_pwd)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def authenticate(self, email: str, password: str) -> Optional[User]:
        user = self.get_by_email(email)
        if not user:
            return None
        if not verify_password(password, user.password):
            return None
        return user
