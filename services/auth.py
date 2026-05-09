from app.models.user import User
from app.services.db import SessionLocal
from app.core.security import hash_password, verify_password

def create_user(email: str, password: str):
    db = SessionLocal()

    existing = db.query(User).filter(User.email == email).first()
    if existing:
        return None

    user = User(
        email=email,
        password=hash_password(password)
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


def authenticate_user(email: str, password: str):
    db = SessionLocal()

    user = db.query(User).filter(User.email == email).first()
    if not user:
        return None

    if not verify_password(password, user.password):
        return None

    return user