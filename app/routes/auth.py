from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.user import UserRegister, UserLogin, TokenResponse
from app.services.auth import create_user, authenticate_user
from app.core.security import create_access_token

router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/register")
def register(data: UserRegister, db: Session = Depends(get_db)):
    user = create_user(db, data.email, data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already exists"
        )
    return {"message": "User registered successfully"}

@router.post("/login", response_model=TokenResponse)
def login(data: UserLogin, db: Session = Depends(get_db)):
    user = authenticate_user(db, data.email, data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    token = create_access_token({"user_id": str(user.id)})
    return {"access_token": token}