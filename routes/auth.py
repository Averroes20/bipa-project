from fastapi import APIRouter, HTTPException
from app.schemas.user import UserRegister, UserLogin, TokenResponse
from app.services.auth import create_user, authenticate_user
from app.core.security import create_access_token

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/register")
def register(data: UserRegister):
    user = create_user(data.email, data.password)

    if not user:
        raise HTTPException(400, "Email already exists")

    return {"message": "User created"}


@router.post("/login", response_model=TokenResponse)
def login(data: UserLogin):
    user = authenticate_user(data.email, data.password)

    if not user:
        raise HTTPException(401, "Invalid credentials")

    token = create_access_token({"user_id": str(user.id)})

    return {"access_token": token}