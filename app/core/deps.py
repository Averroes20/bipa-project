from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
from app.core.security import decode_token
from app.core.database import get_db

security = HTTPBearer()

def get_current_user(token=Depends(security)) -> str:
    payload = decode_token(token.credentials)

    if not payload or "user_id" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return str(payload["user_id"])