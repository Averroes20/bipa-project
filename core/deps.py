from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer
from app.core.security import decode_token

security = HTTPBearer()

def get_current_user(token=Depends(security)):
    payload = decode_token(token.credentials)

    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")

    return str(payload["user_id"])