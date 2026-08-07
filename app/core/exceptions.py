from fastapi import Request, status
from fastapi.responses import JSONResponse
from app.core.logger import logger

class BIPAException(Exception):
    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        self.message = message
        self.status_code = status_code

class AudioProcessingException(BIPAException):
    def __init__(self, message: str = "Audio processing failed"):
        super().__init__(message, status.HTTP_400_BAD_REQUEST)

class ModelInferenceException(BIPAException):
    def __init__(self, message: str = "Model inference failed"):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR)

async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error occurred", "error": str(exc)}
    )

async def bipa_exception_handler(request: Request, exc: BIPAException):
    logger.error(f"BIPA Error: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "error_code": exc.__class__.__name__}
    )
