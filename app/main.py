from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import IntegrityError
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import uvicorn
from .core.config import settings
from .core.redis import redis_service
from .core.middleware import RateLimitMiddleware
from .db.database import sync_engine, Base
from .api.v1.auth import router as auth_router
from .api.v1.workouts import router as workout_router
from .api.v1.users import router as profile_router
from .api.v1.plans import router as plan_router
from .api.v1.routines import router as routine_router
from .api.v1.social import router as social_router
from .api.v1.stats import router as stats_router
from .api.v1.dashboard import router as dashboard_router
from .api.v1.voice_commands import router as voice_router
from .api.v1.websockets import router as ws_router
from .api.v1.water import router as water_router
from .api.v1.chatbot import router as chatbot_router
from .api.v1.ai import router as ai_router

# Setup logging - WARNING level to reduce verbosity
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Suppress SQLAlchemy debug logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)

# Lifespan context manager (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await redis_service.connect()
    logger.info("Application startup complete")
    yield
    # Shutdown
    await redis_service.disconnect()
    logger.info("Application shutdown complete")

app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

# Mount static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Exception handler for database integrity errors
@app.exception_handler(IntegrityError)
async def integrity_exception_handler(request: Request, exc: IntegrityError):
    logger.error(f"Integrity error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": "Database integrity error. This usually means a unique constraint failed (e.g. username or email already exists)."},
    )

# General exception handler
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Please check server logs."},
    )

# Custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        loc = error.get("loc")
        msg = error.get("msg")
        field = loc[-1] if loc else "field"
        errors.append(f"{field}: {msg}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": ", ".join(errors)},
    )

# Create database tables
Base.metadata.create_all(bind=sync_engine)

@app.get("/api/v1/test")
async def test_endpoint():
    return {"status": "ok"}

# Include API routers
app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(workout_router, prefix=settings.API_V1_STR)
app.include_router(profile_router, prefix=settings.API_V1_STR)
app.include_router(plan_router, prefix=settings.API_V1_STR)
app.include_router(routine_router, prefix=settings.API_V1_STR)
app.include_router(social_router, prefix=settings.API_V1_STR)
app.include_router(stats_router, prefix=settings.API_V1_STR)
app.include_router(dashboard_router, prefix=settings.API_V1_STR)
app.include_router(voice_router, prefix=settings.API_V1_STR)
app.include_router(ws_router, prefix=settings.API_V1_STR)
app.include_router(water_router, prefix=settings.API_V1_STR)
app.include_router(chatbot_router, prefix=settings.API_V1_STR)
app.include_router(ai_router, prefix=settings.API_V1_STR)

# Add RateLimitMiddleware
app.add_middleware(RateLimitMiddleware, redis_service=redis_service, limit=100, window=60)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "AI Fitness Backend Running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)