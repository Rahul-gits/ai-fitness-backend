from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# IMPORTANT: import settings
from app.core.config import settings

# ---------------------------
# Async database setup
# ---------------------------
ASYNC_DATABASE_URL = settings.get_database_url(is_async=True)

async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False,   # disable heavy logging
    future=True,
    pool_pre_ping=True
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

# ---------------------------
# Sync database setup
# ---------------------------
SYNC_DATABASE_URL = settings.get_database_url(is_async=False)

if SYNC_DATABASE_URL.startswith("sqlite"):
    sync_engine = create_engine(
        SYNC_DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False
    )
else:
    sync_engine = create_engine(
        SYNC_DATABASE_URL,
        echo=False
    )

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=sync_engine
)

# Base model
Base = declarative_base()

# ---------------------------
# Dependency for FastAPI
# ---------------------------
async def get_async_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()