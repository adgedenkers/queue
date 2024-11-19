from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, status, Header
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.future import select
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
import base64
import aiofiles
import os
import json
import uuid
import magic
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
import aiofiles.os as async_os
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

# Load environment variables
load_dotenv()

# Configuration class using environment variables
class Settings:
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql+asyncpg://user:password@localhost:5432/queue_db"
    ).replace('postgresql://', 'postgresql+asyncpg://')
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    MAX_IMAGE_SIZE: int = int(os.getenv("MAX_IMAGE_SIZE", 10 * 1024 * 1024))
    ALLOWED_IMAGE_TYPES: set = {
        'image/jpeg', 'image/png', 'image/gif', 'image/webp'
    }

settings = Settings()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler(
    'api.log', 
    maxBytes=10485760,
    backupCount=5
)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Database setup
engine = create_async_engine(settings.DATABASE_URL, echo=True)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# Custom exceptions
class DatabaseError(Exception):
    pass

class FileUploadError(Exception):
    pass

class ValidationError(Exception):
    pass

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    auth_token = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class QueueDB(Base):
    __tablename__ = "queue"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    raw_text = Column(Text)
    status = Column(String, default="pending")
    active = Column(Boolean, default=True)
    options = Column(JSONB, default=dict)
    properties = Column(JSONB, nullable=False)
    created = Column(DateTime, default=datetime.utcnow)
    updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    images = relationship("QueueImage", back_populates="queue_item")
    user = relationship("User")

class QueueImage(Base):
    __tablename__ = "queue_images"
    
    id = Column(Integer, primary_key=True, index=True)
    queue_id = Column(Integer, ForeignKey("queue.id"))
    filename = Column(String)
    mime_type = Column(String)
    file_size = Column(Integer)
    created = Column(DateTime, default=datetime.utcnow)
    
    queue_item = relationship("QueueDB", back_populates="images")

# Pydantic Models
class ImageData(BaseModel):
    base64_data: str
    filename: str

class QueueCreate(BaseModel):
    user_id: int = Field(..., gt=0, description="User ID must be positive")
    raw_text: str = Field(..., min_length=1, max_length=10000)
    options: Dict[str, Any] = Field(default_factory=dict)
    properties: Dict[str, Any] = Field(...)
    images: Optional[List[ImageData]] = []

    @validator('properties')
    def validate_properties(cls, v):
        if not v or len(v) == 0:
            raise ValueError('properties must contain at least one key/value pair')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 1,
                "raw_text": "Example queue item text",
                "options": {"flag1": True, "setting2": "value"},
                "properties": {"required_key": "required_value"},
                "images": []
            }
        }

# Database dependency
async def get_db():
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

# Helper functions
async def validate_image(image: UploadFile) -> None:
    """Validate image file"""
    contents = await image.read()
    await image.seek(0)  # Reset file pointer
    
    if len(contents) > settings.MAX_IMAGE_SIZE:
        raise ValidationError(
            f"Image size exceeds maximum allowed size of {settings.MAX_IMAGE_SIZE/1024/1024}MB"
        )
    
    mime_type = magic.from_buffer(contents, mime=True)
    if mime_type not in settings.ALLOWED_IMAGE_TYPES:
        raise ValidationError(f"Invalid image type: {mime_type}")

async def save_upload_file(upload_file: UploadFile, destination: str) -> None:
    """Save uploaded file to destination"""
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        async with aiofiles.open(destination, 'wb') as out_file:
            content = await upload_file.read()
            await out_file.write(content)
    except Exception as e:
        raise FileUploadError(f"Error saving file: {str(e)}")

async def save_base64_image(base64_data: str, filename: str) -> str:
    """Save base64 image data to file system asynchronously"""
    try:
        ext = os.path.splitext(filename)[1]
        new_filename = f"{uuid.uuid4()}{ext}"
        file_path = os.path.join(settings.UPLOAD_DIR, new_filename)
        
        image_data = base64.b64decode(base64_data)
        
        if len(image_data) > settings.MAX_IMAGE_SIZE:
            raise ValidationError(
                f"Image size exceeds maximum allowed size of {settings.MAX_IMAGE_SIZE/1024/1024}MB"
            )
            
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(image_data)
            
        return new_filename
    except Exception as e:
        raise FileUploadError(f"Invalid image data: {str(e)}")

# Authentication dependency
async def verify_token(
    user_id: int,
    x_token: str = Header(..., description="User's authentication token"),
    db: AsyncSession = Depends(get_db)
):
    """Verify the user's token matches their ID"""
    try:
        result = await db.execute(
            select(User)
            .where(User.id == user_id)
            .where(User.auth_token == x_token)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return user
    except Exception as e:
        logger.error(f"Token verification error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication error"
        )

# FastAPI app
app = FastAPI()

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up application")
    async with engine.begin() as conn:
        # Drop specific tables in correct order
        await conn.execute(text("DROP TABLE IF EXISTS queue_images"))
        await conn.execute(text("DROP TABLE IF EXISTS queue"))
        await conn.execute(text("DROP TABLE IF EXISTS users"))
        
        # Create tables
        await conn.run_sync(Base.metadata.create_all)
    
    # Ensure upload directory exists
    if not os.path.exists(settings.UPLOAD_DIR):
        await async_os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application")
    try:
        if os.path.exists(settings.UPLOAD_DIR):
            for root, dirs, files in os.walk(settings.UPLOAD_DIR, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(settings.UPLOAD_DIR)
    except Exception as e:
        logger.error(f"Error cleaning up upload directory: {str(e)}")

# API Endpoints
@app.get("/healthcheck")
async def healthcheck(db: AsyncSession = Depends(get_db)):
    """Health check endpoint"""
    try:
        await db.execute(text("SELECT 1"))
        db_status = "the denkers.co api is fully operational"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        db_status = "disconnected"
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": db_status
    }

@app.post("/queue/", status_code=status.HTTP_201_CREATED)
async def create_queue_item(
    queue_data: QueueCreate,
    x_token: str = Header(..., description="User's authentication token"),
    db: AsyncSession = Depends(get_db)
):
    """Create queue item with JSON data"""
    try:
        # Verify token matches user_id
        user = await verify_token(queue_data.user_id, x_token, db)
        
        db_queue = QueueDB(
            user_id=user.id,
            raw_text=queue_data.raw_text,
            options=queue_data.options,
            properties=queue_data.properties
        )
        db.add(db_queue)
        await db.flush()
        
        if queue_data.images:
            for img in queue_data.images:
                filename = await save_base64_image(img.base64_data, img.filename)
                mime_type = magic.from_file(
                    os.path.join(settings.UPLOAD_DIR, filename), 
                    mime=True
                )
                file_size = os.path.getsize(
                    os.path.join(settings.UPLOAD_DIR, filename)
                )
                
                db_image = QueueImage(
                    queue_id=db_queue.id,
                    filename=filename,
                    mime_type=mime_type,
                    file_size=file_size
                )
                db.add(db_image)
        
        await db.commit()
        await db.refresh(db_queue)
        
        return {
            "id": db_queue.id,
            "status": "success",
            "created": db_queue.created,
            "image_count": len(db_queue.images)
        }
        
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except FileUploadError as e:
        logger.error(f"File upload error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred"
        )

@app.get("/queue/{queue_id}")
async def get_queue_item(
    queue_id: int,
    user_id: int,
    x_token: str = Header(..., description="User's authentication token"),
    db: AsyncSession = Depends(get_db)
):
    """Get queue item details"""
    try:
        # Verify token matches user_id
        user = await verify_token(user_id, x_token, db)
        
        result = await db.execute(
            select(QueueDB)
            .where(QueueDB.id == queue_id, QueueDB.user_id == user.id)
        )
        queue_item = result.scalar_one_or_none()
        
        if queue_item is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Queue item {queue_id} not found"
            )
        
        return {
            "id": queue_item.id,
            "user_id": queue_item.user_id,
            "raw_text": queue_item.raw_text,
            "status": queue_item.status,
            "options": queue_item.options,
            "properties": queue_item.properties,
            "created": queue_item.created,
            "updated": queue_item.updated,
            "images": [
                {
                    "id": img.id,
                    "filename": img.filename,
                    "mime_type": img.mime_type,
                    "file_size": img.file_size,
                    "created": img.created
                } for img in queue_item.images
            ]
        }
        
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred"
        )