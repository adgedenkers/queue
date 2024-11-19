# main.py

# current production script

# author:    Adge Denkers
# full path: /home/ubuntu/queue/main.py
# version:   3.4
# created:   2023-09-26
# updated:   2024-11-18


from passlib.context import CryptContext
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, UploadFile, Query, File, Form, Depends, status, Header
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, text, UniqueConstraint 
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
from pydantic import BaseModel, Field, validator, EmailStr
from sqlalchemy.orm import selectinload
from dotenv import load_dotenv
import aiofiles.os as async_os
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
import enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from sqlalchemy import Column, String, Integer, Numeric, Text, ARRAY, DateTime, Enum, ForeignKey
import openai

from openai import OpenAI, OpenAIError, AsyncOpenAI

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Load environment variables
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
load_dotenv()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# OpenAI API client
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Configuration classes using environment variables
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

class Gender(str, enum.Enum):
    MENS = "Mens"
    WOMENS = "Womens"
    UNISEX = "Unisex"
    YOUTH = "Youth"
    KIDS = "Kids"

class ListingStatus(str, enum.Enum):
    NOT_LISTED = "Not Listed"
    ACTIVE = "Active"
    SOLD = "Sold"
    ENDED = "Ended"
    DRAFT = "Draft"

class PaymentStatus(str, enum.Enum):
    PENDING = "Pending"
    PAID = "Paid"
    REFUNDED = "Refunded"
    CANCELLED = "Cancelled"

class ShippingStatus(str, enum.Enum):
    NOT_SHIPPED = "Not Shipped"
    SHIPPED = "Shipped"
    DELIVERED = "Delivered"
    RETURNED = "Returned"

settings = Settings()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Set up logging
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Setup Database Connection
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
engine = create_async_engine(settings.DATABASE_URL, echo=True)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Custom exceptions
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class DatabaseError(Exception):
    pass

class FileUploadError(Exception):
    pass

class ValidationError(Exception):
    pass

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Database Models
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SQLAlchemy: User Model
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    auth_token = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcoffset)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SQLAlchemy: Queue Model
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
    
    # images = relationship("QueueImage", back_populates="queue_item")
    # user = relationship("User")
    images = relationship("QueueImage", back_populates="queue_item", lazy="selectin")
    user = relationship("User", lazy="selectin")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SQLAlchemy: QueueImage Model
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class QueueImage(Base):
    __tablename__ = "queue_images"
    
    id = Column(Integer, primary_key=True, index=True)
    queue_id = Column(Integer, ForeignKey("queue.id"))
    filename = Column(String)
    mime_type = Column(String)
    file_size = Column(Integer)
    created = Column(DateTime, default=datetime.utcnow)
    
    #queue_item = relationship("QueueDB", back_populates="images")
    queue_item = relationship("QueueDB", back_populates="images", lazy="selectin")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SQLAlchemy: Shoe Model
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class Shoe(Base):
    __tablename__ = 'shoes'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    brand = Column(String(255), nullable=True, index=True)
    model = Column(String(255), nullable=True, index=True)
    gender = Column(Enum(Gender), nullable=True, index=True)
    size = Column(Numeric(5, 2), nullable=True, index=True)
    width = Column(String(50), nullable=True, default='M')
    color = Column(String(255), nullable=True, index=True)
    shoe_type = Column(String(100), nullable=True, index=True)
    style = Column(String(100), nullable=True)
    
    material = Column(String(255), nullable=True)
    heel_type = Column(String(100), nullable=True)
    occasion = Column(String(100), nullable=True)
    condition = Column(String(50), nullable=True, default='Brand New, in Box')
    special_features = Column(ARRAY(Text), nullable=True)
    
    upc = Column(String(20), nullable=True, unique=True)
    msrp = Column(Numeric(10, 2), nullable=True)
    average_ebay_selling_price = Column(Numeric(10, 2), nullable=True)
    category = Column(String(255), nullable=True, index=True)
    
    photos = Column(ARRAY(Text), nullable=True)
    description = Column(Text, nullable=True)
    ebay_listing_id = Column(String(50), nullable=True, unique=True)
    ebay_listing_url = Column(String(255), nullable=True)
    listing_status = Column(
        Enum(ListingStatus),
        nullable=True,
        default=ListingStatus.NOT_LISTED
    )
    listing_start_date = Column(DateTime, nullable=True)
    listing_end_date = Column(DateTime, nullable=True)
    
    sale_price = Column(Numeric(10, 2), nullable=True)
    buyer_username = Column(String(255), nullable=True)
    payment_status = Column(
        Enum(PaymentStatus),
        nullable=True,
        default=PaymentStatus.PENDING
    )
    shipping_status = Column(
        Enum(ShippingStatus),
        nullable=True,
        default=ShippingStatus.NOT_SHIPPED
    )
    shipping_tracking_number = Column(String(100), nullable=True)
    
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="shoes")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Pydantic Models
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Pydantic: ImageData model
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ImageData(BaseModel):
    base64_data: str
    filename: str

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Pydantic: QueueCreate model
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class QueueCreate(BaseModel):
    user_id: int = Field(..., gt=0, description="User ID must be positive")
    raw_text: str = Field(..., min_length=1, max_length=10000)
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict)
    images: Optional[List[ImageData]] = []

    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 1,
                "raw_text": "Adidas Samba Classics Mens size 12",
                "options": {},
                "properties": {},
                "images": []
            }
        }


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Pydantic: ShoeBase model
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ShoeBase(BaseModel):
    brand: str = Field(..., min_length=1, max_length=255)
    model: str = Field(..., min_length=1, max_length=255)
    gender: Gender
    size: float = Field(..., gt=0, lt=100)
    width: str = Field(default='M', max_length=50)
    color: str = Field(..., min_length=1, max_length=255)
    shoe_type: str = Field(..., min_length=1, max_length=100)
    style: str = Field(..., min_length=1, max_length=100)
    
    material: Optional[str] = Field(None, max_length=255)
    heel_type: Optional[str] = Field(None, max_length=100)
    occasion: Optional[str] = Field(None, max_length=100)
    condition: str = Field(default='Brand New, in Box', max_length=50)
    special_features: Optional[List[str]] = None
    
    category: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None

    @validator('size')
    def validate_size(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError('Size must be a number')
        return float(v)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Pydantic: ShoeCreate model
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ShoeCreate(ShoeBase):
    upc: Optional[str] = Field(None, max_length=20)
    msrp: Optional[float] = Field(None, ge=0)
    average_ebay_selling_price: Optional[float] = Field(None, ge=0)
    photos: Optional[List[str]] = None

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Pydantic: ShoeUpdate model
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ShoeUpdate(BaseModel):
    brand: Optional[str] = None
    model: Optional[str] = None
    gender: Optional[Gender] = None
    size: Optional[float] = None
    color: Optional[str] = None
    shoe_type: Optional[str] = None
    style: Optional[str] = None
    listing_status: Optional[ListingStatus] = None
    shipping_status: Optional[ShippingStatus] = None
    payment_status: Optional[PaymentStatus] = None

class ShoeResponse(ShoeBase):
    id: int
    user_id: int
    ebay_listing_id: Optional[str] = None
    ebay_listing_url: Optional[str] = None
    listing_status: ListingStatus
    listing_start_date: Optional[datetime] = None
    listing_end_date: Optional[datetime] = None
    sale_price: Optional[float] = None
    buyer_username: Optional[str] = None
    payment_status: PaymentStatus
    shipping_status: ShippingStatus
    shipping_tracking_number: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Pydantic: UserBase model
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=150)
    email: EmailStr
    is_active: Optional[bool] = Field(default=True)

    class Config:
        orm_mode = True

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Pydantic: UserCreate model
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=255, description="Password must be at least 8 characters long.")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Pydantic: UserUpdate model
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class UserUpdate(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=150)
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=8, max_length=255)
    is_active: Optional[bool] = None

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Pydantic: UserResponse model
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class UserResponse(UserBase):
    id: int
    created_at: datetime
    updated_at: datetime

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Database dependency
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
async def get_db():
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Helper functions
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Validate image file
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Save uploaded file
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

async def save_upload_file(upload_file: UploadFile, destination: str) -> None:
    """Save uploaded file to destination"""
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        async with aiofiles.open(destination, 'wb') as out_file:
            content = await upload_file.read()
            await out_file.write(content)
    except Exception as e:
        raise FileUploadError(f"Error saving file: {str(e)}")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Save base64 image data
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Authentication dependency
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

async def verify_token(
    user_id: int,
    x_token: str = Header(...),
    db: AsyncSession = Depends(get_db)
):
    try:
        # Remove hardcoded token
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

    except Exception as e:
        logger.error(f"Token verification error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication error"
        )

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# FastAPI app
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
app = FastAPI()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# FastAPI App Startup and shutdown events
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up application")
    async with engine.begin() as conn:
        #pass
        # Drop specific tables in correct order
        #await conn.execute(text("DROP TABLE IF EXISTS queue_images"))
        #await conn.execute(text("DROP TABLE IF EXISTS queue"))
        #await conn.execute(text("DROP TABLE IF EXISTS users"))
        
        # Create tables
        #await conn.run_sync(Base.metadata.create_all)
        
        async with engine.begin() as conn:
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

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Base API Endpoints
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Healthcheck endpoint
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

@app.get("/healthcheck")
async def healthcheck(db: AsyncSession = Depends(get_db)):
    """Health check endpoint"""
    try:
        await db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        db_status = "disconnected"
    
    return {
        "status": "the denkers.co api is fully operational",
        "timestamp": datetime.utcnow().isoformat(),
        "database": db_status
    }

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Queue API Endpoints
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# -----------------------------------------------------------------------------
# List queue items with optional filters
# -----------------------------------------------------------------------------

@app.get("/queue/")
async def list_queue_items(
    user_id: int = Header(..., description="User ID"),
    x_token: str = Header(..., description="User Authentication Token"),
    skip: int = Query(0),
    limit: int = Query(100),
    status: Optional[str] = Query(None),
    active: Optional[bool] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """List queue items with optional filters"""
    try:
        user = await verify_token(user_id, x_token, db)
        
        query = select(QueueDB)\
            .options(selectinload(QueueDB.images))\
            .where(QueueDB.user_id == user.id)
        
        if status:
            query = query.where(QueueDB.status == status)
        if active is not None:
            query = query.where(QueueDB.active == active)
            
        query = query.offset(skip).limit(limit)\
            .order_by(QueueDB.created.desc())
            
        result = await db.execute(query)
        queue_items = result.scalars().all()
        
        return [{
            "id": item.id,
            "raw_text": item.raw_text,
            "status": item.status,
            "active": item.active,
            "options": item.options,
            "properties": item.properties,
            "created": item.created,
            "updated": item.updated,
            "images": [{
                "id": img.id,
                "filename": img.filename,
                "mime_type": img.mime_type,
                "file_size": img.file_size,
                "created": img.created
            } for img in item.images]
        } for item in queue_items]
        
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred"
        )

# -----------------------------------------------------------------------------
# Create a new queue item
# -----------------------------------------------------------------------------

@app.post("/queue/", status_code=status.HTTP_201_CREATED)
async def create_queue_item(
    queue_data: QueueCreate,
    x_token: str = Header(..., description="User Authentication Token"),
    db: AsyncSession = Depends(get_db)
):
    try:
        # Log incoming request data
        logger.info(f"Incoming queue_data: {queue_data.dict()}")

        # Verify token matches user_id
        user = await verify_token(queue_data.user_id, x_token, db)

        # Create Queue item
        db_queue = QueueDB(
            user_id=user.id if hasattr(user, 'id') else 1,
            raw_text=queue_data.raw_text,
            options=queue_data.options,
            properties=queue_data.properties
        )
        db.add(db_queue)
        await db.flush()

        # Add images if they are provided
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

        # Reload the queue item with images to ensure it is in a valid async context
        result = await db.execute(
            select(QueueDB).options(selectinload(QueueDB.images)).where(QueueDB.id == db_queue.id)
        )
        db_queue = result.scalar_one()

        return {
            "id": db_queue.id,
            "status": "success",
            "created": db_queue.created,
            "image_count": len(db_queue.images) if db_queue.images else 0
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

# -----------------------------------------------------------------------------
# Get queue item details
# -----------------------------------------------------------------------------

@app.get("/queue/{queue_id}")
async def get_queue_item(
    queue_id: int,
    user_id: int = Header(..., description="User ID"),
    x_token: str = Header(..., description="User Authentication Token"),
    db: AsyncSession = Depends(get_db)
):
    """Get queue item details"""
    try:
        user = await verify_token(user_id, x_token, db)
        
        result = await db.execute(
            select(QueueDB)
            .options(selectinload(QueueDB.images))
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
    
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Shoes API Endpoints
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# -----------------------------------------------------------------------------
# Supporting Functions
# -----------------------------------------------------------------------------

async def get_shoe_data(shoe_description: str, model: str = "gpt-4") -> Dict[str, Any]:
    try:
        client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        
        initial_response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise data extractor. Respond only with the requested JSON format."},
                {"role": "user", "content": shoe_description}
            ]
        )
        
        if not initial_response.choices:
            raise ValueError("No response from OpenAI API")
            
        shoe_data = json.loads(initial_response.choices[0].message.content)
        
        with open('shoe_prompt.txt', 'r') as file:
            shoe_prompt = file.read()
            
        enhanced_response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": shoe_prompt},
                {"role": "user", "content": json.dumps(shoe_data)}
            ]
        )
        
        if not enhanced_response.choices:
            raise ValueError("No enhanced response from OpenAI API")
            
        return json.loads(enhanced_response.choices[0].message.content)
        
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="External service unavailable"
        )
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid response format from external service"
        )
    except Exception as e:
        logger.error(f"Unexpected error in get_shoe_data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing shoe data"
        )


# -----------------------------------------------------------------------------
# Function: save_shoe_data_to_db
# -----------------------------------------------------------------------------
async def save_shoe_data_to_db(
    enhanced_data: Dict[str, Any], 
    user_id: int,
    db: AsyncSession
) -> Shoe:
    """
    Save enhanced shoe data from ChatGPT to the database.
    
    Args:
        enhanced_data (Dict[str, Any]): The enhanced shoe data from ChatGPT
        user_id (int): The user ID who owns this shoe
        db (AsyncSession): Database session
        
    Returns:
        Shoe: The created shoe database object
    """
    try:
        # Convert string values to appropriate enums where needed
        gender_value = enhanced_data.get("gender", "unisex").lower()
        if gender_value in [e.value for e in Gender]:
            gender = Gender(gender_value)
        else:
            gender = Gender.UNISEX
            
        # Prepare shoe data dictionary
        shoe_data = {
            "user_id": user_id,
            "brand": enhanced_data.get("brand", ""),
            "model": enhanced_data.get("model", ""),
            "gender": gender,
            "size": float(enhanced_data.get("size", 0.0)),
            "width": enhanced_data.get("width", "M"),
            "color": enhanced_data.get("color", ""),
            "shoe_type": enhanced_data.get("shoe_type", ""),
            "style": enhanced_data.get("style", ""),
            "material": enhanced_data.get("material"),
            "heel_type": enhanced_data.get("heel_type"),
            "occasion": enhanced_data.get("occasion"),
            "condition": enhanced_data.get("condition", "Brand New, in Box"),
            "special_features": enhanced_data.get("special_features", []),
            "upc": enhanced_data.get("upc"),
            "msrp": float(enhanced_data.get("msrp", 0.0)) if enhanced_data.get("msrp") else None,
            "average_ebay_selling_price": float(enhanced_data.get("average_ebay_selling_price", 0.0)) if enhanced_data.get("average_ebay_selling_price") else None,
            "category": enhanced_data.get("category"),
            "description": enhanced_data.get("description"),
            "listing_status": ListingStatus.NOT_LISTED,
            "payment_status": PaymentStatus.PENDING,
            "shipping_status": ShippingStatus.NOT_SHIPPED
        }

        # Create new shoe object
        db_shoe = Shoe(**shoe_data)
        
        # Add to database
        db.add(db_shoe)
        await db.commit()
        await db.refresh(db_shoe)
        
        logger.info(f"Successfully saved shoe data to database with ID: {db_shoe.id}")
        return db_shoe
        
    except ValueError as e:
        logger.error(f"Value error while saving shoe data: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid data format: {str(e)}"
        )
    except SQLAlchemyError as e:
        logger.error(f"Database error while saving shoe data: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error saving to database"
        )

# # Example usage in an endpoint:
# @app.post("/shoes/from_gpt")
# async def create_shoe_from_gpt(
#     enhanced_data: Dict[str, Any],
#     user_id: int = Header(..., description="User ID"),
#     x_token: str = Header(..., description="User Authentication Token"),
#     db: AsyncSession = Depends(get_db)
# ):
#     """Create a new shoe record from GPT-enhanced data"""
#     try:
#         # Verify user
#         user = await verify_token(user_id, x_token, db)
        
#         # Save the shoe data
#         db_shoe = await save_shoe_data_to_db(enhanced_data, user.id, db)
        
#         return {
#             "status": "success",
#             "message": "Shoe record created successfully",
#             "shoe_id": db_shoe.id
#         }
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error creating shoe from GPT data: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="An unexpected error occurred"
#         )

# -----------------------------------------------------------------------------
# POST /shoes/                  Create a new shoe record
# -----------------------------------------------------------------------------
@app.post("/shoes/")
async def create_shoe(
    shoe: ShoeCreate,
    user_id: int,
    x_token: str = Header(..., description="User Authentication Token"),
    db: AsyncSession = Depends(get_db)
):
    try:
        user = await verify_token(user_id, x_token, db)
        
        shoe_description = f"{shoe.brand} {shoe.model} {shoe.gender} size {shoe.size}"
        if shoe.color:
            shoe_description += f" {shoe.color}"
        
        try:
            supplemental_data = await get_shoe_data(shoe_description)
            supp_shoe_dict = {
                "user_id": user.id,
                "brand": supplemental_data.get("brand", shoe.brand),
                "model": supplemental_data.get("model", shoe.model),
                "gender": supplemental_data.get("gender", shoe.gender),
                "size": supplemental_data.get("size", shoe.size),
                "width": supplemental_data.get("width", "M"),
                "color": supplemental_data.get("color", shoe.color),
                "shoe_type": supplemental_data.get("shoe_type", ""),
                "style": supplemental_data.get("style", ""),
                "material": supplemental_data.get("material"),
                "heel_type": supplemental_data.get("heel_type"),
                "occasion": supplemental_data.get("occasion"),
                "condition": supplemental_data.get("condition", "Brand New, in Box"),
                "special_features": supplemental_data.get("special_features", []),
                "upc": supplemental_data.get("upc"),
                "msrp": supplemental_data.get("msrp"),
                "category": supplemental_data.get("category")
            }
            db_shoe = Shoe(**supp_shoe_dict)
            
        except Exception as e:
            logger.error(f"Error processing supplemental data: {str(e)}")
            shoe_dict = shoe.dict()
            db_shoe = Shoe(**shoe_dict, user_id=user.id)
            
        db.add(db_shoe)
        await db.commit()
        await db.refresh(db_shoe)
        
        return {
            "status": "success",
            "message": "Shoe record created successfully",
            "shoe_id": db_shoe.id
        }
        
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred"
        )

# -----------------------------------------------------------------------------
# GET /shoes/                               List shoes with optional filters
# -----------------------------------------------------------------------------

@app.get("/shoes/", response_model=List[ShoeResponse])
async def list_shoes(
    user_id: int,
    #x_token: str = Header(..., description="User Authentication Token"),
    skip: int = 0,
    limit: int = 100,
    brand: Optional[str] = None,
    gender: Optional[Gender] = None,
    listing_status: Optional[ListingStatus] = None,
    db: AsyncSession = Depends(get_db)
):
    """List shoes with optional filters"""
    try:
        # Verify token matches user_id
        #user = await verify_token(user_id, x_token, db)
        
        query = select(Shoe).where(Shoe.user_id == 1)
        
        if brand:
            query = query.where(Shoe.brand.ilike(f"%{brand}%"))
        if gender:
            query = query.where(Shoe.gender == gender)
        if listing_status:
            query = query.where(Shoe.listing_status == listing_status)
            
        query = query.offset(skip).limit(limit)
        result = await db.execute(query)
        shoes = result.scalars().all()
        
        return shoes
        
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred"
        )

# -----------------------------------------------------------------------------
# GET /shoes/{shoe_id}                                      Get shoe details
# -----------------------------------------------------------------------------
@app.get("/shoes/{shoe_id}", response_model=ShoeResponse)
async def get_shoe(
    shoe_id: int,
    user_id: int,
    x_token: str = Header(..., description="User Authentication Token"),
    db: AsyncSession = Depends(get_db)
):
    """Get shoe details"""
    try:
        # Verify token matches user_id
        user = await verify_token(user_id, x_token, db)
        
        result = await db.execute(
            select(Shoe)
            .where(Shoe.id == shoe_id, Shoe.user_id == user.id)
        )
        shoe = result.scalar_one_or_none()
        
        if not shoe:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Shoe {shoe_id} not found"
            )
            
        return shoe
        
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred"
        )

# -----------------------------------------------------------------------------
# PATCH /shoes/{shoe_id}                                Update a shoe
# -----------------------------------------------------------------------------

@app.patch("/shoes/{shoe_id}", response_model=ShoeResponse)
async def update_shoe(
    shoe_id: int,
    shoe_update: ShoeUpdate,
    user_id: int,
    x_token: str = Header(..., description="User Authentication Token"),
    db: AsyncSession = Depends(get_db)
):
    """Update shoe details"""
    try:
        # Verify token matches user_id
        user = await verify_token(user_id, x_token, db)
        
        result = await db.execute(
            select(Shoe)
            .where(Shoe.id == shoe_id, Shoe.user_id == user.id)
        )
        shoe = result.scalar_one_or_none()
        
        if not shoe:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Shoe {shoe_id} not found"
            )
        
        update_data = shoe_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(shoe, field, value)
        
        shoe.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(shoe)
        return shoe
        
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred"
        )

# -----------------------------------------------------------------------------
# DELETE /shoes/{shoe_id}                           Delete a shoe
# -----------------------------------------------------------------------------

@app.delete("/shoes/{shoe_id}")
async def delete_shoe(
    shoe_id: int,
    user_id: int,
    x_token: str = Header(..., description="User Authentication Token"),
    db: AsyncSession = Depends(get_db)
):
    """Delete a shoe"""
    try:
        # Verify token matches user_id
        user = await verify_token(user_id, x_token, db)
        
        result = await db.execute(
            select(Shoe)
            .where(Shoe.id == shoe_id, Shoe.user_id == user.id)
        )
        shoe = result.scalar_one_or_none()
        
        if not shoe:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Shoe {shoe_id} not found"
            )
        
        await db.delete(shoe)
        await db.commit()
        
        return {"message": f"Shoe {shoe_id} successfully deleted"}
        
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred"
        )

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Users API Endpoints
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# -----------------------------------------------------------------------------
# GET /users/                           Get List of all users
# -----------------------------------------------------------------------------

@app.get("/users/", response_model=List[UserResponse])
async def list_users(
    user_id: int = Header(..., description="User ID"), 
    x_token: str = Header(..., description="User Authentication Token"), 
    db: AsyncSession = Depends(get_db)
):
    """List all users"""
    try:
        user = await verify_token(user_id, x_token, db) 
        result = await db.execute(select(User))
        users = result.scalars().all()
        return users

    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred"
        )

# -----------------------------------------------------------------------------
# GET /user/{user_id}                   Get user details
# -----------------------------------------------------------------------------
@app.get("/user/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get user details"""
    try:
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {user_id} not found"
            )
            
        return user
        
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred"
        )

# -----------------------------------------------------------------------------
# POST /users/                          Create a new user
# -----------------------------------------------------------------------------
@app.post("/users/", response_model=UserResponse)
async def create_user(
    user: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new user"""
    try:
        db_user = User(**user.dict())
        db.add(db_user)
        await db.commit()
        await db.refresh(db_user)
        
        return db_user
        
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred"
        )

# -----------------------------------------------------------------------------
# PATCH /user/{user_id}                 Update user details
# -----------------------------------------------------------------------------

@app.patch("/user/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update user details"""
    try:
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {user_id} not found"
            )
        
        update_data = user_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)
        
        user.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(user)
        return user
        
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred"
        )

# -----------------------------------------------------------------------------
# DELETE /user/{user_id}                Delete user
# -----------------------------------------------------------------------------
@app.delete("/user/{user_id}")
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete a user"""
    try:
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {user_id} not found"
            )
        
        await db.delete(user)
        await db.commit()
        
        return {"message": f"User {user_id} successfully deleted"}
        
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred"
        )