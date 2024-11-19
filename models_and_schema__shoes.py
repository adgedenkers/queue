from datetime import datetime
from typing import List, Optional
from sqlalchemy import Column, String, Integer, Numeric, Text, ARRAY, TIMESTAMP, Enum
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import VARCHAR
from pydantic import BaseModel, Field, validator
import enum

Base = declarative_base()

class Gender(str, enum.Enum):
    MENS = "mens"
    WOMENS = "womens"
    UNISEX = "unisex"
    YOUTH = "youth"
    KIDS = "kids"

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

class Shoe(Base):
    __tablename__ = 'shoes'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Basic shoe information
    brand = Column(String(255), nullable=False, index=True)
    model = Column(String(255), nullable=False, index=True)
    gender = Column(Enum(Gender), nullable=False, index=True)
    size = Column(Numeric(5, 2), nullable=False, index=True)
    width = Column(String(50), default='M')
    color = Column(String(255), nullable=False, index=True)
    shoe_type = Column(String(100), nullable=False, index=True)
    style = Column(String(100), nullable=False)
    
    # Additional details
    material = Column(String(255))
    heel_type = Column(String(100))
    occasion = Column(String(100))
    condition = Column(String(50), nullable=False, default='Brand New, in Box')
    special_features = Column(ARRAY(Text))
    
    # Product identifiers and pricing
    upc = Column(String(20), unique=True)
    msrp = Column(Numeric(10, 2))
    average_ebay_selling_price = Column(Numeric(10, 2))
    category = Column(String(255), index=True)
    
    # Listing details
    photos = Column(ARRAY(Text))
    description = Column(Text)
    ebay_listing_id = Column(String(50), unique=True)
    ebay_listing_url = Column(String(255))
    listing_status = Column(
        Enum(ListingStatus),
        default=ListingStatus.NOT_LISTED,
        nullable=False
    )
    listing_start_date = Column(TIMESTAMP)
    listing_end_date = Column(TIMESTAMP)
    
    # Sales information
    sale_price = Column(Numeric(10, 2))
    buyer_username = Column(String(255))
    payment_status = Column(
        Enum(PaymentStatus),
        default=PaymentStatus.PENDING,
        nullable=False
    )
    shipping_status = Column(
        Enum(ShippingStatus),
        default=ShippingStatus.NOT_SHIPPED,
        nullable=False
    )
    shipping_tracking_number = Column(String(100))
    
    # Timestamps
    created_at = Column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        nullable=False
    )
    updated_at = Column(
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

class ShoeBase(BaseModel):
    """Base Pydantic model for shoe data validation"""
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

class ShoeCreate(ShoeBase):
    """Pydantic model for creating a new shoe"""
    upc: Optional[str] = Field(None, max_length=20)
    msrp: Optional[float] = Field(None, ge=0)
    average_ebay_selling_price: Optional[float] = Field(None, ge=0)
    photos: Optional[List[str]] = None

class ShoeUpdate(ShoeBase):
    """Pydantic model for updating an existing shoe"""
    brand: Optional[str] = None
    model: Optional[str] = None
    gender: Optional[Gender] = None
    size: Optional[float] = None
    color: Optional[str] = None
    shoe_type: Optional[str] = None
    style: Optional[str] = None

class ShoeInDB(ShoeBase):
    """Pydantic model for shoe data from database"""
    id: int
    ebay_listing_id: Optional[str] = None
    ebay_listing_url: Optional[str] = None
    listing_status: ListingStatus = ListingStatus.NOT_LISTED
    listing_start_date: Optional[datetime] = None
    listing_end_date: Optional[datetime] = None
    sale_price: Optional[float] = None
    buyer_username: Optional[str] = None
    payment_status: PaymentStatus = PaymentStatus.PENDING
    shipping_status: ShippingStatus = ShippingStatus.NOT_SHIPPED
    shipping_tracking_number: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True