from sqlalchemy import Column, String, Integer, Numeric, Text, ARRAY, TIMESTAMP
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import VARCHAR
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional

Base = declarative_base()

class Shoe(Base):
    __tablename__ = 'shoes'

    id = Column(Integer, primary_key=True, autoincrement=True)
    brand = Column(String(255), nullable=False)
    model = Column(String(255), nullable=False)
    gender = Column(String(50), nullable=False)
    size = Column(Numeric(5, 2), nullable=False)
    width = Column(String(50), default='M')
    color = Column(String(255), nullable=False)
    shoe_type = Column(String(100), nullable=False)
    style = Column(String(100), nullable=False)
    material = Column(String(255))
    heel_type = Column(String(100))
    occasion = Column(String(100))
    condition = Column(String(50), nullable=False, default='Brand New, in Box')
    special_features = Column(ARRAY(Text))
    upc = Column(String(20))
    msrp = Column(Numeric(10, 2))
    average_ebay_selling_price = Column(Numeric(10, 2))
    category = Column(String(255))
    photos = Column(ARRAY(Text))
    description = Column(Text)
    ebay_listing_id = Column(String(50))
    ebay_listing_url = Column(String(255))
    listing_status = Column(String(50), default='Not Listed')
    listing_start_date = Column(TIMESTAMP)
    listing_end_date = Column(TIMESTAMP)
    sale_price = Column(Numeric(10, 2))
    buyer_username = Column(String(255))
    payment_status = Column(String(50), default='Pending')
    shipping_status = Column(String(50), default='Not Shipped')
    shipping_tracking_number = Column(String(100))
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)


# Pydantic model
class ShoeModel(BaseModel):
    brand: str
    model: str
    gender: str
    size: float
    width: Optional[str] = 'M'
    color: str
    shoe_type: str
    style: str
    material: Optional[str] = None
    heel_type: Optional[str] = None
    occasion: Optional[str] = None
    condition: Optional[str] = 'Brand New, in Box'
    special_features: Optional[List[str]] = None
    upc: Optional[str] = None
    msrp: Optional[float] = None
    average_ebay_selling_price: Optional[float] = None
    category: Optional[str] = None
    photos: Optional[List[str]] = None
    description: Optional[str] = None
    ebay_listing_id: Optional[str] = None
    ebay_listing_url: Optional[str] = None
    listing_status: Optional[str] = 'Not Listed'
    listing_start_date: Optional[datetime] = None
    listing_end_date: Optional[datetime] = None
    sale_price: Optional[float] = None
    buyer_username: Optional[str] = None
    payment_status: Optional[str] = 'Pending'
    shipping_status: Optional[str] = 'Not Shipped'
    shipping_tracking_number: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True
