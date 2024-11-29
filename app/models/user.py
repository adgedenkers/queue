from sqlalchemy import Column, Integer, String, DateTime, Boolean, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
from datetime import datetime
from app.db.base import Base

class User(Base):
    __tablename__ = "users"
    
    # Primary Key
    id = Column(Integer, primary_key=True, index=True)
    
    # UUID for public identification
    uuid = Column(UUID(as_uuid=True), unique=True, nullable=False, default=uuid.uuid4)
    
    # Authentication Fields
    username = Column(String, nullable=True)
    auth_token = Column(String, nullable=True)
    
    # Name Fields (from people table)
    name = Column(String, nullable=True)
    name_last = Column(String, nullable=True)
    name_first = Column(String, nullable=True)
    name_middle = Column(String, nullable=True)
    name_prefix = Column(String, nullable=True)
    name_suffix = Column(String, nullable=True)
    
    # Contact Information
    email = Column(String, nullable=True)
    mobile = Column(String, nullable=True)
    
    # Personal Information
    dob = Column(DateTime, nullable=True)
    place_of_birth = Column(String, nullable=True)
    
    # Status and Timestamps
    is_active = Column(Boolean, nullable=False, server_default=text('true'))
    created_at = Column(DateTime, nullable=False, server_default=func.current_timestamp())
    updated_at = Column(DateTime, nullable=False, server_default=func.current_timestamp(), onupdate=func.current_timestamp())