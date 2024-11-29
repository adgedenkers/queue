from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
from uuid import UUID

class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=150)
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = True
    
    # New fields from people table
    name: Optional[str] = None
    name_last: Optional[str] = None
    name_first: Optional[str] = None
    name_middle: Optional[str] = None
    name_prefix: Optional[str] = None
    name_suffix: Optional[str] = None
    mobile: Optional[str] = None
    dob: Optional[datetime] = None
    place_of_birth: Optional[str] = None

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=255)

class UserUpdate(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=150)
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=8, max_length=255)
    is_active: Optional[bool] = None
    
    # New fields from people table
    name: Optional[str] = None
    name_last: Optional[str] = None
    name_first: Optional[str] = None
    name_middle: Optional[str] = None
    name_prefix: Optional[str] = None
    name_suffix: Optional[str] = None
    mobile: Optional[str] = None
    dob: Optional[datetime] = None
    place_of_birth: Optional[str] = None

class UserResponse(UserBase):
    id: int
    uuid: UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True