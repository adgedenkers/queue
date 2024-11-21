# db_init.py

import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.sql import text
from sqlalchemy.exc import SQLAlchemyError
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get database URL from environment variable or use default
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+asyncpg://user:password@localhost:5432/queue_db"
).replace('postgresql://', 'postgresql+asyncpg://')

# SQL statements for creating tables
CREATE_TABLES_SQL = """
-- Create enum types first
DO $$ BEGIN
    CREATE TYPE gender_enum AS ENUM ('Mens', 'Womens', 'Unisex', 'Youth', 'Kids');
    EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE listing_status_enum AS ENUM ('Not Listed', 'Active', 'Sold', 'Ended', 'Draft');
    EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE payment_status_enum AS ENUM ('Pending', 'Paid', 'Refunded', 'Cancelled');
    EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE shipping_status_enum AS ENUM ('Not Shipped', 'Shipped', 'Delivered', 'Returned');
    EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR,
    auth_token VARCHAR,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create queue table
CREATE TABLE IF NOT EXISTS queue (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    raw_text TEXT,
    status VARCHAR,
    active BOOLEAN,
    options JSONB,
    properties JSONB,
    created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create queue_images table
CREATE TABLE IF NOT EXISTS queue_images (
    id SERIAL PRIMARY KEY,
    queue_id INTEGER REFERENCES queue(id),
    filename VARCHAR,
    mime_type VARCHAR,
    file_size INTEGER,
    created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create shoes table
CREATE TABLE IF NOT EXISTS shoes (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    queue_id INTEGER REFERENCES queue(id),
    brand VARCHAR,
    model VARCHAR,
    gender gender_enum,
    size NUMERIC(5,2),
    width VARCHAR,
    color VARCHAR,
    shoe_type VARCHAR,
    style VARCHAR,
    material VARCHAR,
    heel_type VARCHAR,
    occasion VARCHAR,
    condition VARCHAR,
    special_features TEXT[],
    upc VARCHAR,
    msrp NUMERIC(10,2),
    average_ebay_selling_price NUMERIC(10,2),
    category VARCHAR,
    photos TEXT[],
    description TEXT,
    ebay_listing_id VARCHAR,
    ebay_listing_url VARCHAR,
    listing_status listing_status_enum DEFAULT 'Not Listed',
    listing_start_date TIMESTAMP,
    listing_end_date TIMESTAMP,
    sale_price NUMERIC(10,2),
    buyer_username VARCHAR,
    payment_status payment_status_enum DEFAULT 'Pending',
    shipping_status shipping_status_enum DEFAULT 'Not Shipped',
    shipping_tracking_number VARCHAR,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Insert default user if not exists
INSERT INTO users (id, username, auth_token)
VALUES (1, 'default', '6b1f9eeb7e3a1f6e3b3f1a2c5a7d9e1b')
ON CONFLICT (id) DO UPDATE 
SET auth_token = EXCLUDED.auth_token,
    updated_at = CURRENT_TIMESTAMP;
"""

async def init_db():
    """Initialize the database by creating all necessary tables"""
    try:
        # Create engine
        engine = create_async_engine(DATABASE_URL, echo=True)
        
        logger.info("Starting database initialization...")
        
        # Create tables
        async with engine.begin() as conn:
            # Execute the SQL statements
            await conn.execute(text(CREATE_TABLES_SQL))
            
            # Verify tables were created
            result = await conn.execute(
                text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
            )
            tables = [row[0] for row in result]
            
            logger.info("Created tables: %s", tables)
            
        logger.info("Database initialization completed successfully!")
        
        # Close the engine
        await engine.dispose()
        
    except SQLAlchemyError as e:
        logger.error("Database error: %s", str(e))
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", str(e))
        raise

if __name__ == "__main__":
    try:
        asyncio.run(init_db())
    except KeyboardInterrupt:
        logger.info("Database initialization interrupted by user")
    except Exception as e:
        logger.error("Failed to initialize database: %s", str(e))
        exit(1)