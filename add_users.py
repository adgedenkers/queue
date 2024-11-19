import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from main_dev import async_session, User
from datetime import datetime

async def add_user(username: str, auth_token: str):
    async with async_session() as session:
        async with session.begin():
            user = User(
                username=username,
                auth_token=auth_token,
                created_at=datetime.utcnow()
            )
            session.add(user)
        await session.commit()

if __name__ == "__main__":
    new_username = "adge"
    new_auth_token = "6b1f9eeb7e3a1f6e3b3f1a2c5a7d9e1b"
    asyncio.run(add_user(new_username, new_auth_token))
