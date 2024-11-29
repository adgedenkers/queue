from fastapi import APIRouter

# Create the main API router
api_router = APIRouter()

# Later we'll add routes like:
# from .endpoints import users, queue, shoes
# api_router.include_router(users.router, prefix="/users", tags=["users"])
# api_router.include_router(queue.router, prefix="/queue", tags=["queue"])
# api_router.include_router(shoes.router, prefix="/shoes", tags=["shoes"])