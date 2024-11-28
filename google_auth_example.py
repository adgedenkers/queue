from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.templating import Jinja2Templates
from fastapi_login import LoginManager
from fastapi_login.exceptions import InvalidCredentialsException  

from requests_oauthlib import OAuth2Session
from typing import Optional

app = FastAPI()

# Google OAuth 2.0 credentials
client_id = "your_client_id"
client_secret = "your_client_secret"
authorization_base_url = "https://accounts.google.com/o/oauth2/v2/auth"
token_url = "https://oauth2.googleapis.com/token"
scope = ["https://www.googleapis.com/auth/userinfo.email",  
 "https://www.googleapis.com/auth/userinfo.profile"]  


# User model (you might have a database for this)
class User:
    def __init__(self, id: str, email: str):
        self.id = id
        self.email = email

# In-memory user store (replace with your actual user database)
users = {}

# Initialize login manager
manager = LoginManager(secret="your_secret_key", token_url="/auth/token")  # Replace with a strong secret key
manager.init_app(app)

# OAuth2 scheme for API endpoints
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

@manager.user_loader
def load_user(user_id: str):
    if user_id in users:
        return users[user_id]
    return None

# Login route
@app.get("/login")
async def login(request: Request):
    google = OAuth2Session(client_id, scope=scope, redirect_uri="https://api.denkers.co/callback")
    authorization_url, state = google.authorization_url(authorization_base_url)
    request.session['oauth_state'] = state
    return RedirectResponse(authorization_url)

# Callback route
@app.get("/callback")
async def callback(request: Request):
    google = OAuth2Session(client_id, state=request.session['oauth_state'], redirect_uri="https://api.denkers.co/callback")
    token = google.fetch_token(token_url, client_secret=client_secret, authorization_response=str(request.url))
    request.session['oauth_token'] = token

    # Fetch user info
    response = google.get('https://www.googleapis.com/auth/userinfo.profile')
    user_info = response.json()
    email = user_info['email']
    user_id = user_info['id']

    # Create or retrieve user
    user = User(user_id, email)
    users[user_id] = user

    # Generate and return JWT token
    access_token = manager.create_access_token(data={"sub": user_id})
    return {"access_token": access_token, "token_type": "bearer"}

# Token route for login manager
@app.post("/auth/token")
async def login_for_access_token(request: Request):
    try:
        form = await request.form()
        email = form.get("username")
        password = form.get("password")  # Not used for Google Auth
        user = load_user(email)  # Assuming email is used as user ID
        if not user:
            raise InvalidCredentialsException
        access_token = manager.create_access_token(data={"sub": user.id})
        return {"access_token": access_token, "token_type": "bearer"}
    except InvalidCredentialsException:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate":  
 "Bearer"},
        )

# Protected route
@app.get("/profile")
async def profile(user=Depends(manager)):
    return f"Hello {user.email}!"

# Logout route
@app.get("/logout")
async def logout(request: Request):
    request.session.pop('oauth_token', None)  # Clear session data
    return RedirectResponse(url_for("login"))

# Simple HTML page for login (using Jinja2Templates)
templates = Jinja2Templates(directory="templates")

@app.get("/login.html", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})  


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile="key.pem", ssl_certfile="cert.pem")