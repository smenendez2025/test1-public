# pullbrain-AI/auth.py
import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel

# --- CONFIGURACIÓN ---

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 90 

if not SECRET_KEY:
    raise ValueError("No JWT_SECRET_KEY set for JWT authentication. Please set this environment variable.")

class TokenData(BaseModel):
    sub: Optional[str] = None # "sub" (subject) es el nombre estándar para el ID de usuario en JWT.

def create_access_token(data: dict):
    """Crea un nuevo token JWT."""
    to_encode = data.copy()
    if "role" not in to_encode:
        to_encode["role"] = "user"
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- DEPENDENCIA DE FASTAPI --- 
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") 

async def verify_token(token: str = Depends(oauth2_scheme)) -> str:
    """Decodifica y valida el token JWT para obtener el ID de usuario."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(**payload)
        return token_data 
    except JWTError:
        # Si la decodificación falla (firma inválida, token expirado, etc.), es un error.
        raise credentials_exception
    
async def verify_admin(token_data: TokenData = Depends(verify_token)) -> str:
    """
    Dependencia que verifica que el usuario autenticado tiene el rol 'admin'.
    Reutiliza verify_token para la validación inicial.
    """
    if token_data.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Acceso denegado: se requieren permisos de administrador."
        )
    return token_data.sub 
