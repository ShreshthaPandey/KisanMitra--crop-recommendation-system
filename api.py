from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from pathlib import Path
from datetime import datetime, timedelta, timezone
import sqlite3
import hashlib
import secrets
import hmac


app = FastAPI(title="KisanMitra API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- DATABASE ----------------

DB_PATH = Path(__file__).with_name("kisanmitra.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                state TEXT,
                district TEXT,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS auth_tokens (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)


initialize_database()


# ---------------- SCHEMAS ----------------

class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    state: str = ""
    district: str = ""


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# ---------------- HELPERS ----------------

def hash_password(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        bytes.fromhex(salt),
        260000
    ).hex()


def create_token(user_id):
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    expires = (
        datetime.now(timezone.utc) + timedelta(days=7)
    ).isoformat()

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO auth_tokens
            (token_hash, user_id, expires_at)
            VALUES (?, ?, ?)
            """,
            (token_hash, user_id, expires)
        )

    return token


def user_response(user):
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "state": user["state"],
        "district": user["district"]
    }


# ---------------- BASIC ROUTES ----------------

@app.get("/")
def home():
    return {"message": "KisanMitra API is running"}


@app.get("/api/health")
def health():
    return {
        "status": "success",
        "message": "FastAPI backend connected successfully"
    }


# ---------------- REGISTRATION ----------------

@app.post("/api/auth/register")
def register(data: RegisterRequest):

    if len(data.password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least 6 characters"
        )

    if not data.name.strip():
        raise HTTPException(
            status_code=400,
            detail="Name is required"
        )

    email = str(data.email).lower().strip()

    with get_db() as conn:

        existing = conn.execute(
            "SELECT id FROM users WHERE email = ?",
            (email,)
        ).fetchone()

        if existing:
            raise HTTPException(
                status_code=409,
                detail="Email is already registered"
            )

        salt = secrets.token_hex(16)
        password_hash = hash_password(data.password, salt)

        cursor = conn.execute(
            """
            INSERT INTO users
            (name, email, password_hash, salt, state, district, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.name.strip(),
                email,
                password_hash,
                salt,
                data.state,
                data.district,
                datetime.now(timezone.utc).isoformat()
            )
        )

        user_id = cursor.lastrowid

        user = conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()

    token = create_token(user_id)

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user_response(user)
    }


# ---------------- LOGIN ----------------

@app.post("/api/auth/login")
def login(data: LoginRequest):

    email = str(data.email).lower().strip()

    with get_db() as conn:

        user = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,)
        ).fetchone()

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    entered_hash = hash_password(
        data.password,
        user["salt"]
    )

    if not hmac.compare_digest(
        entered_hash,
        user["password_hash"]
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    token = create_token(user["id"])

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user_response(user)
    }