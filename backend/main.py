# ============================================================
# INFOSHARE BACKEND - MAIN.PY
# ============================================================
# Complete FastAPI application with Google OAuth, JWT, and RAG
# ============================================================

import os
import sys
import time
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import pandas as pd
import uuid
import numpy as np

from fastapi import FastAPI, Depends, HTTPException, Request, status, Header, UploadFile, File, Form
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from google.auth.transport import requests
from google.oauth2 import id_token
from dotenv import load_dotenv
import jwt

# ============================================================
# IMPORTS: Database & Models
# ============================================================
from database import engine, Base, get_db, async_session_maker
from models import User

# ============================================================
# STARTUP LOGS & .env LOADING
# ============================================================
print("\n" + "=" * 70)
print("🚀 INFOSHARE BACKEND STARTUP")
print("=" * 70)

BACKEND_DIR = Path(__file__).resolve().parent
ENV_FILE = BACKEND_DIR / ".env"

print(f"📁 Backend directory: {BACKEND_DIR}")
print(f"🔍 Looking for .env at: {ENV_FILE}")
print(f"✅ .env file exists: {ENV_FILE.exists()}")

if ENV_FILE.exists():
    print(f"📖 Reading .env file...")
    with open(ENV_FILE, "r") as f:
        content = f.read()
        print("📄 .env content preview:")
        for line in content.split("\n")[:5]:
            if line.strip() and not line.startswith("#"):
                print(f"   ✓ {line[:50]}...")

load_dotenv(dotenv_path=str(ENV_FILE), override=True)

# ============================================================
# ENV VARIABLES & CONFIGURATION
# ============================================================
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "infoshare-dev-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hib-insurance-index")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")

print(f"\n📋 API Keys Check:")
print(f"   OPENROUTER_KEY: {OPENROUTER_KEY[:30] + '...' if OPENROUTER_KEY else '❌ NOT FOUND'}")
print(f"   PINECONE_API_KEY: {PINECONE_API_KEY[:30] + '...' if PINECONE_API_KEY else '❌ NOT FOUND'}")

print("\n🔑 Environment Variables:")
print(f"   GOOGLE_CLIENT_ID: {GOOGLE_CLIENT_ID[:40] + '...' if GOOGLE_CLIENT_ID else '❌ NOT FOUND'}")
print(f"   GOOGLE_API_KEY: {GOOGLE_API_KEY[:30] + '...' if GOOGLE_API_KEY else '❌ NOT FOUND'}")
print(f"   SECRET_KEY length: {len(SECRET_KEY)} ({'✅ OK' if len(SECRET_KEY) >= 20 else '⚠️ TOO SHORT'})")
print(f"   PINECONE_INDEX_NAME: {PINECONE_INDEX_NAME}")

if not GOOGLE_CLIENT_ID:
    print("\n❌ CRITICAL ERROR: GOOGLE_CLIENT_ID is missing!")
    sys.exit(1)

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

print("\n✅ Configuration loaded successfully!")
print("=" * 70 + "\n")

# ============================================================
# RAG IMPORTS
# ============================================================
try:
    from llama_index.llms.openrouter import OpenRouter
    from pinecone import Pinecone

    RAG_AVAILABLE = True
    print("✅ RAG dependencies loaded successfully")
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"⚠️ RAG dependencies not installed: {e}")

# ============================================================
# DATA PATH CONFIGURATION
# ============================================================
DOC_PATH = BACKEND_DIR / "data"

print(f"\n📁 Data directory configuration:")
print(f"   Path: {DOC_PATH}")
print(f"   Exists: {DOC_PATH.exists()}")

DOC_PATH.mkdir(parents=True, exist_ok=True)
print(f"   ✅ Data directory ready")

llm = None
PINECONE_BATCH_SIZE = 96

# ============================================================
# INITIALIZE LLM
# ============================================================
if RAG_AVAILABLE:
    try:
        print("\n📊 Initializing LLM with OpenRouter...")
        llm = OpenRouter(
            api_key=OPENROUTER_KEY,
            max_tokens=512,
            context_window=4096,
            model="google/gemma-3n-e2b-it:free",
        )
        test_response = llm.complete("Say 'RAG is enabled'")
        print(f"✅ LLM initialized: {str(test_response)[:50]}...")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize LLM: {e}")
        llm = None

# ============================================================
# JWT HELPERS
# ============================================================
def create_access_token(email: str, expires_delta: timedelta | None = None) -> str:
    """Create JWT access token."""
    if expires_delta is None:
        expires_delta = timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)

    to_encode = {
        "sub": email,
        "exp": datetime.utcnow() + expires_delta,
        "iat": datetime.utcnow(),
    }
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    print(f"🎫 JWT token created for {email}: {encoded_jwt[:30]}...")
    return encoded_jwt


async def get_current_user(authorization: str = Header(None)) -> str:
    """Validate JWT token from Authorization header."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
        )

    try:
        parts = authorization.split()
        if len(parts) != 2:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format",
            )

        scheme, token = parts
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme",
            )

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str | None = payload.get("sub")

        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )

        print(f"✅ JWT validated for: {email}")
        return email

    except jwt.ExpiredSignatureError:
        print("❌ Token expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
        )
    except jwt.InvalidTokenError as e:
        print(f"❌ Invalid token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
        )

# ============================================================
# GLOBAL VARIABLES
# ============================================================
pinecone_index = None
pc = None

# ============================================================
# LIFESPAN CONTEXT MANAGER
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    global pinecone_index
    global pc
    
    print("\n" + "=" * 70)
    print("🚀 APPLICATION STARTUP")
    print("=" * 70)

    # --- DATABASE TABLES ---
    try:
        print("📊 Creating database tables...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("✅ Database tables created successfully!")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        traceback.print_exc()

    # --- INITIALIZE PINECONE CONNECTION (QUERY ONLY) ---
    if RAG_AVAILABLE and llm:
        try:
            print("\n📊 Initializing Pinecone connection for RAG queries...")

            if not PINECONE_API_KEY or PINECONE_API_KEY == "YOUR_DUMMY_API_KEY":
                raise ValueError("❌ PINECONE_API_KEY not found or is dummy value.")

            print(f"✅ PINECONE_API_KEY loaded: {PINECONE_API_KEY[:30]}...")

            # Initialize Pinecone client
            pc = Pinecone(api_key=PINECONE_API_KEY)
            print(f"✅ Pinecone client initialized")

            # Connect to the index (assumes index already exists with data)
            index_name = PINECONE_INDEX_NAME
            
            if not pc.has_index(index_name):
                print(f"⚠️ WARNING: Index '{index_name}' not found!")
                print(f"   To create embeddings, run: python create_embeddings.py")
                print(f"   The RAG system will not work without data in Pinecone.")
                pinecone_index = None
            else:
                pinecone_index = pc.Index(index_name)
                print(f"✅ Connected to Pinecone index: {index_name}")
                print(f"✅ RAG system ready for queries")
            
        except Exception as e:
            print(f"❌ Pinecone initialization error: {type(e).__name__}: {e}")
            traceback.print_exc()
            pinecone_index = None
            pc = None
    else:
        if not RAG_AVAILABLE:
            print("⚠️ RAG features disabled (missing dependencies)")
        elif not llm:
            print("⚠️ RAG features disabled (LLM not initialized)")

    print("=" * 70 + "\n")
    
    try:
        yield
    finally:
        print("\n🛑 APPLICATION SHUTDOWN")
        print("=" * 70)

# ============================================================
# FASTAPI APP INITIALIZATION
# ============================================================
app = FastAPI(
    title="InfoShare - RAG API with Google Auth",
    description="Insurance chatbot with RAG and Google OAuth",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ============================================================
# PYDANTIC MODELS
# ============================================================
class GoogleLoginRequest(BaseModel):
    id_token: str


class UpdateProfileRequest(BaseModel):
    display_name: str
    bio: str = ""


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str


class RAGStatusResponse(BaseModel):
    """Response model for RAG status check."""
    status: str
    pinecone_connected: bool
    llm_initialized: bool
    message: str

# ============================================================
# HEALTH CHECK ENDPOINTS
# ============================================================
@app.get("/")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "InfoShare Backend",
        "rag_available": RAG_AVAILABLE,
        "rag_ready": pinecone_index is not None and llm is not None,
        "pinecone_connected": pc is not None,
        "database": "sqlite3",
        "data_directory": str(DOC_PATH),
        "data_directory_exists": DOC_PATH.exists(),
    }


@app.get("/rag/status")
async def rag_status():
    """Check RAG system status."""
    return {
        "status": "ready" if pinecone_index else "not_ready",
        "pinecone_connected": pinecone_index is not None,
        "llm_initialized": llm is not None,
        "message": "RAG system is ready for queries" if pinecone_index else "Run 'python create_embeddings.py' to load data"
    }

# ============================================================
# GOOGLE AUTH ENDPOINTS
# ============================================================
@app.post("/auth/google-login")
async def google_login(
    request: GoogleLoginRequest, db: AsyncSession = Depends(get_db)
):
    """Handle Google OAuth ID token from frontend and return JWT."""
    print("\n" + "=" * 60)
    print("🔐 GOOGLE LOGIN ENDPOINT CALLED")
    print("=" * 60)

    try:
        if not GOOGLE_CLIENT_ID:
            print("❌ GOOGLE_CLIENT_ID not configured")
            raise HTTPException(
                status_code=500, detail="GOOGLE_CLIENT_ID not configured on server"
            )

        print(f"✅ GOOGLE_CLIENT_ID loaded: {GOOGLE_CLIENT_ID[:30]}...")
        print(f"📨 Request received with token: {request.id_token[:50]}...")
        print("🔐 Verifying token with Google...")

        idinfo = id_token.verify_oauth2_token(
            request.id_token, requests.Request(), GOOGLE_CLIENT_ID
        )

        print("✅ Token verified successfully")

        email = idinfo.get("email")
        name = idinfo.get("name", "User")
        picture = idinfo.get("picture", "")
        google_id = idinfo.get("sub")

        if not email:
            print("❌ Email not found in token")
            raise HTTPException(status_code=400, detail="Email not found in token")

        print(f"👤 User: {email}, Name: {name}")

        stmt = select(User).where(User.email == email)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        is_new_user = False

        if not user:
            print(f"🆕 Creating new user: {email}")
            user = User(
                email=email,
                display_name=name,
                provider_id=google_id,
                picture=picture,
                bio="",
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
            is_new_user = True
            print(f"✅ User created successfully (ID: {user.id})")
        else:
            print(f"✅ User already exists: {email}")

        access_token = create_access_token(email)

        response_data = {
            "access_token": access_token,
            "token_type": "bearer",
            "email": email,
            "name": name,
            "picture": picture,
            "bio": user.bio if user else "",
            "is_new_user": is_new_user,
            "message": "Logged in successfully",
        }

        print(f"📤 Returning response: {response_data}")
        print("=" * 60 + "\n")
        return response_data

    except ValueError as e:
        print(f"❌ Token verification failed: {str(e)}")
        traceback.print_exc()
        print("=" * 60 + "\n")
        raise HTTPException(status_code=400, detail=f"Invalid token: {str(e)}")

    except HTTPException:
        print("=" * 60 + "\n")
        raise

    except Exception as e:
        print(f"❌ Authentication error: {str(e)}")
        print(f"❌ Exception type: {type(e).__name__}")
        traceback.print_exc()
        print("=" * 60 + "\n")
        raise HTTPException(
            status_code=500, detail=f"Authentication error: {str(e)}"
        )


@app.post("/auth/logout")
async def logout():
    """Stateless logout – frontend should clear token."""
    return {"message": "Logged out successfully"}


@app.get("/auth/me")
async def get_current_user_info(user_email: str = Depends(get_current_user)):
    """Return current user email (requires JWT)."""
    return {"email": user_email}


@app.get("/auth/profile")
async def get_profile(
    user_email: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    """Get user profile information."""
    try:
        stmt = select(User).where(User.email == user_email)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return {
            "id": user.id,
            "email": user.email,
            "display_name": user.display_name,
            "picture": user.picture,
            "bio": user.bio,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "updated_at": user.updated_at.isoformat() if user.updated_at else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fetch failed: {str(e)}")


@app.put("/auth/profile")
async def update_profile(
    display_name: str = Form(...),
    bio: str = Form(default=""),
    picture: UploadFile = File(None),
    user_email: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update user profile with optional picture upload."""
    try:
        stmt = select(User).where(User.email == user_email)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if not display_name or not display_name.strip():
            raise HTTPException(
                status_code=400, detail="Display name cannot be empty"
            )

        os.makedirs("uploads", exist_ok=True)

        picture_url = user.picture
        if picture:
            if not picture.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="File must be an image")

            contents = await picture.read()
            if len(contents) > 5 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="Image too large (max 5MB)")

            if user.picture and user.picture.startswith("/uploads/"):
                old_path = user.picture.lstrip("/")
                if os.path.exists(old_path):
                    os.remove(old_path)

            file_extension = picture.filename.split('.')[-1].lower()
            filename = f"profile_{user.id}.{file_extension}"
            filepath = f"uploads/{filename}"

            with open(filepath, "wb") as f:
                f.write(contents)

            picture_url = f"/uploads/{filename}"

        user.display_name = display_name.strip()
        user.bio = bio or ""
        user.picture = picture_url
        user.updated_at = datetime.utcnow()

        db.add(user)
        await db.commit()
        await db.refresh(user)

        return {
            "id": user.id,
            "email": user.email,
            "display_name": user.display_name,
            "picture": user.picture,
            "bio": user.bio,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "updated_at": user.updated_at.isoformat() if user.updated_at else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")

# ============================================================
# RAG QUERY ENDPOINT
# ============================================================
@app.post("/rag/query/", response_model=QueryResponse)
async def process_rag_query(
    request: QueryRequest,
    user_email: str = Depends(get_current_user)
):
    """Query Pinecone for answers using RAG."""

    if not pinecone_index or not llm or not pc:
        raise HTTPException(
            status_code=503, 
            detail="RAG system not ready. Make sure embeddings are loaded. Run: python create_embeddings.py"
        )

    try:
        print(f"\n🔎 Query from {user_email}: {request.query[:60]}...")
        print("📊 Searching Pinecone...")
        
        # Search Pinecone
        search_results = pinecone_index.search(
            namespace="namespace1",
            query={
                "top_k": 10,
                "inputs": {
                    "text": request.query
                }
            }
        )
        
        print(f"✅ Search completed")
        
        # Extract hits from response
        hits = []
        
        if hasattr(search_results, '_data_store'):
            data_store = search_results._data_store
            if isinstance(data_store, dict) and 'result' in data_store:
                result = data_store['result']
                if isinstance(result, dict) and 'hits' in result:
                    hits = result['hits']
                    print(f"✅ Found {len(hits)} hits")
        
        if not hits:
            print("⚠️ No hits found")
            return QueryResponse(
                answer="I don't have information about that. Try asking about:\n• Medicines and prices\n• Hospitals and services\n• Lab services\n• Health insurance coverage"
            )

        print(f"✅ Processing {len(hits)} hits...")

        context_chunks = []
        
        # Extract text from hits
        for i, hit in enumerate(hits, 1):
            try:
                if not isinstance(hit, dict):
                    continue
                
                fields = hit.get('fields', {})
                if not isinstance(fields, dict):
                    continue
                
                text = fields.get('text', '')
                source = fields.get('source_file', 'unknown')
                score = hit.get('_score', 0)
                
                if text and len(text.strip()) > 3:
                    context_chunks.append(text)
                    print(f"   [{i}] {source} (score: {score:.3f})")
            except Exception as e:
                print(f"   ⚠️ Error processing hit {i}: {str(e)[:50]}")
                continue

        if not context_chunks:
            print("⚠️ No valid text extracted")
            return QueryResponse(
                answer="Database has limited information on that topic."
            )

        # Build context
        context = "\n\n".join(context_chunks)
        print(f"✅ Context: {len(context)} chars from {len(context_chunks)} sources")

        # Generate answer with LLM
        print("🤖 Generating answer...")
        
        prompt = f"""You are a professional health insurance assistant for HIB (Health Insurance Board) Nepal.

Answer ONLY using the provided information. If not available, say so clearly.

Be professional, clear, and formatted well. Use bullet points, tables, or numbering as appropriate.

===== DATABASE INFORMATION =====
{context}
===== END INFORMATION =====

QUESTION: {request.query}

ANSWER:"""

        response = llm.complete(prompt)
        answer_text = str(response).strip()
        
        print(f"✅ Answer generated: {len(answer_text)} chars")
        
        return QueryResponse(answer=answer_text)

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {str(e)[:100]}")
        traceback.print_exc()
        return QueryResponse(answer="An error occurred processing your query.")

# ============================================================
# END OF FILE
# ============================================================