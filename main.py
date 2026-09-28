from dotenv import load_dotenv
load_dotenv()

import asyncio
import os
import re
import traceback
import uuid
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field

from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from tts import create_dialogue_voice
from video_builder import build_video
from script_generator import generate_script

from jobs import (
    create_job,
    set_status,
    set_result,
    set_error,
    get_job,
    JobStatus,
)

from rate_limiter import (
    limiter,
    check_daily_limit,
    get_daily_usage,
    DAILY_GENERATION_LIMIT,
)

from db import (
    init_db,
    create_user,
    get_user_by_email,
    get_user,
    create_session,
    get_user_by_session,
    delete_session,
    verify_password,
    reserve_credits,
    refund_credits,
    set_plan,
    create_project,
    update_project,
    list_projects,
    public_user,
)


# ============================================================
# APP CONFIGURATION
# ============================================================

APP_NAME = "Clip Pirate .ai"
SESSION_COOKIE = "clip_pirate_session"

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title=APP_NAME)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(BASE_DIR / "data" / "videos", exist_ok=True)
os.makedirs(BASE_DIR / "data" / "audio", exist_ok=True)
os.makedirs(BASE_DIR / "data" / "temp_clips", exist_ok=True)

app.mount(
    "/videos",
    StaticFiles(directory=BASE_DIR / "data" / "videos"),
    name="videos",
)

# IMPORTANT:
# /app is the public ClipPirate entry point.
app.mount(
    "/app",
    StaticFiles(directory=STATIC_DIR, html=True),
    name="static",
)


# ============================================================
# DATABASE
# ============================================================

init_db()


# ============================================================
# REQUEST MODELS
# ============================================================

class SignupRequest(BaseModel):
    name: str = Field(
        min_length=2,
        max_length=80
    )

    email: EmailStr

    password: str = Field(
        min_length=8,
        max_length=128
    )


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class VideoRequest(BaseModel):
    topic: str = Field(
        min_length=1,
        max_length=300
    )

    duration: int = Field(
        default=30,
        ge=15,
        le=60
    )

    aspect_ratio: str = Field(
        default="4:3"
    )

    voice: str = Field(
        default="professional"
    )

    captions: bool = True


# ============================================================
# AUTHENTICATION HELPERS
# ============================================================

def current_user(request: Request):
    """
    Return the currently logged-in user.

    Raises 401 when there is no valid session.
    """

    token = request.cookies.get(SESSION_COOKIE)

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Please log in to continue."
        )

    user = get_user_by_session(token)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Please log in to continue."
        )

    return user


# ============================================================
# FILE / VIDEO HELPERS
# ============================================================

def safe_filename(topic: str) -> str:
    slug = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "-",
        topic.strip()
    ).strip("-").lower()

    return (
        f"{slug or 'video'}-"
        f"{uuid.uuid4().hex[:8]}"
    )


def credit_cost(duration: int) -> int:
    if duration <= 30:
        return 1

    if duration <= 45:
        return 2

    return 3


def normalize_ratio(ratio: str):
    allowed = {
        "4:3",
        "9:16",
        "16:9",
        "1:1",
    }

    if ratio not in allowed:
        raise HTTPException(
            status_code=422,
            detail="Unsupported aspect ratio."
        )

    return ratio


def normalize_voice(voice: str):
    allowed = {
        "professional",
        "energetic",
        "calm",
    }

    if voice not in allowed:
        raise HTTPException(
            status_code=422,
            detail="Unsupported voice."
        )

    return voice


# ============================================================
# BASIC API HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": APP_NAME
    }


@app.get("/auth/health")
def auth_health():
    return {
        "ok": True,
        "service": "Clip Pirate .ai authentication"
    }


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

@app.get("/")
async def root():
    """
    Root URL is deliberately NOT the ClipPirate application.

    The actual application is available at /app.
    """

    return {
        "service": APP_NAME,
        "status": "running",
        "app": "/app"
    }


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/dashboard")
async def dashboard(request: Request):
    """
    Dashboard is protected.

    Logged-in users:
        /dashboard -> dashboard

    Logged-out users:
        /dashboard -> /app
    """

    try:
        current_user(request)

    except HTTPException:
        return RedirectResponse(
            url="/app",
            status_code=303
        )

    return FileResponse(
        STATIC_DIR / "index.html"
    )


# ============================================================
# AUTHENTICATION
# ============================================================

@app.post("/auth/signup")
def signup(
    body: SignupRequest,
    response: Response
):
    """
    Create a new account.

    Duplicate email returns:
        Email already exists.
    """

    existing_user = get_user_by_email(
        body.email
    )

    if existing_user:
        raise HTTPException(
            status_code=409,
            detail="Email already exists."
        )

    try:
        user_id = create_user(
            body.name,
            body.email,
            body.password
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc)
        ) from exc

    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=409,
            detail="Email already exists."
        )

    token = create_session(user_id)

    response.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=30 * 86400,
        path="/",
    )

    user = get_user(user_id)

    return {
        "user": public_user(user)
    }


@app.post("/auth/login")
def login(
    body: LoginRequest,
    response: Response
):
    user = get_user_by_email(
        body.email
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password."
        )

    if not verify_password(
        body.password,
        user["password_hash"]
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password."
        )

    token = create_session(
        user["id"]
    )

    response.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=30 * 86400,
        path="/",
    )

    return {
        "user": public_user(
            get_user(user["id"])
        )
    }


# ============================================================
# AUTH COMPATIBILITY ALIASES
# ============================================================

@app.post("/signup")
def signup_alias(
    body: SignupRequest,
    response: Response
):
    return signup(
        body,
        response
    )


@app.post("/login")
def login_alias(
    body: LoginRequest,
    response: Response
):
    return login(
        body,
        response
    )


# ============================================================
# LOGOUT
# ============================================================

@app.post("/auth/logout")
def logout(
    request: Request,
    response: Response
):
    token = request.cookies.get(
        SESSION_COOKIE
    )

    if token:
        delete_session(token)

    response.delete_cookie(
        SESSION_COOKIE,
        path="/"
    )

    return {
        "ok": True
    }


# ============================================================
# CURRENT USER
# ============================================================

@app.get("/auth/me")
def me(request: Request):
    user = current_user(request)

    return {
        "user": public_user(user)
    }


# ============================================================
# CREDITS
# ============================================================

@app.get("/credits")
def credits(request: Request):
    user = current_user(request)

    return {
        "user": public_user(user)
    }


# ============================================================
# PROJECTS API
# ============================================================

@app.get("/projects")
def projects(request: Request):
    """
    IMPORTANT:

    This is an API endpoint.
    Do NOT turn /projects into a frontend page because
    the frontend uses this endpoint to retrieve project data.
    """

    user = current_user(request)

    project_rows = list_projects(
        user["id"]
    )

    return {
        "projects": [
            dict(row)
            for row in project_rows
        ]
    }


# ============================================================
# VIDEO GENERATION BACKGROUND JOB
# ============================================================

async def process_video_job(
    job_id: str,
    user_id: str,
    topic: str,
    settings: dict,
    project_id: str
):
    set_status(
        job_id,
        JobStatus.PROCESSING
    )

    file_stem = safe_filename(
        topic
    )

    audio_file = (
        BASE_DIR
        / "data"
        / "audio"
        / f"{file_stem}.mp3"
    )

    video_file = (
        BASE_DIR
        / "data"
        / "videos"
        / f"{file_stem}.mp4"
    )

    cost = settings[
        "credit_cost"
    ]

    try:

        # ----------------------------------------------------
        # SCRIPT GENERATION
        # ----------------------------------------------------

        script_text, dialogue = await asyncio.to_thread(
            generate_script,
            topic,
            settings["duration"]
        )

        # ----------------------------------------------------
        # TEXT TO SPEECH
        # ----------------------------------------------------

        timeline = await create_dialogue_voice(
            dialogue,
            str(audio_file),
            voice_profile=settings["voice"]
        )

        # ----------------------------------------------------
        # VIDEO CREATION
        # ----------------------------------------------------

        await asyncio.to_thread(
            build_video,
            str(audio_file),
            str(video_file),
            topic,
            timeline,
            settings["aspect_ratio"],
            settings["captions"],
            settings["duration"]
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        url = (
            f"/videos/{file_stem}.mp4"
        )

        set_result(
            job_id,
            {
                "success": True,
                "topic": topic,
                "script": script_text,
                "video": str(video_file),
                "download_url": url,
            }
        )

        update_project(
            project_id,
            user_id,
            status="completed",
            script=script_text,
            video_url=url
        )

    except Exception as exc:

        traceback.print_exc()

        # Refund credits if generation fails.
        refund_credits(
            user_id,
            cost
        )

        error_message = str(exc)

        set_error(
            job_id,
            error_message
        )

        update_project(
            project_id,
            user_id,
            status="failed",
            error=error_message
        )


# ============================================================
# GENERATE VIDEO
# ============================================================

@app.post("/generate")
@limiter.limit("5/minute")
async def generate_video(
    request: Request,
    body: VideoRequest
):
    user = current_user(request)

    topic = body.topic.strip()

    if not topic:
        raise HTTPException(
            status_code=400,
            detail="Topic cannot be empty."
        )

    normalize_ratio(
        body.aspect_ratio
    )

    normalize_voice(
        body.voice
    )

    cost = credit_cost(
        body.duration
    )

    # --------------------------------------------------------
    # DAILY PLATFORM LIMIT
    # --------------------------------------------------------

    if not check_daily_limit():
        raise HTTPException(
            status_code=429,
            detail=(
                f"Daily platform safety limit of "
                f"{DAILY_GENERATION_LIMIT} reached. "
                f"Try again later."
            )
        )

    # --------------------------------------------------------
    # RESERVE USER CREDITS
    # --------------------------------------------------------

    ok, refreshed_user = reserve_credits(
        user["id"],
        cost
    )

    if not ok:
        raise HTTPException(
            status_code=402,
            detail={
                "code": "credits_exhausted",
                "message": (
                    "You have reached your plan limit. "
                    "Upgrade to continue generating videos."
                ),
                "upgrade_required": True,
            }
        )

    # --------------------------------------------------------
    # CREATE JOB
    # --------------------------------------------------------

    job_id = create_job()

    project_id = uuid.uuid4().hex

    settings = {
        "duration": body.duration,
        "aspect_ratio": body.aspect_ratio,
        "voice": body.voice,
        "captions": body.captions,
        "credit_cost": cost,
    }

    create_project(
        project_id,
        user["id"],
        topic,
        body.duration,
        body.aspect_ratio,
        body.voice,
        body.captions,
        cost
    )

    asyncio.create_task(
        process_video_job(
            job_id,
            user["id"],
            topic,
            settings,
            project_id
        )
    )

    return {
        "job_id": job_id,
        "project_id": project_id,
        "status": "pending",
        "credits": public_user(
            refreshed_user
        )
    }


# ============================================================
# JOB STATUS
# ============================================================

@app.get("/status/{job_id}")
def check_status(
    request: Request,
    job_id: str
):
    current_user(request)

    job = get_job(
        job_id
    )

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found."
        )

    if job["status"] == JobStatus.FAILED:
        return {
            "job_id": job_id,
            "status": job["status"],
            "error": job["error"],
        }

    if job["status"] == JobStatus.COMPLETED:
        return {
            "job_id": job_id,
            "status": job["status"],
            **job["result"],
        }

    return {
        "job_id": job_id,
        "status": job["status"],
    }


# ============================================================
# USAGE
# ============================================================

@app.get("/usage")
def usage(request: Request):
    user = current_user(request)

    return {
        "user": public_user(user),
        "platform_used_today": get_daily_usage(),
        "platform_daily_limit": DAILY_GENERATION_LIMIT,
    }


# ============================================================
# PLANS
# ============================================================

@app.get("/plans")
def plans():
    return {
        "plans": [
            {
                "id": "free",
                "name": "Free",
                "credits": 50,
                "price": 0,
            },
            {
                "id": "pro",
                "name": "Pro",
                "credits": 500,
                "price": None,
            },
        ]
    }


# ============================================================
# DEVELOPMENT PLAN UPGRADE
# ============================================================

@app.post("/subscription/dev-upgrade")
def dev_upgrade(
    request: Request
):
    """
    Development-only plan switch.

    This must remain disabled in production unless
    ALLOW_DEV_UPGRADE=true is explicitly configured.
    """

    if os.getenv(
        "ALLOW_DEV_UPGRADE",
        "false"
    ).lower() != "true":

        raise HTTPException(
            status_code=404,
            detail="Not found."
        )

    user = current_user(
        request
    )

    updated = set_plan(
        user["id"],
        "pro"
    )

    return {
        "user": public_user(
            updated
        )
    }