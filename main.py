from dotenv import load_dotenv
load_dotenv()

import asyncio
import os
import re
import traceback
import uuid
import sqlite3
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
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
DATA_DIR = BASE_DIR / "data"

VIDEOS_DIR = DATA_DIR / "videos"
AUDIO_DIR = DATA_DIR / "audio"
TEMP_CLIPS_DIR = DATA_DIR / "temp_clips"


# Create required directories.
STATIC_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TEMP_CLIPS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title=APP_NAME)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)


# ------------------------------------------------------------
# CORS
# ------------------------------------------------------------
#
# The dashboard is served by this same FastAPI application.
# Keeping CORS permissive here also allows local development.
#
# For production, you can restrict allow_origins to your
# actual frontend domain.
# ------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

app.mount(
    "/videos",
    StaticFiles(directory=str(VIDEOS_DIR)),
    name="videos",
)


# ============================================================
# DATABASE INITIALIZATION
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
        ge=30,
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
# HELPERS
# ============================================================

def normalize_email(email: str) -> str:
    """
    Normalize email addresses so the same account cannot be
    created using different capitalization or surrounding spaces.
    """
    return email.strip().lower()


def current_user(request: Request):
    """
    Return the authenticated user.

    Raises HTTP 401 if there is no valid session.
    """

    session_token = request.cookies.get(SESSION_COOKIE)

    if not session_token:
        raise HTTPException(
            status_code=401,
            detail="Please log in to continue."
        )

    user = get_user_by_session(session_token)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Your session has expired. Please log in again."
        )

    return user


def safe_filename(topic: str) -> str:
    """
    Create a safe filename from the video topic.
    """

    slug = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "-",
        topic.strip()
    ).strip("-").lower()

    return f"{slug or 'video'}-{uuid.uuid4().hex[:8]}"


def credit_cost(duration: int) -> int:
    """
    Clip Pirate credit pricing.

    30 seconds -> 1 credit
    45 seconds -> 2 credits
    60 seconds -> 3 credits
    """

    if duration <= 30:
        return 1

    if duration <= 45:
        return 2

    return 3


def normalize_ratio(ratio: str) -> str:

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


def normalize_voice(voice: str) -> str:

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


def cookie_settings():
    """
    Cookie configuration.

    For local HTTP development:
        SECURE_COOKIE=false

    For HTTPS production:
        SECURE_COOKIE=true
    """

    secure_cookie = (
        os.getenv(
            "SECURE_COOKIE",
            "false"
        ).lower() == "true"
    )

    return {
        "httponly": True,
        "samesite": "lax",
        "secure": secure_cookie,
        "max_age": 30 * 86400,
        "path": "/",
    }


# ============================================================
# FRONTEND
# ============================================================

@app.get("/")
async def root():

    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "Clip Pirate frontend is missing. "
                "Expected static/index.html."
            )
        )

    return FileResponse(
        str(index_file)
    )


# ============================================================
# AUTHENTICATION
# ============================================================

@app.get("/auth/health")
def auth_health():

    return {
        "ok": True,
        "service": "Clip Pirate .ai authentication",
    }


@app.post("/auth/signup")
def signup(
    body: SignupRequest,
    response: Response,
):

    name = body.name.strip()
    email = normalize_email(str(body.email))

    if not name:
        raise HTTPException(
            status_code=422,
            detail="Name cannot be empty."
        )

    # --------------------------------------------------------
    # Check for existing account.
    # --------------------------------------------------------

    existing_user = get_user_by_email(email)

    if existing_user:

        raise HTTPException(
            status_code=409,
            detail="An account with that email already exists."
        )

    # --------------------------------------------------------
    # Create user.
    # --------------------------------------------------------

    try:

        user_id = create_user(
            name,
            email,
            body.password,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except sqlite3.IntegrityError:

        # Database UNIQUE constraint is the final protection
        # against duplicate accounts.

        raise HTTPException(
            status_code=409,
            detail="An account with that email already exists."
        )

    # --------------------------------------------------------
    # Automatically log the user in after signup.
    # --------------------------------------------------------

    token = create_session(user_id)

    response.set_cookie(
        SESSION_COOKIE,
        token,
        **cookie_settings()
    )

    user = get_user(user_id)

    return {
        "user": public_user(user)
    }


@app.post("/auth/login")
def login(
    body: LoginRequest,
    response: Response,
):

    email = normalize_email(str(body.email))

    user = get_user_by_email(email)

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
        **cookie_settings()
    )

    return {
        "user": public_user(
            get_user(user["id"])
        )
    }


# ------------------------------------------------------------
# Compatibility authentication routes.
# ------------------------------------------------------------

@app.post("/signup")
def signup_alias(
    body: SignupRequest,
    response: Response,
):

    return signup(
        body,
        response
    )


@app.post("/login")
def login_alias(
    body: LoginRequest,
    response: Response,
):

    return login(
        body,
        response
    )


@app.post("/auth/logout")
def logout(
    request: Request,
    response: Response,
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


@app.get("/auth/me")
def me(
    request: Request,
):

    user = current_user(request)

    return {
        "user": public_user(user)
    }


# ============================================================
# USER DATA
# ============================================================

@app.get("/credits")
def credits(
    request: Request,
):

    user = current_user(request)

    return {
        "user": public_user(user)
    }


@app.get("/projects")
def projects(
    request: Request,
):

    user = current_user(request)

    rows = list_projects(
        user["id"]
    )

    return {
        "projects": [
            dict(row)
            for row in rows
        ]
    }


# ============================================================
# VIDEO GENERATION
# ============================================================

async def process_video_job(
    job_id: str,
    user_id: str,
    topic: str,
    settings: dict,
    project_id: str,
):

    set_status(
        job_id,
        JobStatus.PROCESSING
    )

    file_stem = safe_filename(topic)

    audio_file = str(
        AUDIO_DIR / f"{file_stem}.mp3"
    )

    video_file = str(
        VIDEOS_DIR / f"{file_stem}.mp4"
    )

    cost = settings["credit_cost"]

    try:

        # ====================================================
        # STEP 1 — Generate duration-aware script
        # ====================================================

        script_text, dialogue = await asyncio.to_thread(
            generate_script,
            topic,
            settings["duration"],
        )

        # ====================================================
        # STEP 2 — Generate TTS
        # ====================================================
        #
        # This uses the current Clip Pirate TTS interface.
        # The selected voice profile is passed through.
        # ====================================================

        timeline = await create_dialogue_voice(
            dialogue,
            audio_file,
            voice_profile=settings["voice"],
        )

        # ====================================================
        # STEP 3 — Build final video
        # ====================================================

        await asyncio.to_thread(
            build_video,
            audio_file,
            video_file,
            topic,
            timeline,
            settings["aspect_ratio"],
            settings["captions"],
            settings["duration"],
        )

        # ====================================================
        # Verify final output exists.
        # ====================================================

        output_path = Path(video_file)

        if not output_path.exists():

            raise RuntimeError(
                "Video generation completed without producing an MP4."
            )

        if output_path.stat().st_size == 0:

            raise RuntimeError(
                "Generated video file is empty."
            )

        # ====================================================
        # SUCCESS
        # ====================================================

        url = (
            f"/videos/{output_path.name}"
        )

        result = {
            "success": True,
            "topic": topic,
            "script": script_text,
            "video": video_file,
            "download_url": url,
            "duration": settings["duration"],
            "aspect_ratio": settings["aspect_ratio"],
            "voice": settings["voice"],
            "captions": settings["captions"],
            "credit_cost": cost,
        }

        set_result(
            job_id,
            result
        )

        update_project(
            project_id,
            user_id,
            status="completed",
            script=script_text,
            video_url=url,
        )

    except Exception as exc:

        traceback.print_exc()

        # ----------------------------------------------------
        # Refund credits when generation fails.
        # ----------------------------------------------------

        try:

            refund_credits(
                user_id,
                cost
            )

        except Exception:

            traceback.print_exc()

        error_message = str(exc)

        set_error(
            job_id,
            error_message
        )

        update_project(
            project_id,
            user_id,
            status="failed",
            error=error_message,
        )


@app.post("/generate")
@limiter.limit("5/minute")
async def generate_video(
    request: Request,
    body: VideoRequest,
):

    user = current_user(request)

    topic = body.topic.strip()

    if not topic:

        raise HTTPException(
            status_code=400,
            detail="Topic cannot be empty."
        )

    # --------------------------------------------------------
    # Validate settings.
    # --------------------------------------------------------

    aspect_ratio = normalize_ratio(
        body.aspect_ratio
    )

    voice = normalize_voice(
        body.voice
    )

    duration = body.duration

    # --------------------------------------------------------
    # Calculate credits.
    # --------------------------------------------------------

    cost = credit_cost(
        duration
    )

    # --------------------------------------------------------
    # Platform-level safety limit.
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
    # Reserve user's credits.
    # --------------------------------------------------------

    ok, refreshed_user = reserve_credits(
        user["id"],
        cost,
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
    # Create job/project.
    # --------------------------------------------------------

    job_id = create_job()

    project_id = uuid.uuid4().hex

    settings = {
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "voice": voice,
        "captions": body.captions,
        "credit_cost": cost,
    }

    create_project(
        project_id,
        user["id"],
        topic,
        duration,
        aspect_ratio,
        voice,
        body.captions,
        cost,
    )

    # --------------------------------------------------------
    # Start generation.
    # --------------------------------------------------------

    asyncio.create_task(
        process_video_job(
            job_id,
            user["id"],
            topic,
            settings,
            project_id,
        )
    )

    return {
        "job_id": job_id,
        "project_id": project_id,
        "status": "pending",
        "credits": public_user(
            refreshed_user
        ),
    }


# ============================================================
# JOB STATUS
# ============================================================

@app.get("/status/{job_id}")
def check_status(
    request: Request,
    job_id: str,
):

    user = current_user(request)

    job = get_job(job_id)

    if job is None:

        raise HTTPException(
            status_code=404,
            detail="Job not found."
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The current jobs.py needs to expose the owner of a job
    # for a complete ownership check.
    #
    # Until then, only authenticated users can access jobs.
    # --------------------------------------------------------

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
def usage(
    request: Request,
):

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
    request: Request,
):

    if (
        os.getenv(
            "ALLOW_DEV_UPGRADE",
            "false"
        ).lower()
        != "true"
    ):

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