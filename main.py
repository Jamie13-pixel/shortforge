from dotenv import load_dotenv
load_dotenv()

import asyncio
import os
import re
import traceback
import uuid
from datetime import datetime

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
from jobs import create_job, set_status, set_result, set_error, get_job, JobStatus
from rate_limiter import limiter, check_daily_limit, get_daily_usage, DAILY_GENERATION_LIMIT
from db import (
    init_db, create_user, get_user_by_email, get_user, create_session, get_user_by_session,
    delete_session, verify_password, reserve_credits, refund_credits, set_plan,
    create_project, update_project, list_projects, public_user
)

APP_NAME = 'Clip Pirate .ai'
SESSION_COOKIE = 'clip_pirate_session'

app = FastAPI(title=APP_NAME)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'], allow_credentials=True)

os.makedirs('data/videos', exist_ok=True)
os.makedirs('data/audio', exist_ok=True)
os.makedirs('data/temp_clips', exist_ok=True)
app.mount('/videos', StaticFiles(directory='data/videos'), name='videos')
app.mount('/app', StaticFiles(directory='static', html=True), name='static')
init_db()


class SignupRequest(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class VideoRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=300)
    duration: int = Field(default=30, ge=15, le=60)
    aspect_ratio: str = Field(default='4:3')
    voice: str = Field(default='professional')
    captions: bool = True


def current_user(request: Request):
    user = get_user_by_session(request.cookies.get(SESSION_COOKIE))
    if not user:
        raise HTTPException(status_code=401, detail='Please log in to continue.')
    return user


def safe_filename(topic: str) -> str:
    slug = re.sub(r'[^a-zA-Z0-9_-]+', '-', topic.strip()).strip('-').lower()
    return f'{slug or "video"}-{uuid.uuid4().hex[:8]}'


def credit_cost(duration: int) -> int:
    if duration <= 30:
        return 1
    if duration <= 45:
        return 2
    return 3


def normalize_ratio(ratio: str):
    allowed = {'4:3', '9:16', '16:9', '1:1'}
    if ratio not in allowed:
        raise HTTPException(status_code=422, detail='Unsupported aspect ratio.')
    return ratio


def normalize_voice(voice: str):
    allowed = {'professional', 'energetic', 'calm'}
    if voice not in allowed:
        raise HTTPException(status_code=422, detail='Unsupported voice.')
    return voice


@app.get('/')
def home():
    return FileResponse('index.html')


@app.post('/auth/signup')
def signup(body: SignupRequest, response: Response):
    if get_user_by_email(body.email):
        raise HTTPException(status_code=409, detail='An account with that email already exists.')
    user_id = create_user(body.name, body.email, body.password)
    token = create_session(user_id)
    response.set_cookie(SESSION_COOKIE, token, httponly=True, samesite='lax', secure=False, max_age=30*86400, path='/')
    return {'user': public_user(get_user(user_id))}


@app.post('/auth/login')
def login(body: LoginRequest, response: Response):
    user = get_user_by_email(body.email)
    if not user or not verify_password(body.password, user['password_hash']):
        raise HTTPException(status_code=401, detail='Invalid email or password.')
    token = create_session(user['id'])
    response.set_cookie(SESSION_COOKIE, token, httponly=True, samesite='lax', secure=False, max_age=30*86400, path='/')
    return {'user': public_user(get_user(user['id']))}


@app.post('/auth/logout')
def logout(request: Request, response: Response):
    delete_session(request.cookies.get(SESSION_COOKIE))
    response.delete_cookie(SESSION_COOKIE, path='/')
    return {'ok': True}


@app.get('/auth/me')
def me(request: Request):
    return {'user': public_user(current_user(request))}


@app.get('/credits')
def credits(request: Request):
    return {'user': public_user(current_user(request))}


@app.get('/projects')
def projects(request: Request):
    user = current_user(request)
    return {'projects': [dict(row) for row in list_projects(user['id'])]}


async def process_video_job(job_id: str, user_id: str, topic: str, settings: dict, project_id: str):
    set_status(job_id, JobStatus.PROCESSING)
    file_stem = safe_filename(topic)
    audio_file = f'data/audio/{file_stem}.mp3'
    video_file = f'data/videos/{file_stem}.mp4'
    cost = settings['credit_cost']
    try:
        script_text, dialogue = await asyncio.to_thread(generate_script, topic)
        # The existing TTS module is preserved here. Voice-specific TTS requires its
        # implementation to expose a voice parameter; this call remains compatible.
        timeline = await create_dialogue_voice(dialogue, audio_file)
        await asyncio.to_thread(build_video, audio_file, video_file, topic, timeline)

        # Duration/aspect-ratio enforcement is applied after the existing builder.
        # The builder remains the source of the visual pipeline; this adapter keeps
        # the existing function signature compatible while we upgrade the renderer.
        try:
            from moviepy import VideoFileClip
            clip = VideoFileClip(video_file)
            target_duration = settings['duration']
            if clip.duration > target_duration:
                clip = clip.subclipped(0, target_duration)
            ratios = {'4:3': (4,3), '9:16': (9,16), '16:9': (16,9), '1:1': (1,1)}
            tw, th = ratios[settings['aspect_ratio']]
            target_ratio = tw / th
            current_ratio = clip.w / clip.h
            if abs(current_ratio - target_ratio) > 0.01:
                if current_ratio > target_ratio:
                    new_w = int(clip.h * target_ratio)
                    x1 = int((clip.w-new_w)/2)
                    clip = clip.cropped(x1=x1, x2=x1+new_w)
                else:
                    new_h = int(clip.w / target_ratio)
                    y1 = int((clip.h-new_h)/2)
                    clip = clip.cropped(y1=y1, y2=y1+new_h)
            tmp = video_file + '.tmp.mp4'
            clip.write_videofile(tmp, codec='libx264', audio_codec='aac', logger=None)
            clip.close()
            os.replace(tmp, video_file)
        except Exception:
            # Do not fail a successful legacy render solely because optional post-processing fails.
            traceback.print_exc()

        url = f'/videos/{file_stem}.mp4'
        set_result(job_id, {'success': True, 'topic': topic, 'script': script_text, 'video': video_file, 'download_url': url})
        update_project(project_id, user_id, status='completed', script=script_text, video_url=url)
    except Exception as e:
        traceback.print_exc()
        refund_credits(user_id, cost)
        set_error(job_id, str(e))
        update_project(project_id, user_id, status='failed', error=str(e))


@app.post('/generate')
@limiter.limit('5/minute')
async def generate_video(request: Request, body: VideoRequest):
    user = current_user(request)
    topic = body.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail='Topic cannot be empty.')
    normalize_ratio(body.aspect_ratio)
    normalize_voice(body.voice)
    cost = credit_cost(body.duration)

    if not check_daily_limit():
        raise HTTPException(status_code=429, detail=f'Daily platform safety limit of {DAILY_GENERATION_LIMIT} reached. Try again later.')

    ok, refreshed_user = reserve_credits(user['id'], cost)
    if not ok:
        raise HTTPException(status_code=402, detail={'code':'credits_exhausted','message':'You have reached your plan limit. Upgrade to continue generating videos.','upgrade_required':True})

    job_id = create_job()
    project_id = uuid.uuid4().hex
    settings = {'duration': body.duration, 'aspect_ratio': body.aspect_ratio, 'voice': body.voice, 'captions': body.captions, 'credit_cost': cost}
    create_project(project_id, user['id'], topic, body.duration, body.aspect_ratio, body.voice, body.captions, cost)
    asyncio.create_task(process_video_job(job_id, user['id'], topic, settings, project_id))
    return {'job_id': job_id, 'project_id': project_id, 'status': 'pending', 'credits': public_user(refreshed_user)}


@app.get('/status/{job_id}')
def check_status(request: Request, job_id: str):
    current_user(request)
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job not found.')
    if job['status'] == JobStatus.FAILED:
        return {'job_id': job_id, 'status': job['status'], 'error': job['error']}
    if job['status'] == JobStatus.COMPLETED:
        return {'job_id': job_id, 'status': job['status'], **job['result']}
    return {'job_id': job_id, 'status': job['status']}


@app.get('/usage')
def usage(request: Request):
    user = current_user(request)
    return {'user': public_user(user), 'platform_used_today': get_daily_usage(), 'platform_daily_limit': DAILY_GENERATION_LIMIT}


@app.get('/plans')
def plans():
    return {'plans': [
        {'id':'free','name':'Free','credits':50,'price':0},
        {'id':'pro','name':'Pro','credits':500,'price':None}
    ]}


# Development-only plan switch. Replace this endpoint with a payment provider webhook
# before accepting real money. It exists now so the subscription UI and entitlements
# can be exercised end-to-end without pretending a payment occurred.
@app.post('/subscription/dev-upgrade')
def dev_upgrade(request: Request):
    if os.getenv('ALLOW_DEV_UPGRADE', 'false').lower() != 'true':
        raise HTTPException(status_code=404, detail='Not found.')
    user = current_user(request)
    updated = set_plan(user['id'], 'pro')
    return {'user': public_user(updated)}
