import uuid
from enum import Enum
from typing import Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


_jobs: dict[str, dict] = {}


def create_job() -> str:
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {
        "status": JobStatus.PENDING,
        "result": None,
        "error": None,
    }
    return job_id


def set_status(job_id: str, status: JobStatus):
    if job_id in _jobs:
        _jobs[job_id]["status"] = status


def set_result(job_id: str, result: dict):
    if job_id in _jobs:
        _jobs[job_id]["status"] = JobStatus.COMPLETED
        _jobs[job_id]["result"] = result


def set_error(job_id: str, error: str):
    if job_id in _jobs:
        _jobs[job_id]["status"] = JobStatus.FAILED
        _jobs[job_id]["error"] = error


def get_job(job_id: str) -> Optional[dict]:
    return _jobs.get(job_id)