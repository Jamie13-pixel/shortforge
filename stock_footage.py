import os
import time
import requests

PEXELS_API_KEY = os.environ["PEXELS_API_KEY"]
HEADERS = {"Authorization": PEXELS_API_KEY}


def _query_pexels(query: str, count: int):
    url = "https://api.pexels.com/videos/search"
    params = {
        "query": query,
        "orientation": "portrait",
        "per_page": count,
    }
    response = requests.get(url, headers=HEADERS, params=params, timeout=15)
    response.raise_for_status()
    return response.json().get("videos", [])


def search_video_clips(query: str, count: int = 3):
    query = (query or "").strip() or "abstract background"

    try:
        videos = _query_pexels(query, count)
    except Exception as e:
        print(f"[stock_footage] Pexels search failed for '{query}': {e}")
        videos = []

    if not videos:
        print(f"[stock_footage] No results for '{query}', falling back to generic footage")
        try:
            videos = _query_pexels("abstract background", count)
        except Exception as e:
            print(f"[stock_footage] Fallback search also failed: {e}")
            videos = []

    clip_urls = []
    for video in videos:
        files = video["video_files"]
        portrait_files = [f for f in files if f.get("height", 0) > f.get("width", 0)]
        pool = portrait_files or files
        pool_sorted = sorted(pool, key=lambda f: f.get("width", 0))
        mid_index = min(len(pool_sorted) - 1, len(pool_sorted) // 2)
        chosen = pool_sorted[mid_index]
        clip_urls.append(chosen["link"])

    return clip_urls


def download_clip(url: str, dest_path: str, retries: int = 3, timeout: int = 60):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
            return dest_path
        except Exception as e:
            last_error = e
            print(f"[stock_footage] Download attempt {attempt}/{retries} failed for {url}: {e}")
            if os.path.exists(dest_path):
                try:
                    os.remove(dest_path)
                except Exception:
                    pass
            time.sleep(1.5 * attempt)  # brief backoff before retrying

    raise ConnectionError(f"Failed to download {url} after {retries} attempts: {last_error}")