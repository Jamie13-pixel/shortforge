def search_video_clips(query: str, count: int = 2):
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
        chosen = pool_sorted[0]  # smallest available file, not mid-size
        clip_urls.append(chosen["link"])

    return clip_urls