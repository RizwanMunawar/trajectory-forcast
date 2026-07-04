import os
import hashlib
import urllib.request
from pathlib import Path


def download_if_url(source: str, cache_dir: str = ".tf_cache") -> str:
    """Download the source if it is a URL, otherwise return it unchanged.

    Downloads are cached by URL so the same video is not fetched twice.

    Args:
        source (str): Local path or http(s) URL to the video.
        cache_dir (str): Folder where downloaded videos are cached.

    Returns:
        str: A local file path to the video.
    """
    if not source.startswith(("http://", "https://")):
        return source

    os.makedirs(cache_dir, exist_ok=True)

    # create deterministic filename
    url_hash = hashlib.md5(source.encode()).hexdigest()
    filename = source.split("/")[-1]
    local_path = Path(cache_dir) / f"{url_hash}_{filename}"

    if local_path.exists():
        return str(local_path)

    print(f"Downloading sample video from {source}...")
    urllib.request.urlretrieve(source, local_path)
    print(f"Saved to {local_path}")

    return str(local_path)
