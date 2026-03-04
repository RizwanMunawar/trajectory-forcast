import os
import hashlib
import urllib.request
from pathlib import Path


def download_if_url(source: str, cache_dir: str = ".tf_cache") -> str:
    """
    If source is a URL, download it locally and return local path.
    Otherwise return original source.
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
