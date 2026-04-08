import os
import sys
import hashlib
import json
import sqlite3
import requests
import glob
from pathlib import Path
import threading
import concurrent.futures
import codecs
import shutil

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: Pillow not found. Image workflow extraction disabled.")

try:
    from mutagen.mp4 import MP4
    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False
    print("Warning: mutagen not found. MP4 tag extraction will rely entirely on av fallback.")

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False
    print("Warning: av (PyAV) not found. MP4 metadata extraction fallback disabled.")

# Add the reference module to the Python path
# Assuming downloader_utils.py is inside nodes/, the root is one level up
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_dir, 'unifiedefficientloader-reference'))
try:
    from unifiedefficientloader import UnifiedSafetensorsLoader
    HAS_UNIFIED_LOADER = True
except ImportError:
    HAS_UNIFIED_LOADER = False
    print("Warning: unifiedefficientloader not found. Safetensors metadata extraction disabled.")

# --- Database Path Logic ---
try:
    import folder_paths
    base_user_dir = folder_paths.get_public_user_directory("default")
    if not base_user_dir:
        base_user_dir = root_dir
except (ImportError, AttributeError):
    base_user_dir = root_dir

PLUGIN_DATA_DIR = os.path.join(base_user_dir, "ComfyUI-ModelUtils")
os.makedirs(PLUGIN_DATA_DIR, exist_ok=True)
DEFAULT_DB_PATH = os.path.join(PLUGIN_DATA_DIR, "cache.db")


# --- Workflow Extraction Logic ---

def unescape_and_parse_nested_json(json_string):
    if not isinstance(json_string, str): return None
    stripped = json_string.strip()
    if not (stripped.startswith('{') or stripped.startswith('[')): return None
    try: return json.loads(stripped)
    except json.JSONDecodeError: pass
    try:
        first_parse = json.loads(stripped)
        if isinstance(first_parse, str): return json.loads(first_parse)
    except json.JSONDecodeError: pass
    try:
        decoded = codecs.decode(stripped, "unicode_escape")
        return json.loads(decoded)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError): pass
    try:
        unescaped = stripped.replace('\\\\"', '\\"').replace('\\"', '"').replace("\\\\", "\\")
        unescaped = unescaped.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
        return json.loads(unescaped)
    except json.JSONDecodeError: pass
    return None

def get_workflow_blob(metadata):
    for key in ['workflow', 'prompt']:
        val = metadata.get(key)
        if val:
            parsed = unescape_and_parse_nested_json(val)
            if parsed: return parsed
    for val in metadata.values():
        parsed = unescape_and_parse_nested_json(val)
        if isinstance(parsed, dict) and (any(k.isdigit() for k in parsed.keys()) or 'nodes' in parsed):
            return parsed
    return None

def process_image_workflow_and_compress(filepath_str):
    if not HAS_PIL: return {"images_processed": 0, "workflows_extracted": 0, "space_saved": 0}
    p = Path(filepath_str)
    if p.suffix.lower() not in ['.png', '.jpg', '.jpeg']: return {"images_processed": 0, "workflows_extracted": 0, "space_saved": 0}

    workflow = None
    actual_format = None
    stats = {"images_processed": 1, "workflows_extracted": 0, "space_saved": 0}
    try:
        with Image.open(p) as img:
            actual_format = img.format
            # CivitAI sometimes serves PNGs with .jpeg extension in URL
            if actual_format == 'PNG':
                workflow = get_workflow_blob(img.info)
                if workflow:
                    # Save workflow
                    json_path = p.with_suffix('.json')
                    if not json_path.exists():
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(workflow, f, indent=2, ensure_ascii=False)
                        print(f"Extracted workflow to {json_path}")
                        stats["workflows_extracted"] = 1

                    # Compress image to WebP
                    webp_path = p.with_suffix('.webp')
                    if not webp_path.exists():
                        orig_size = os.path.getsize(p)
                        img.save(webp_path, 'WEBP', quality=85, method=4)
                        new_size = os.path.getsize(webp_path)
                        stats["space_saved"] = max(0, orig_size - new_size)
                        print(f"Compressed {p.name} (was actually {actual_format}) to {webp_path.name}")

        # Delete original if we created a webp replacement
        if workflow and p.with_suffix('.webp').exists() and actual_format == 'PNG':
            os.remove(p)
            print(f"Deleted original uncompressed image: {p.name}")
    except Exception as e:
        print(f"Failed to process image {p.name}: {e}")
    return stats

# --- Video Workflow Logic ---
def get_tag_value(mp4, keys):
    for key in keys:
        if key in mp4.tags: return mp4.tags[key][0] if mp4.tags[key] else None
    for key in keys:
        search_suffix = key.split(':')[-1].lower()
        for tag_key, val in mp4.tags.items():
            if tag_key.lower().endswith(f":{search_suffix}"): return val[0] if val else None
    return None

def extract_metadata_av(mp4_path):
    if not HAS_AV: return None
    try:
        with av.open(str(mp4_path)) as container:
            return {k.lower(): v for k, v in container.metadata.items()}
    except Exception as e:
        print(f"av extraction error for {mp4_path}: {e}")
        return None

def process_video_workflow(filepath_str):
    p = Path(filepath_str)
    if p.suffix.lower() != '.mp4': return {"images_processed": 0, "workflows_extracted": 0, "space_saved": 0}

    json_path = p.with_suffix('.json')
    if json_path.exists(): return {"images_processed": 0, "workflows_extracted": 0, "space_saved": 0}

    workflow_data = None
    mp4 = None
    stats = {"images_processed": 0, "workflows_extracted": 0, "space_saved": 0}
    if HAS_MUTAGEN:
        try: mp4 = MP4(p)
        except: pass

    if mp4:
        for tag_keys in [['----:QuickTime:Workflow', 'workflow'], ['----:QuickTime:Prompt', 'prompt'], ['©cmt']]:
            val = get_tag_value(mp4, tag_keys)
            if val:
                if isinstance(val, bytes): val = val.decode('utf-8', errors='ignore')
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, dict) and ('prompt' in parsed or any(k.isdigit() for k in parsed.keys()) or 'nodes' in parsed):
                        if 'prompt' in parsed and isinstance(parsed['prompt'], str):
                            try: workflow_data = json.loads(parsed['prompt'])
                            except: workflow_data = parsed
                        else:
                            workflow_data = parsed
                        break
                except json.JSONDecodeError:
                    # Try more aggressive parsing if it's a nested string
                    parsed = unescape_and_parse_nested_json(val)
                    if parsed:
                        workflow_data = parsed
                        break

    if not workflow_data:
        av_data = extract_metadata_av(p)
        if av_data:
            for key in ['workflow', 'prompt', 'comment', 'description']:
                if key in av_data:
                    val = av_data[key]
                    try:
                        parsed = json.loads(val)
                        if isinstance(parsed, dict):
                            if 'prompt' in parsed and isinstance(parsed['prompt'], str):
                                try: workflow_data = json.loads(parsed['prompt'])
                                except: workflow_data = parsed
                            else:
                                workflow_data = parsed
                            break
                    except json.JSONDecodeError:
                        parsed = unescape_and_parse_nested_json(val)
                        if parsed:
                            workflow_data = parsed
                            break

    if workflow_data:
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(workflow_data, f, indent=2, ensure_ascii=False)
            print(f"Extracted workflow to {json_path.name}")
            stats["workflows_extracted"] = 1
        except Exception as e:
            print(f"Error saving workflow for {p.name}: {e}")
    return stats

# --- Constants for Civitai NSFW levels ---
NSFW_LEVELS = {
    "None": 1,
    "Soft": 2,
    "Mature": 4,
    "X": 8,
    "XXX": 16,
    "All": 32  # Anything
}

class HashCache:
    def __init__(self, db_path=None):
        self.db_path = db_path if db_path else DEFAULT_DB_PATH
        self.is_sqlite = self.db_path.endswith('.db') or self.db_path.endswith('.sqlite')
        self.lock = threading.Lock()
        if self.is_sqlite:
            self._init_sqlite()
        else:
            self.cache = self._load_json()

    def _init_sqlite(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_hashes (
                    filepath TEXT PRIMARY KEY,
                    mtime REAL,
                    size INTEGER,
                    hash TEXT
                )
            ''')
            conn.commit()

    def _load_json(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_json(self):
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=4)

    def _get_sqlite_hash(self, filepath, mtime, size):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT mtime, size, hash FROM file_hashes WHERE filepath=?", (filepath,))
            row = cursor.fetchone()
            if row:
                db_mtime, db_size, db_hash = row
                if db_mtime == mtime and db_size == size:
                    return db_hash
        return None

    def _set_sqlite_hash(self, filepath, mtime, size, file_hash):
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO file_hashes (filepath, mtime, size, hash)
                VALUES (?, ?, ?, ?)
            ''', (filepath, mtime, size, file_hash))
            conn.commit()

    def get_hash(self, filepath):
        if not os.path.exists(filepath):
            return None
        mtime = os.path.getmtime(filepath)
        size = os.path.getsize(filepath)
        key = filepath

        with self.lock:
            if self.is_sqlite:
                cached_hash = self._get_sqlite_hash(key, mtime, size)
                if cached_hash: return cached_hash
            else:
                if key in self.cache:
                    entry = self.cache[key]
                    if entry["mtime"] == mtime and entry["size"] == size:
                        return entry["hash"]

        # Calculate new hash (outside the lock to allow parallel hashing)
        print(f"Calculating SHA256 for: {filepath}")
        sha256_hash = self.calculate_sha256(filepath)

        with self.lock:
            if self.is_sqlite:
                self._set_sqlite_hash(key, mtime, size, sha256_hash)
            else:
                self.cache[key] = {
                    "mtime": mtime,
                    "size": size,
                    "hash": sha256_hash
                }
                self._save_json()

        return sha256_hash

    @staticmethod
    def calculate_sha256(filename):
        sha256_hash = hashlib.sha256()
        with open(filename, "rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

class CivitaiAPI:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://civitai.com/api/v1"

    def get_headers(self):
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get_model_info_by_hash(self, file_hash):
        url = f"{self.base_url}/model-versions/by-hash/{file_hash}"
        response = requests.get(url, headers=self.get_headers())
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None
        else:
            response.raise_for_status()

class Downloader:
    @staticmethod
    def download_file(url, target_path, headers=None):
        if headers is None:
            headers = {}

        temp_path = target_path + ".downloading"

        downloaded_size = 0
        if os.path.exists(temp_path):
            downloaded_size = os.path.getsize(temp_path)

        if downloaded_size > 0:
            headers["Range"] = f"bytes={downloaded_size}-"

        response = requests.get(url, headers=headers, stream=True)

        if response.status_code == 416: # Range not satisfiable, file probably changed or completed
            os.remove(temp_path)
            downloaded_size = 0
            if "Range" in headers:
                del headers["Range"]
            response = requests.get(url, headers=headers, stream=True)

        if response.status_code not in [200, 206]:
            print(f"Failed to download from {url}, status code: {response.status_code}")
            return False

        mode = "ab" if response.status_code == 206 else "wb"

        total_size = int(response.headers.get('content-length', 0)) + downloaded_size

        print(f"Downloading: {target_path} ({downloaded_size}/{total_size} bytes)")

        with open(temp_path, mode) as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        os.rename(temp_path, target_path)
        return True

def get_potential_preview_files(base_path):
    exts = ["png", "jpg", "jpeg", "webp", "mp4"]
    files = []
    for ext in exts:
        files.append(f"{base_path}.preview.{ext}")
        files.append(f"{base_path}.{ext}")
    return files

def preview_exists(base_path):
    previews = get_potential_preview_files(base_path)
    for p in previews:
        if os.path.exists(p):
            return True
    return False

def next_example_image_path(base_path, ext):
    i = 0
    while len(glob.glob(f"{base_path}.example.{i}.*")) > 0:
        i += 1
    return f"{base_path}.example.{i}.{ext}"

def filter_images_by_nsfw(images, threshold_level):
    valid_images = []
    for img in images:
        rating = img.get("nsfwLevel", 1)
        if rating <= threshold_level:
            valid_images.append(img)
    return valid_images

def extract_safetensors_metadata(filepath):
    if not HAS_UNIFIED_LOADER:
        return None
    try:
        # Use low_memory=True to only read the header and metadata without loading tensors
        loader = UnifiedSafetensorsLoader(filepath, low_memory=True)
        metadata = loader.metadata()
        loader.close()
        return metadata if metadata else None
    except Exception as e:
        print(f"Failed to read safetensors metadata from {filepath} using unifiedefficientloader: {e}")
        return None

def process_file(filepath, cache, api, nsfw_threshold, max_examples):
    filepath_str = str(filepath)
    base_path = os.path.splitext(filepath_str)[0]
    is_safetensors = filepath_str.lower().endswith(".safetensors")

    stats = {"new_hashes": 0, "images_processed": 0, "workflows_extracted": 0, "space_saved": 0}

    # Check if all targets already exist
    info_path = f"{base_path}.civitai.info"
    st_metadata_path = f"{base_path}.metadata.json"

    has_info = os.path.exists(info_path)
    has_preview = preview_exists(base_path)
    has_st_metadata = not is_safetensors or os.path.exists(st_metadata_path)

    # Process existing images/videos for workflow extraction & compression BEFORE we possibly skip the file
    for p in get_potential_preview_files(base_path):
        if os.path.exists(p):
            ext_lower = p.lower()
            if ext_lower.endswith(('.png', '.jpg', '.jpeg')):
                res = process_image_workflow_and_compress(p)
                for k in stats: stats[k] += res.get(k, 0)
            elif ext_lower.endswith('.mp4'):
                res = process_video_workflow(p)
                for k in stats: stats[k] += res.get(k, 0)

    for ex in glob.glob(f"{base_path}.example.*"):
        ext_lower = ex.lower()
        if ext_lower.endswith(('.png', '.jpg', '.jpeg')):
            res = process_image_workflow_and_compress(ex)
            for k in stats: stats[k] += res.get(k, 0)
        elif ext_lower.endswith('.mp4'):
            res = process_video_workflow(ex)
            for k in stats: stats[k] += res.get(k, 0)

    # If everything exists and we don't want examples or already have enough examples, we might skip API logic
    if has_info and has_preview and has_st_metadata and max_examples == 0:
        print(f"Skipping fully populated model: {filepath_str}")
        return stats

    print(f"Processing model: {filepath_str}")

    # Safetensors built-in metadata extraction
    if is_safetensors and not has_st_metadata:
        metadata = extract_safetensors_metadata(filepath_str)
        if metadata:
            with open(st_metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
            print(f"Saved safetensors metadata to {st_metadata_path}")

    mtime = os.path.getmtime(filepath_str)
    size = os.path.getsize(filepath_str)

    # We want to know if it's a NEW hash or cached
    is_cached = False
    with cache.lock:
        if cache.is_sqlite:
            cached_hash = cache._get_sqlite_hash(filepath_str, mtime, size)
            if cached_hash: is_cached = True
        else:
            if filepath_str in cache.cache:
                entry = cache.cache[filepath_str]
                if entry["mtime"] == mtime and entry["size"] == size:
                    is_cached = True

    file_hash = cache.get_hash(filepath_str)
    if not file_hash:
        return stats

    if not is_cached:
        stats["new_hashes"] += 1

    try:
        model_info = api.get_model_info_by_hash(file_hash)
    except Exception as e:
        print(f"API Error for {filepath_str}: {e}")
        return stats

    if not model_info:
        print(f"Model not found on Civitai: {filepath_str}")
        return stats

    # 1. Save Info
    if not has_info:
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=4)
        print(f"Saved metadata to {info_path}")

    images = model_info.get("images", [])
    valid_images = filter_images_by_nsfw(images, nsfw_threshold)

    if not valid_images:
        print(f"No valid images found matching NSFW criteria for {filepath_str}")
        return stats

    # 2. Save Preview
    if not has_preview and len(valid_images) > 0:
        preview_img = valid_images[0]
        url = preview_img.get("url")
        ext = "jpeg" # Default fallback
        if ".png" in url.lower(): ext = "png"
        elif ".mp4" in url.lower(): ext = "mp4"
        elif ".jpg" in url.lower() or ".jpeg" in url.lower(): ext = "jpeg"

        preview_path = f"{base_path}.preview.{ext}"
        if Downloader.download_file(url, preview_path):
            if ext in ("png", "jpg", "jpeg"):
                res = process_image_workflow_and_compress(preview_path)
                for k in stats: stats[k] += res.get(k, 0)
            elif ext == "mp4":
                res = process_video_workflow(preview_path)
                for k in stats: stats[k] += res.get(k, 0)

    # 3. Save Examples
    if max_examples > 0:
        existing_examples_count = len(glob.glob(f"{base_path}.example.*"))
        # We divide by 2 roughly if there are json files, let's just count unique example indices
        existing_indices = set()
        for f in glob.glob(f"{base_path}.example.*"):
            parts = f.split('.example.')
            if len(parts) > 1:
                idx = parts[1].split('.')[0]
                if idx.isdigit(): existing_indices.add(idx)

        needed_examples = max_examples - len(existing_indices)

        if needed_examples > 0:
            example_images = valid_images[1:max_examples+1]
            # Skip images we probably already downloaded by skipping the first N
            example_images_to_dl = example_images[len(existing_indices):]

            for img in example_images_to_dl:
                url = img.get("url")
                ext = "jpeg"
                if ".png" in url.lower(): ext = "png"
                elif ".mp4" in url.lower(): ext = "mp4"
                elif ".jpg" in url.lower() or ".jpeg" in url.lower(): ext = "jpeg"

                example_path = next_example_image_path(base_path, ext)
                if Downloader.download_file(url, example_path):
                    if ext in ("png", "jpg", "jpeg"):
                        res = process_image_workflow_and_compress(example_path)
                        for k in stats: stats[k] += res.get(k, 0)
                    elif ext == "mp4":
                        res = process_video_workflow(example_path)
                        for k in stats: stats[k] += res.get(k, 0)
    return stats

import numpy as np
from PIL import Image, ImageOps
import torch

def load_image_tensor(path):
    if not os.path.exists(path):
        return None
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        image = img.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return image
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

def get_model_workflows(base_path):
    workflows = []
    # Main workflow
    main_json = f"{base_path}.json"
    if os.path.exists(main_json):
        workflows.append(main_json)

    # Example workflows
    # They are named like base_path.example.N.json
    # or base_path.preview.json (if extracted from preview)
    preview_json = f"{base_path}.preview.json"
    if os.path.exists(preview_json) and preview_json not in workflows:
        workflows.append(preview_json)

    example_jsons = glob.glob(f"{base_path}.example.*.json")
    # Sort them by index
    example_jsons.sort()
    for ej in example_jsons:
        if ej not in workflows:
            workflows.append(ej)

    return workflows

def get_model_metadata_file(base_path):
    # Prefer civitai.info if available
    civitai_info = f"{base_path}.civitai.info"
    if os.path.exists(civitai_info):
        return civitai_info

    # Fallback to metadata.json (safetensors metadata)
    metadata_json = f"{base_path}.metadata.json"
    if os.path.exists(metadata_json):
        return metadata_json

    return None

def scan_and_process(scan_dirs, recursive=True, db_path=None, nsfw_level="All", max_examples=0, api_key=None, threads=4):
    """
    Scans directories for models and fetches their metadata/previews.
    Adapted for direct Python API calls rather than argparse CLI.
    """
    db_path = db_path if db_path else DEFAULT_DB_PATH
    cache = HashCache(db_path)
    api = CivitaiAPI(api_key)

    model_exts = [".safetensors", ".ckpt", ".pt"]

    files_to_process = []
    if isinstance(scan_dirs, str):
        scan_dirs = [scan_dirs]

    for scan_dir in scan_dirs:
        path = Path(scan_dir)
        if recursive:
            for ext in model_exts:
                # rglob includes all directories, so we must filter out hidden ones
                for f in path.rglob(f"*{ext}"):
                    if not any(part.startswith('.') for part in f.parts if part != '.' and part != '..'):
                        files_to_process.append(f)
        else:
            for ext in model_exts:
                for f in path.glob(f"*{ext}"):
                    if not any(part.startswith('.') for part in f.parts if part != '.' and part != '..'):
                        files_to_process.append(f)

    nsfw_threshold = NSFW_LEVELS.get(nsfw_level, NSFW_LEVELS["All"])

    print(f"Found {len(files_to_process)} models to process.")

    global_stats = {
        "total_processed": len(files_to_process),
        "new_hashes": 0,
        "images_processed": 0,
        "workflows_extracted": 0,
        "space_saved": 0
    }

    # Import and use ComfyUI's ProgressBar
    try:
        import comfy.utils
        pbar = comfy.utils.ProgressBar(len(files_to_process))
    except ImportError:
        pbar = None

    if threads > 1:
        print(f"Starting {threads} threads...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(process_file, fp, cache, api, nsfw_threshold, max_examples) for fp in files_to_process]
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    if res:
                        for k in global_stats: global_stats[k] += res.get(k, 0)
                except Exception as e:
                    print(f"Error in thread: {e}")
                if pbar:
                    pbar.update(1)
    else:
        for filepath in files_to_process:
            res = process_file(filepath, cache, api, nsfw_threshold, max_examples)
            if res:
                for k in global_stats: global_stats[k] += res.get(k, 0)
            if pbar:
                pbar.update(1)

    return global_stats
