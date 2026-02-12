"""
Configuration pour le script VM — Pipeline 3 threads
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _get_int_env(name, default):
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return int(val)


def _get_float_env(name, default):
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return float(val)


class Config:
    # WebSocket
    WS_URL = os.getenv('WS_URL', 'wss://your-worker.workers.dev/ws/vm')
    VM_PASSWORD = os.getenv('VM_PASSWORD')

    BASE_DIR = Path(__file__).parent
    WORK_DIR = BASE_DIR / 'work'
    BLEND_FILE = WORK_DIR / 'current.blend'
    CACHE_DIR = WORK_DIR / 'cache'
    DICT_FILE = WORK_DIR / 'zstd_dictionary.dict'

    BLENDER_EXECUTABLE = os.getenv('BLENDER_EXECUTABLE', 'blender')
    BLENDER_SCRIPT = BASE_DIR / 'bake_all.py'

    BAKE_THREADS = _get_int_env(
        'BAKE_THREADS',
        max(1, (os.cpu_count() or 1) - 2)
    )

    HEARTBEAT_INTERVAL = _get_int_env('HEARTBEAT_INTERVAL', 3)

    MAX_RECONNECT_ATTEMPTS = _get_int_env('MAX_RECONNECT_ATTEMPTS', 10)
    RECONNECT_DELAY = _get_int_env('RECONNECT_DELAY', 5)

    TARGET_UPLOAD_TIME = _get_float_env('TARGET_UPLOAD_TIME', 20.0)
    MIN_BATCH_SIZE = _get_int_env('MIN_BATCH_SIZE', 5)
    MAX_BATCH_SIZE = _get_int_env('MAX_BATCH_SIZE', 50)
    DEFAULT_BATCH_SIZE = _get_int_env('DEFAULT_BATCH_SIZE', 10)
    BATCH_INTERVAL = _get_float_env('BATCH_INTERVAL', 2.0)

    ZSTD_LEVEL = _get_int_env('ZSTD_LEVEL', 3)
    ZSTD_DICT_SIZE = _get_int_env('ZSTD_DICT_SIZE', 256 * 1024)
    ZSTD_MIN_TRAINING_SAMPLES = _get_int_env('ZSTD_MIN_TRAINING_SAMPLES', 10)

    S3_MULTIPART_THRESHOLD = _get_int_env(
        'S3_MULTIPART_THRESHOLD',
        5 * 1024 * 1024
    )
    S3_MULTIPART_CHUNK_SIZE = _get_int_env(
        'S3_MULTIPART_CHUNK_SIZE',
        5 * 1024 * 1024
    )

    PROGRESS_REPORT_INTERVAL = _get_float_env(
        'PROGRESS_REPORT_INTERVAL',
        2.0
    )

    @classmethod
    def ensure_dirs(cls):
        cls.WORK_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate(cls):
        if not cls.VM_PASSWORD:
            raise ValueError(
                "VM_PASSWORD non défini. Définissez-le dans .env ou variable d'environnement."
            )
        cls.ensure_dirs()
        return True
