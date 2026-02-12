import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration globale"""

    # WebSocket
    WS_URL = os.getenv('WS_URL', 'wss://your-worker.pages.dev/ws/vm')
    VM_PASSWORD = os.getenv('VM_PASSWORD')

    # Chemins
    BASE_DIR = Path(__file__).parent
    WORK_DIR = BASE_DIR / 'work'
    BLEND_FILE = WORK_DIR / 'current.blend'
    CACHE_DIR = WORK_DIR / 'cache'
    DICT_FILE = WORK_DIR / 'zstd_dictionary.dict'

    # Blender
    BLENDER_EXECUTABLE = os.getenv('BLENDER_EXECUTABLE', 'blender')
    BLENDER_SCRIPT = BASE_DIR / 'bake_all.py'

    # Threading pour le bake
    BAKE_THREADS = int(os.getenv('BAKE_THREADS', str(max(1, (os.cpu_count() or 1) - 2))))

    # Timing
    HEARTBEAT_INTERVAL = int(os.getenv('HEARTBEAT_INTERVAL', '3'))

    # Limites reconnexion
    MAX_RECONNECT_ATTEMPTS = int(os.getenv('MAX_RECONNECT_ATTEMPTS', '10'))
    RECONNECT_DELAY = int(os.getenv('RECONNECT_DELAY', '5'))

    # ── Pipeline batch ──
    # Durée cible d'upload par batch (secondes) — le batch size s'adapte pour viser ce temps
    TARGET_UPLOAD_TIME = float(os.getenv('TARGET_UPLOAD_TIME', '20.0'))
    # Limites du batch adaptatif
    MIN_BATCH_SIZE = int(os.getenv('MIN_BATCH_SIZE', '5'))
    MAX_BATCH_SIZE = int(os.getenv('MAX_BATCH_SIZE', '50'))
    DEFAULT_BATCH_SIZE = int(os.getenv('DEFAULT_BATCH_SIZE', '10'))
    # Intervalle de scan batch (secondes)
    BATCH_INTERVAL = float(os.getenv('BATCH_INTERVAL', '2.0'))

    # ── Compression zstd ──
    ZSTD_LEVEL = int(os.getenv('ZSTD_LEVEL', '3'))
    ZSTD_DICT_SIZE = int(os.getenv('ZSTD_DICT_SIZE', str(256 * 1024)))
    # Nombre minimum d'échantillons pour entraîner le dictionnaire
    ZSTD_MIN_TRAINING_SAMPLES = int(os.getenv('ZSTD_MIN_TRAINING_SAMPLES', '10'))

    # ── S3 multipart ──
    S3_MULTIPART_THRESHOLD = int(os.getenv('S3_MULTIPART_THRESHOLD', str(5 * 1024 * 1024)))
    S3_MULTIPART_CHUNK_SIZE = int(os.getenv('S3_MULTIPART_CHUNK_SIZE', str(5 * 1024 * 1024)))

    # ── Progression ──
    PROGRESS_REPORT_INTERVAL = float(os.getenv('PROGRESS_REPORT_INTERVAL', '2.0'))

    @classmethod
    def ensure_dirs(cls):
        cls.WORK_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate(cls):
        if not cls.VM_PASSWORD:
            raise ValueError(
                "VM_PASSWORD non défini. "
                "Définissez-le dans .env ou comme variable d'environnement"
            )
        cls.ensure_dirs()
        return True