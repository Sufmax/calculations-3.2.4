"""
pipeline.py — Pipeline 3 threads pour le cache Blender (batches tar.zst + boto3).
Fix Storj 411: Body=bytes (jamais de file object ni multipart).
"""

import logging
import re
import threading
import time
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, List, Optional, Set

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from compression import ZstdDictManager, compress_batch
from config import Config
from progress import ProgressTracker

logger = logging.getLogger(__name__)

CACHE_EXTENSIONS = {
    '.bphys', '.vdb', '.uni', '.gz',
    '.png', '.exr', '.abc', '.obj', '.ply',
}

FRAME_PATTERNS = [
    re.compile(r'_(\d{4,6})_\d+\.bphys$'),
    re.compile(r'_(\d{4,6})\.bphys$'),
    re.compile(r'_(\d{4,6})\.vdb$'),
    re.compile(r'data_(\d{4,6})\.vdb$'),
    re.compile(r'_(\d+)\.\w+$'),
]


def extract_frame_number(filepath: Path) -> Optional[int]:
    name = filepath.name
    for pattern in FRAME_PATTERNS:
        m = pattern.search(name)
        if m:
            return int(m.group(1))
    return None


class _CacheEventHandler(FileSystemEventHandler):
    def __init__(self, watcher: 'FrameWatcher'):
        self._watcher = watcher
        super().__init__()

    def on_created(self, event: FileSystemEvent):
        if not event.is_directory:
            self._watcher._on_file(Path(event.src_path))

    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory:
            self._watcher._on_file(Path(event.src_path))


class FrameWatcher:
    def __init__(self, cache_dir: Path, frame_queue: Queue, progress: ProgressTracker, ws_client, already_secured: Optional[Set[int]] = None):
        self.cache_dir = cache_dir
        self.frame_queue = frame_queue
        self.progress = progress
        self.ws_client = ws_client
        self._seen_files: Set[str] = set()
        self._already_secured = already_secured or set()
        self._observer: Optional[Observer] = None
        self._stop_event = threading.Event()

    def start(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._scan_existing()
        self._observer = Observer()
        self._observer.schedule(_CacheEventHandler(self), str(self.cache_dir), recursive=True)
        self._observer.start()
        logger.info("FrameWatcher démarré")

    def stop(self):
        self._stop_event.set()
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)

    def _scan_existing(self):
        for ext in CACHE_EXTENSIONS:
            for fp in self.cache_dir.rglob(f'*{ext}'):
                if fp.is_file():
                    self._process_file(fp, initial=True)

    def _on_file(self, path: Path):
        if path.suffix.lower() not in CACHE_EXTENSIONS:
            return
        self._process_file(path, initial=False)

    def _process_file(self, path: Path, initial: bool):
        key = str(path)
        if key in self._seen_files:
            return
        self._seen_files.add(key)

        frame = extract_frame_number(path)
        if frame is not None:
            self.progress.register_baked_frame(frame)
            if not initial and self.ws_client and self.ws_client.is_connected():
                self.ws_client.send_threadsafe({
                    'type': 'PROGRESS_BAKED',
                    'frame': frame,
                    'total': self.progress.total_frames,
                    'timestamp': time.time(),
                })
            if frame in self._already_secured:
                return

        if not initial and not self._wait_stable(path):
            return

        self.frame_queue.put(path)

    def _wait_stable(self, path: Path, timeout: float = 3.0) -> bool:
        last_size = -1
        waited = 0.0
        while waited < timeout:
            try:
                size = path.stat().st_size
                if size == last_size and size > 0:
                    return True
                last_size = size
            except OSError:
                return False
            time.sleep(0.3)
            waited += 0.3
        return last_size > 0


class BatchCompressor:
    def __init__(self, cache_dir: Path, frame_queue: Queue, batch_queue: Queue, progress: ProgressTracker, dict_manager: ZstdDictManager, ws_client, work_dir: Path):
        self.cache_dir = cache_dir
        self.frame_queue = frame_queue
        self.batch_queue = batch_queue
        self.progress = progress
        self.dict_manager = dict_manager
        self.ws_client = ws_client
        self.work_dir = work_dir

        self._pending_files: List[Path] = []
        self._pending_frames: List[int] = []
        self._dict_training_samples: List[Path] = []
        self._dict_trained = False

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.batch_size = Config.DEFAULT_BATCH_SIZE

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    def update_batch_size(self):
        speed = self.progress.upload_speed_bps
        ratio = self.progress.compression_ratio
        if speed <= 0 or ratio <= 0:
            return
        confirmed = [b for b in self.progress.batches.values() if b.status == 'confirmed' and len(b.frames) > 0]
        if not confirmed:
            return
        avg_raw = sum(b.raw_size / len(b.frames) for b in confirmed) / len(confirmed)
        if avg_raw <= 0:
            return
        compressed_per_frame = avg_raw / ratio
        optimal = (speed * Config.TARGET_UPLOAD_TIME) / compressed_per_frame
        self.batch_size = max(Config.MIN_BATCH_SIZE, min(Config.MAX_BATCH_SIZE, int(optimal)))

    def flush(self):
        if self._pending_files:
            self._compress_batch()

    def _run(self):
        while not self._stop_event.is_set():
            try:
                try:
                    fp = self.frame_queue.get(timeout=Config.BATCH_INTERVAL)
                    self._add_file(fp)
                except Empty:
                    pass

                while not self.frame_queue.empty():
                    try:
                        fp = self.frame_queue.get_nowait()
                        self._add_file(fp)
                    except Empty:
                        break

                if len(self._pending_files) >= self.batch_size:
                    self._compress_batch()
            except Exception as e:
                logger.error(f"Compressor error: {e}", exc_info=True)
                time.sleep(1.0)

        if self._pending_files:
            self._compress_batch()

    def _add_file(self, path: Path):
        self._pending_files.append(path)
        frame = extract_frame_number(path)
        if frame is not None:
            self._pending_frames.append(frame)
        if not self._dict_trained and len(self._dict_training_samples) < 30:
            self._dict_training_samples.append(path)

    def _compress_batch(self):
        if not self._pending_files:
            return

        if not self._dict_trained and len(self._dict_training_samples) >= Config.ZSTD_MIN_TRAINING_SAMPLES:
            if self.dict_manager.train(self._dict_training_samples):
                self.dict_manager.save_to_file(Config.DICT_FILE)
                self._dict_trained = True

        files = self._pending_files[:]
        frames = self._pending_frames[:]
        self._pending_files.clear()
        self._pending_frames.clear()

        batch = self.progress.create_batch(frames)
        compressed, raw_size = compress_batch(files, self.cache_dir, self.dict_manager)

        self.progress.register_compressed(batch.batch_id, len(compressed), raw_size)

        self.work_dir.mkdir(parents=True, exist_ok=True)
        batch_file = self.work_dir / f"batch_{batch.batch_id:04d}.tar.zst"
        batch_file.write_bytes(compressed)

        if self.ws_client and self.ws_client.is_connected():
            self.ws_client.send_threadsafe({
                'type': 'PROGRESS_COMPRESSED',
                'frames': frames,
                'batchId': batch.batch_id,
                'compressedSize': int(batch_file.stat().st_size),
                'rawSize': int(raw_size),
                'timestamp': time.time(),
            })

        self.batch_queue.put((batch.batch_id, batch_file, frames))
        self.update_batch_size()


class BatchUploader:
    def __init__(self, batch_queue: Queue, progress: ProgressTracker, s3_credentials: Dict, ws_client, cache_prefix: str):
        self.batch_queue = batch_queue
        self.progress = progress
        self.ws_client = ws_client
        self.cache_prefix = cache_prefix

        self._s3 = boto3.client(
            's3',
            endpoint_url=s3_credentials['endpoint'],
            aws_access_key_id=s3_credentials['accessKeyId'],
            aws_secret_access_key=s3_credentials['secretAccessKey'],
            region_name=s3_credentials.get('region', 'us-east-1'),
            config=BotoConfig(
                signature_version='s3v4',
                retries={'max_attempts': 5, 'mode': 'adaptive'},
                s3={'addressing_style': 'path'},
            ),
        )
        self._bucket = s3_credentials['bucket']
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="Uploader")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=30)

    def upload_dict(self, dict_bytes: bytes, work_dir: Path):
        key = f"{self.cache_prefix}dictionary.zstd"
        try:
            self._s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=dict_bytes,
                ContentLength=len(dict_bytes),
                ContentType='application/octet-stream',
                Metadata={'type': 'zstd-dictionary'}
            )
            self._notify_secured(frames=[], batch_id=0, r2_key=key, upload_speed_bps=int(self.progress.upload_speed_bps))
        except Exception as e:
            logger.error(f"Erreur upload dictionnaire: {e}", exc_info=True)

    def _run(self):
        while not self._stop_event.is_set():
            try:
                batch_id, batch_file, frames = self.batch_queue.get(timeout=1.0)
                self._upload_batch(batch_id, batch_file, frames)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Uploader error: {e}", exc_info=True)
                time.sleep(1.0)

    def _upload_batch(self, batch_id: int, batch_file: Path, frames: List[int]):
        key = f"{self.cache_prefix}batch_{batch_id:04d}.tar.zst"
        start = time.time()

        try:
            # ── Fix Storj 411 ─────────────────────────────────────
            # Storj refuse les requêtes sans Content-Length header.
            # Passer un file object comme Body → botocore utilise
            # chunked transfer encoding → pas de Content-Length → 411.
            # Solution : lire en bytes AVANT d'envoyer.
            # Pas de multipart non plus (même problème sur UploadPart).
            # ──────────────────────────────────────────────────────
            data = batch_file.read_bytes()

            self._s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentLength=len(data),
                ContentType='application/octet-stream',
                Metadata={
                    'batch_id': str(batch_id),
                    'frames': ','.join(str(f) for f in frames),
                    'frame_count': str(len(frames)),
                },
            )

            duration = time.time() - start
            self.progress.register_secured(batch_id, key, duration)

            head = self._s3.head_object(Bucket=self._bucket, Key=key)
            etag = str(head.get('ETag') or '').strip('"')

            self._notify_secured(
                frames=frames,
                batch_id=batch_id,
                r2_key=key,
                upload_speed_bps=int(self.progress.upload_speed_bps),
                size=len(data),
                etag=etag
            )

            try:
                batch_file.unlink()
            except OSError:
                pass

        except Exception as e:
            self.progress.register_batch_failed(batch_id)
            logger.error(f"Erreur upload batch #{batch_id}: {e}")

    def _notify_secured(self, frames: List[int], batch_id: int, r2_key: str, upload_speed_bps: int, size: Optional[int] = None, etag: Optional[str] = None):
        if not self.ws_client or not self.ws_client.is_connected():
            return
        msg = {
            'type': 'PROGRESS_SECURED',
            'frames': frames,
            'batchId': batch_id,
            'r2Key': r2_key,
            'uploadSpeedBps': upload_speed_bps,
            'timestamp': time.time(),
        }
        if size is not None:
            msg['size'] = int(size)
        if etag is not None:
            msg['etag'] = etag
        self.ws_client.send_threadsafe(msg)


class Pipeline:
    def __init__(self, cache_dir: Path, ws_client, s3_credentials: Dict, total_frames: int = 250, already_secured: Optional[Set[int]] = None, dict_bytes: Optional[bytes] = None, work_dir: Optional[Path] = None):
        self.cache_dir = cache_dir
        self.ws_client = ws_client
        self.s3_credentials = s3_credentials
        self.cache_prefix = s3_credentials.get('cachePrefix', 'cache/')

        self.work_dir = work_dir or (Path(__file__).parent / 'work' / 'batches')
        self._frame_queue: Queue = Queue()
        self._batch_queue: Queue = Queue()

        self.progress = ProgressTracker(total_frames=total_frames, already_secured=already_secured)

        self.dict_manager = ZstdDictManager()
        if dict_bytes:
            self.dict_manager.load_from_bytes(dict_bytes)
        elif Config.DICT_FILE.exists():
            self.dict_manager.load_from_file(Config.DICT_FILE)

        self.watcher = FrameWatcher(cache_dir, self._frame_queue, self.progress, ws_client, already_secured)
        self.compressor = BatchCompressor(cache_dir, self._frame_queue, self._batch_queue, self.progress, self.dict_manager, ws_client, self.work_dir)
        self.uploader = BatchUploader(self._batch_queue, self.progress, s3_credentials, ws_client, self.cache_prefix)

        self._stop_event = threading.Event()
        self._progress_thread: Optional[threading.Thread] = None

    def start(self):
        self.watcher.start()
        self.compressor.start()
        self.uploader.start()
        self._progress_thread = threading.Thread(target=self._progress_loop, daemon=True)
        self._progress_thread.start()

    def stop(self):
        self._stop_event.set()
        self.watcher.stop()
        self.compressor.stop()
        self.uploader.stop()
        if self._progress_thread:
            self._progress_thread.join(timeout=5)

    def finalize(self):
        self.compressor.flush()
        timeout = 120.0
        waited = 0.0
        while not self._batch_queue.empty() and waited < timeout:
            time.sleep(0.5)
            waited += 0.5
        if self.dict_manager.is_trained and self.dict_manager.dict_bytes:
            self.uploader.upload_dict(self.dict_manager.dict_bytes, self.work_dir)

    def _progress_loop(self):
        while not self._stop_event.is_set():
            time.sleep(Config.PROGRESS_REPORT_INTERVAL)
            status = self.progress.get_status_dict()
            status['currentBatchSize'] = self.compressor.batch_size
            if self.ws_client and self.ws_client.is_connected():
                self.ws_client.send_threadsafe({
                    'type': 'PROGRESS_UPDATE',
                    'uploadPercent': int(status['securedPercent']),
                    'diskBytes': 0,
                    'diskFiles': int(status['bakedFrames']),
                    'uploadedBytes': 0,
                    'uploadedFiles': int(status['securedFrames']),
                    'errors': 0,
                    'rateBytesPerSec': int(status['uploadSpeedBps']),
                    'progress': status,
                })
