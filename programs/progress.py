"""
progress.py — Suivi de progression triple pour le pipeline de cache.

Trois niveaux :
  1. Baked     — frames calculées par Blender (sur disque local)
  2. Compressed — frames incluses dans un batch compressé
  3. Secured   — frames confirmées uploadées dans R2/S3
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class BatchInfo:
    """Information sur un batch de frames."""
    batch_id: int
    frames: List[int]
    compressed_size: int = 0
    raw_size: int = 0
    r2_key: str = ''
    upload_duration: float = 0.0
    status: str = 'pending'  # pending | compressing | uploading | confirmed | failed


class ProgressTracker:
    """Suivi de progression triple avec métriques temps réel."""

    def __init__(self, total_frames: int = 250, already_secured: Optional[Set[int]] = None):
        self.total_frames = total_frames

        # Frames par étape
        self.baked_frames: Set[int] = set()
        self.compressed_frames: Set[int] = set()
        self.secured_frames: Set[int] = set(already_secured or set())

        # Batches
        self.batches: Dict[int, BatchInfo] = {}
        self._next_batch_id = 1

        # Métriques
        self.upload_speed_bps: float = 0.0
        self.compression_ratio: float = 4.0  # estimation initiale
        self.baking_speed_fps: float = 0.0

        # Pour calcul vitesse de bake
        self._bake_count_window: List[float] = []
        self._last_bake_time: float = time.time()

    # ── Propriétés de progression ──

    @property
    def baked_percent(self) -> float:
        if self.total_frames <= 0:
            return 0.0
        return min(100.0, len(self.baked_frames) / self.total_frames * 100)

    @property
    def compressed_percent(self) -> float:
        if self.total_frames <= 0:
            return 0.0
        return min(100.0, len(self.compressed_frames) / self.total_frames * 100)

    @property
    def secured_percent(self) -> float:
        if self.total_frames <= 0:
            return 0.0
        return min(100.0, len(self.secured_frames) / self.total_frames * 100)

    @property
    def last_baked_frame(self) -> int:
        return max(self.baked_frames) if self.baked_frames else 0

    @property
    def last_secured_frame(self) -> int:
        return max(self.secured_frames) if self.secured_frames else 0

    # ── ETA ──

    @property
    def eta_baking(self) -> float:
        """Secondes restantes pour finir le bake."""
        remaining = self.total_frames - len(self.baked_frames)
        if self.baking_speed_fps <= 0 or remaining <= 0:
            return 0.0
        return remaining / self.baking_speed_fps

    @property
    def eta_secured(self) -> float:
        """Secondes restantes pour tout sécuriser."""
        remaining = self.total_frames - len(self.secured_frames)
        if remaining <= 0:
            return 0.0
        # Estimation basée sur la taille compressée moyenne par frame
        confirmed = [b for b in self.batches.values() if b.status == 'confirmed']
        if not confirmed or self.upload_speed_bps <= 0:
            return remaining * 2.0  # estimation grossière
        avg_compressed_per_frame = sum(
            b.compressed_size / max(len(b.frames), 1) for b in confirmed
        ) / len(confirmed)
        total_remaining_bytes = remaining * avg_compressed_per_frame
        return total_remaining_bytes / self.upload_speed_bps

    # ── Enregistrement ──

    def register_baked_frame(self, frame: int):
        """Enregistre qu'une frame a été calculée par Blender."""
        self.baked_frames.add(frame)
        now = time.time()
        self._bake_count_window.append(now)
        # Garder une fenêtre de 5 secondes pour calculer la vitesse
        cutoff = now - 5.0
        self._bake_count_window = [t for t in self._bake_count_window if t > cutoff]
        if len(self._bake_count_window) >= 2:
            elapsed = self._bake_count_window[-1] - self._bake_count_window[0]
            if elapsed > 0:
                self.baking_speed_fps = (len(self._bake_count_window) - 1) / elapsed

    def create_batch(self, frames: List[int]) -> BatchInfo:
        """Crée un nouveau batch."""
        batch = BatchInfo(
            batch_id=self._next_batch_id,
            frames=list(frames),
            status='compressing',
        )
        self.batches[batch.batch_id] = batch
        self._next_batch_id += 1
        return batch

    def register_compressed(self, batch_id: int, compressed_size: int, raw_size: int):
        """Enregistre la compression d'un batch."""
        batch = self.batches.get(batch_id)
        if not batch:
            return
        batch.compressed_size = compressed_size
        batch.raw_size = raw_size
        batch.status = 'uploading'
        self.compressed_frames.update(batch.frames)
        if raw_size > 0 and compressed_size > 0:
            self.compression_ratio = raw_size / compressed_size

    def register_secured(
        self,
        batch_id: int,
        r2_key: str,
        upload_duration: float,
    ):
        """Enregistre qu'un batch est confirmé dans R2."""
        batch = self.batches.get(batch_id)
        if not batch:
            return
        batch.r2_key = r2_key
        batch.upload_duration = upload_duration
        batch.status = 'confirmed'
        self.secured_frames.update(batch.frames)
        # Mettre à jour la vitesse d'upload
        if upload_duration > 0 and batch.compressed_size > 0:
            self.upload_speed_bps = batch.compressed_size / upload_duration

    def register_batch_failed(self, batch_id: int):
        """Marque un batch comme échoué."""
        batch = self.batches.get(batch_id)
        if batch:
            batch.status = 'failed'
            # Retirer les frames du set compressed car non sécurisées
            self.compressed_frames -= set(batch.frames)

    # ── Sérialisation ──

    def get_status_dict(self) -> dict:
        """Retourne un dict complet pour envoi via WebSocket."""
        recent_batches = sorted(
            self.batches.values(),
            key=lambda b: b.batch_id,
            reverse=True,
        )[:10]

        return {
            'totalFrames': self.total_frames,
            'bakedFrames': len(self.baked_frames),
            'bakedPercent': round(self.baked_percent, 1),
            'lastBakedFrame': self.last_baked_frame,
            'compressedFrames': len(self.compressed_frames),
            'compressedPercent': round(self.compressed_percent, 1),
            'securedFrames': len(self.secured_frames),
            'securedPercent': round(self.secured_percent, 1),
            'lastSecuredFrame': self.last_secured_frame,
            'uploadSpeedBps': round(self.upload_speed_bps),
            'compressionRatio': round(self.compression_ratio, 1),
            'bakingSpeedFps': round(self.baking_speed_fps, 2),
            'etaBaking': round(self.eta_baking, 1),
            'etaSecured': round(self.eta_secured, 1),
            'currentBatchSize': 0,  # rempli par le pipeline
            'batches': [
                {
                    'id': b.batch_id,
                    'frames': b.frames,
                    'compressedSize': b.compressed_size,
                    'rawSize': b.raw_size,
                    'r2Key': b.r2_key,
                    'status': b.status,
                }
                for b in recent_batches
            ],
        }