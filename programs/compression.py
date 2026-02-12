"""
compression.py — Compression zstd avec dictionnaire pour cache Blender.

Le dictionnaire capture les motifs récurrents entre frames de simulation,
améliorant le ratio de compression de x3-5 (sans dict) à x6-10 (avec dict).
"""

import io
import logging
import tarfile
from pathlib import Path
from typing import List, Optional, Tuple

import zstandard as zstd

from config import Config

logger = logging.getLogger(__name__)


class ZstdDictManager:
    """Gère le dictionnaire zstd pour la compression inter-frames."""

    def __init__(self):
        self._dict_data: Optional[zstd.ZstdCompressionDict] = None
        self._dict_bytes: Optional[bytes] = None
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def dict_bytes(self) -> Optional[bytes]:
        return self._dict_bytes

    def train(self, sample_files: List[Path]) -> bool:
        """Entraîne le dictionnaire sur un ensemble de fichiers d'échantillon."""
        if len(sample_files) < Config.ZSTD_MIN_TRAINING_SAMPLES:
            logger.warning(
                f"Pas assez d'échantillons pour entraîner le dictionnaire "
                f"({len(sample_files)}/{Config.ZSTD_MIN_TRAINING_SAMPLES})"
            )
            return False

        samples: List[bytes] = []
        for f in sample_files:
            try:
                data = f.read_bytes()
                if len(data) > 0:
                    samples.append(data)
            except OSError as e:
                logger.warning(f"Impossible de lire {f}: {e}")

        if len(samples) < Config.ZSTD_MIN_TRAINING_SAMPLES:
            return False

        try:
            dict_data = zstd.train_dictionary(
                Config.ZSTD_DICT_SIZE,
                samples,
            )
            self._dict_data = dict_data
            self._dict_bytes = dict_data.as_bytes()
            self._trained = True
            logger.info(
                f"Dictionnaire zstd entraîné : {len(self._dict_bytes)} octets "
                f"à partir de {len(samples)} échantillons"
            )
            return True
        except Exception as e:
            logger.error(f"Erreur entraînement dictionnaire : {e}")
            return False

    def load_from_bytes(self, data: bytes) -> bool:
        """Charge un dictionnaire depuis des bytes bruts."""
        try:
            self._dict_data = zstd.ZstdCompressionDict(data)
            self._dict_bytes = data
            self._trained = True
            logger.info(f"Dictionnaire zstd chargé : {len(data)} octets")
            return True
        except Exception as e:
            logger.error(f"Erreur chargement dictionnaire : {e}")
            return False

    def load_from_file(self, path: Path) -> bool:
        """Charge un dictionnaire depuis un fichier."""
        if not path.exists():
            return False
        try:
            return self.load_from_bytes(path.read_bytes())
        except OSError as e:
            logger.error(f"Erreur lecture dictionnaire {path}: {e}")
            return False

    def save_to_file(self, path: Path) -> bool:
        """Sauvegarde le dictionnaire dans un fichier."""
        if not self._dict_bytes:
            return False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self._dict_bytes)
            logger.info(f"Dictionnaire sauvegardé : {path}")
            return True
        except OSError as e:
            logger.error(f"Erreur sauvegarde dictionnaire : {e}")
            return False

    def get_compressor(self) -> zstd.ZstdCompressor:
        """Retourne un compresseur configuré avec ou sans dictionnaire."""
        if self._dict_data:
            return zstd.ZstdCompressor(
                dict_data=self._dict_data,
                level=Config.ZSTD_LEVEL,
            )
        return zstd.ZstdCompressor(level=Config.ZSTD_LEVEL)

    def get_decompressor(self) -> zstd.ZstdDecompressor:
        """Retourne un décompresseur configuré avec ou sans dictionnaire."""
        if self._dict_data:
            return zstd.ZstdDecompressor(dict_data=self._dict_data)
        return zstd.ZstdDecompressor()


def compress_batch(
    files: List[Path],
    cache_dir: Path,
    dict_manager: Optional[ZstdDictManager] = None,
) -> Tuple[bytes, int]:
    """
    Compresse un batch de fichiers en tar.zst.
    Retourne (données_compressées, taille_brute).
    """
    # Créer le tar en mémoire
    tar_buffer = io.BytesIO()
    raw_size = 0

    with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
        for f in files:
            if not f.exists():
                continue
            try:
                # Chemin relatif au cache_dir pour la reconstruction
                arcname = str(f.relative_to(cache_dir))
                tar.add(str(f), arcname=arcname)
                raw_size += f.stat().st_size
            except (OSError, ValueError) as e:
                logger.warning(f"Impossible d'ajouter {f} au tar : {e}")

    tar_bytes = tar_buffer.getvalue()

    # Compresser avec zstd
    if dict_manager and dict_manager.is_trained:
        compressor = dict_manager.get_compressor()
    else:
        compressor = zstd.ZstdCompressor(level=Config.ZSTD_LEVEL)

    compressed = compressor.compress(tar_bytes)

    ratio = raw_size / len(compressed) if len(compressed) > 0 else 1.0
    logger.debug(
        f"Batch compressé : {raw_size} → {len(compressed)} octets "
        f"(ratio x{ratio:.1f})"
    )

    return compressed, raw_size


def decompress_batch(
    data: bytes,
    output_dir: Path,
    dict_manager: Optional[ZstdDictManager] = None,
) -> List[Path]:
    """
    Décompresse un tar.zst dans output_dir.
    Retourne la liste des fichiers extraits.
    """
    if dict_manager and dict_manager.is_trained:
        decompressor = dict_manager.get_decompressor()
    else:
        decompressor = zstd.ZstdDecompressor()

    tar_bytes = decompressor.decompress(data)

    extracted: List[Path] = []
    tar_buffer = io.BytesIO(tar_bytes)

    with tarfile.open(fileobj=tar_buffer, mode='r') as tar:
        for member in tar.getmembers():
            if member.isfile():
                # Sécurité : vérifier pas de path traversal
                if '..' in member.name or member.name.startswith('/'):
                    logger.warning(f"Chemin suspect ignoré : {member.name}")
                    continue
                tar.extract(member, path=str(output_dir))
                extracted.append(output_dir / member.name)

    logger.info(f"Batch décompressé : {len(extracted)} fichiers dans {output_dir}")
    return extracted