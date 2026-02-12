"""
resume.py — Téléchargement et restauration du cache depuis R2.

Quand une nouvelle VM prend le relais, elle doit :
1. Télécharger les batches cache déjà sécurisés
2. Les décompresser dans le dossier cache local
3. Télécharger le dictionnaire zstd si disponible
4. Permettre à Blender de reprendre à la bonne frame
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import boto3
from botocore.config import Config as BotoConfig

from compression import ZstdDictManager, decompress_batch
from utils import format_bytes

logger = logging.getLogger(__name__)


class ResumeManager:
    """Gère la reprise de cache depuis R2 pour une nouvelle VM."""

    def __init__(self, s3_credentials: Dict):
        self._s3 = boto3.client(
            's3',
            endpoint_url=s3_credentials['endpoint'],
            aws_access_key_id=s3_credentials['accessKeyId'],
            aws_secret_access_key=s3_credentials['secretAccessKey'],
            region_name=s3_credentials.get('region', 'us-east-1'),
            config=BotoConfig(
                signature_version='s3v4',
                retries={'max_attempts': 3, 'mode': 'adaptive'},
            ),
        )
        self._bucket = s3_credentials['bucket']

    def download_dictionary(self, dict_key: str, output_path: Path) -> Optional[bytes]:
        """Télécharge le dictionnaire zstd depuis R2."""
        try:
            resp = self._s3.get_object(Bucket=self._bucket, Key=dict_key)
            data = resp['Body'].read()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(data)
            logger.info(f"Dictionnaire téléchargé : {dict_key} ({format_bytes(len(data))})")
            return data
        except self._s3.exceptions.NoSuchKey:
            logger.info("Pas de dictionnaire zstd disponible")
            return None
        except Exception as e:
            logger.error(f"Erreur téléchargement dictionnaire : {e}")
            return None

    def download_batches(
        self,
        batch_keys: List[str],
        cache_dir: Path,
        dict_manager: Optional[ZstdDictManager] = None,
    ) -> Set[int]:
        """
        Télécharge et décompresse les batches cache depuis R2.
        Retourne le set des frames restaurées.
        """
        restored_frames: Set[int] = set()
        cache_dir.mkdir(parents=True, exist_ok=True)

        for key in batch_keys:
            try:
                resp = self._s3.get_object(Bucket=self._bucket, Key=key)
                data = resp['Body'].read()
                logger.info(f"Batch téléchargé : {key} ({format_bytes(len(data))})")

                # Décompresser dans le cache_dir
                extracted = decompress_batch(data, cache_dir, dict_manager)
                logger.info(f"  → {len(extracted)} fichiers extraits")

                # Extraire les numéros de frame depuis les métadonnées
                metadata = resp.get('Metadata', {})
                frames_str = metadata.get('frames', '')
                if frames_str:
                    for f in frames_str.split(','):
                        try:
                            restored_frames.add(int(f.strip()))
                        except ValueError:
                            pass

            except Exception as e:
                logger.error(f"Erreur téléchargement batch {key} : {e}")

        logger.info(
            f"Reprise terminée : {len(restored_frames)} frames restaurées "
            f"depuis {len(batch_keys)} batches"
        )
        return restored_frames

    def download_blend(self, blend_key: str, output_path: Path) -> bool:
        """Télécharge le fichier .blend depuis R2."""
        try:
            resp = self._s3.get_object(Bucket=self._bucket, Key=blend_key)
            data = resp['Body'].read()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(data)
            logger.info(f"Fichier .blend téléchargé : {format_bytes(len(data))}")
            return True
        except Exception as e:
            logger.error(f"Erreur téléchargement .blend : {e}")
            return False