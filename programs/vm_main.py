#!/usr/bin/env python3
"""
Point d'entrée principal du script VM — Pipeline 3 threads.
"""

import asyncio
import logging
import sys
from urllib.request import urlopen
from urllib.error import URLError
from typing import Optional, Dict, Set

from config import Config
from utils import setup_logging
from ws_client import WSClient
from pipeline import Pipeline
from resume import ResumeManager
from compression import ZstdDictManager
from blender_runner import BlenderRunner

logger = logging.getLogger(__name__)

ws_client: Optional[WSClient] = None
pipeline: Optional[Pipeline] = None
blender_runner: Optional[BlenderRunner] = None
heartbeat_task: Optional[asyncio.Task] = None
shutdown_event = asyncio.Event()
_blender_done_event = asyncio.Event()

s3_credentials: Optional[Dict] = None
resume_data: Optional[Dict] = None


async def heartbeat_loop():
    logger.info(f"Démarrage heartbeat (interval: {Config.HEARTBEAT_INTERVAL}s)")
    try:
        while not shutdown_event.is_set():
            if ws_client and ws_client.is_connected():
                await ws_client.send_heartbeat()
            await asyncio.sleep(Config.HEARTBEAT_INTERVAL)
    except asyncio.CancelledError:
        pass


async def on_authenticated(message: dict):
    logger.info("Authentifié.")
    global heartbeat_task
    heartbeat_task = asyncio.create_task(heartbeat_loop())


async def on_message(message: dict):
    global s3_credentials, resume_data
    msg_type = message.get('type')

    if msg_type == 'S3_CREDENTIALS':
        s3_credentials = {
            'endpoint': message.get('endpoint'),
            'bucket': message.get('bucket'),
            'region': message.get('region'),
            'accessKeyId': message.get('accessKeyId'),
            'secretAccessKey': message.get('secretAccessKey'),
            'cachePrefix': message.get('cachePrefix', 'cache/'),
        }
        logger.info(f"Credentials S3 reçues (prefix={s3_credentials['cachePrefix']})")

    elif msg_type == 'RESUME_INFO':
        resume_data = message
        logger.info(
            f"RESUME_INFO: {len(message.get('securedFrames', []))} frames sécurisées, "
            f"reprendre à frame {message.get('resumeFromFrame', 1)}"
        )

    elif msg_type == 'BLEND_FILE_URL':
        await handle_blend_file_url(message)

    elif msg_type == 'TERMINATE':
        reason = message.get('reason', 'Non spécifié')
        logger.warning(f"Demande de terminaison: {reason}")
        await shutdown()


async def handle_blend_file_url(message: dict):
    url = message.get('url')
    if not url:
        return

    logger.info("Téléchargement .blend...")
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, _download_url, url)

        Config.BLEND_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(Config.BLEND_FILE, 'wb') as f:
            f.write(data)

        logger.info(f"Fichier .blend sauvegardé ({len(data)} bytes)")
        asyncio.create_task(start_pipeline())

    except Exception as e:
        logger.error(f"Erreur téléchargement .blend: {e}")


def _download_url(url: str) -> bytes:
    try:
        with urlopen(url, timeout=300) as response:
            return response.read()
    except URLError as e:
        raise RuntimeError(f"Erreur téléchargement: {e}")


async def start_pipeline():
    global pipeline, blender_runner
    await asyncio.sleep(2.0)

    if s3_credentials is None:
        logger.error("Pas de credentials S3 reçues — impossible de démarrer")
        _blender_done_event.set()
        return

    logger.info("═══ Démarrage Pipeline 3 Threads ═══")

    try:
        already_secured: Set[int] = set()
        total_frames = 250
        dict_bytes: Optional[bytes] = None

        if resume_data:
            already_secured = set(resume_data.get('securedFrames', []))
            total_frames = int(resume_data.get('totalFrames', 250))
            batch_entries = resume_data.get('cacheBatches', [])
            dict_key = resume_data.get('zstdDictionaryKey')

            if batch_entries or dict_key:
                logger.info(f"Reprise cache: {len(batch_entries)} batches à télécharger")
                resume_mgr = ResumeManager(s3_credentials)

                if dict_key:
                    dict_bytes = resume_mgr.download_dictionary(dict_key, Config.DICT_FILE)

                if batch_entries:
                    batch_keys = [b.get('key') for b in batch_entries if b.get('key')]
                    dict_mgr = ZstdDictManager()
                    if dict_bytes:
                        dict_mgr.load_from_bytes(dict_bytes)
                    restored = resume_mgr.download_batches(batch_keys, Config.CACHE_DIR, dict_mgr)
                    already_secured.update(restored)

        loop = asyncio.get_event_loop()

        pipeline = Pipeline(
            cache_dir=Config.CACHE_DIR,
            ws_client=ws_client,
            s3_credentials=s3_credentials,
            total_frames=total_frames,
            already_secured=already_secured,
            dict_bytes=dict_bytes,
        )
        pipeline.start()

        blender_runner = BlenderRunner(Config.BLEND_FILE, Config.CACHE_DIR)
        return_code = await blender_runner.run()
        logger.info(f"Blender terminé (code: {return_code})")

        if pipeline:
            await loop.run_in_executor(None, pipeline.finalize)

        if ws_client and ws_client.is_connected():
            await ws_client.send_ready_to_terminate()

    except Exception as e:
        logger.error(f"Erreur pipeline: {e}", exc_info=True)
    finally:
        if pipeline:
            pipeline.stop()
        _blender_done_event.set()


async def shutdown():
    logger.info("Arrêt en cours...")
    shutdown_event.set()

    if not _blender_done_event.is_set():
        try:
            await asyncio.wait_for(_blender_done_event.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            pass

    if blender_runner:
        blender_runner.terminate()

    if pipeline:
        pipeline.stop()

    if heartbeat_task:
        heartbeat_task.cancel()

    if ws_client:
        ws_client.disconnect()

    logger.info("Arrêt terminé")


async def main():
    global ws_client
    setup_logging(logging.INFO)
    logger.info("Blender VM Worker - Pipeline 3 Threads")

    try:
        Config.validate()

        loop = asyncio.get_running_loop()
        try:
            import signal
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))
        except (NotImplementedError, AttributeError):
            pass

        # IMPORTANT : WS_URL doit inclure ?invite=... (cf Router)
        ws_client = WSClient(Config.WS_URL, Config.VM_PASSWORD)
        ws_client.on_authenticated = on_authenticated
        ws_client.on_message = on_message
        ws_client.on_disconnected = lambda: logger.warning("Déconnecté")
        ws_client.on_error = lambda e: logger.error(f"Erreur WS: {e}")

        await ws_client.connect()
        await shutdown_event.wait()

    except KeyboardInterrupt:
        await shutdown()
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        await shutdown()
        return 1

    return 0


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)