"""
Client WebSocket robuste pour Blender Coordinator.
Supporte envoi thread-safe depuis les threads du pipeline.
"""

import asyncio
import json
import logging
import time
from typing import Callable, Optional, Any, Dict
import websockets
from websockets.client import WebSocketClientProtocol

from config import Config

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 2


class WSClient:
    def __init__(self, url: str, password: str):
        self.url = url
        self.password = password
        self.ws: Optional[WebSocketClientProtocol] = None
        self.token: Optional[str] = None
        self.reconnect_attempts = 0
        self.is_running = False
        self.is_authenticated = False

        self.on_authenticated: Optional[Callable] = None
        self.on_message: Optional[Callable[[dict], Any]] = None
        self.on_disconnected: Optional[Callable] = None
        self.on_error: Optional[Callable[[Exception], None]] = None

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server_time_delta_ms: int = 0

    async def connect(self):
        self.is_running = True
        self._loop = asyncio.get_running_loop()

        while self.is_running:
            try:
                logger.info(f"Connexion à {self.url}...")

                async with websockets.connect(
                    self.url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10,
                    max_size=10 * 1024 * 1024
                ) as ws:
                    self.ws = ws
                    self.reconnect_attempts = 0
                    logger.info("Connecté au serveur")

                    await self.authenticate()
                    await self.receive_loop()

            except (websockets.exceptions.ConnectionClosed, OSError) as e:
                logger.warning(f"Connexion fermée/perdue: {e}")
            except Exception as e:
                logger.error(f"Erreur connexion: {e}", exc_info=True)
                if self.on_error:
                    self.on_error(e)

            self.is_authenticated = False

            if self.is_running:
                self.reconnect_attempts += 1
                delay = min(30, Config.RECONNECT_DELAY * self.reconnect_attempts)
                logger.info(f"Reconnexion dans {delay}s (tentative {self.reconnect_attempts})")
                await asyncio.sleep(delay)

        if self.on_disconnected:
            self.on_disconnected()

    async def authenticate(self):
        logger.info("Authentification...")
        await self.send({
            'type': 'AUTH',
            'password': self.password,
            'timestamp': int(time.time() * 1000),
            'protocolVersion': PROTOCOL_VERSION,
        })
        await self._wait_for_auth_response()

    async def _wait_for_auth_response(self):
        try:
            response = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
            message = json.loads(response)

            if message.get('type') == 'AUTH_SUCCESS':
                self.token = message.get('token')
                self.is_authenticated = True

                server_time = int(message.get('serverTime') or 0)
                if server_time > 0:
                    local_time = int(time.time() * 1000)
                    self._server_time_delta_ms = server_time - local_time

                logger.info(f"Authentifié (token: {self.token[:8]}..., proto={message.get('protocolVersion')})")

                if self.on_authenticated:
                    await self.on_authenticated(message)

            elif message.get('type') == 'AUTH_FAILED':
                logger.error(f"Authentification échouée: {message.get('reason')}")
                await asyncio.sleep(5)

        except asyncio.TimeoutError:
            logger.error("Timeout auth response")

    async def receive_loop(self):
        while self.is_running and self.ws:
            try:
                message_str = await self.ws.recv()
                message = json.loads(message_str)
                await self.handle_message(message)
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                logger.error(f"Erreur receive_loop: {e}", exc_info=True)
                break

    async def handle_message(self, message: dict):
        msg_type = message.get('type')

        if msg_type == 'TERMINATE':
            logger.warning(f"Demande de terminaison: {message.get('reason')}")
            self.is_running = False

        if self.on_message:
            await self.on_message(message)

    async def send(self, message: dict) -> bool:
        if not self.ws or not self.is_running:
            return False
        try:
            await self.ws.send(json.dumps(message))
            return True
        except Exception as e:
            logger.debug(f"Erreur send(): {e}")
            return False

    def send_threadsafe(self, message: dict) -> bool:
        if not self._loop or not self.is_running:
            return False
        try:
            asyncio.run_coroutine_threadsafe(self.send(message), self._loop)
            return True
        except Exception as e:
            logger.debug(f"Erreur send_threadsafe(): {e}")
            return False

    async def send_heartbeat(self):
        return await self.send({'type': 'ALIVE'})

    async def send_cache_complete(self):
        return await self.send({'type': 'CACHE_COMPLETE'})

    async def send_ready_to_terminate(self):
        return await self.send({'type': 'READY_TO_TERMINATE'})

    def disconnect(self):
        logger.info("Déconnexion...")
        self.is_running = False
        if self.ws and self._loop:
            try:
                asyncio.run_coroutine_threadsafe(self.ws.close(), self._loop)
            except Exception:
                pass

    def is_connected(self) -> bool:
        return self.ws is not None and self.is_authenticated