import hashlib
import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import VectorParams

from settings import settings

logger = logging.getLogger(__name__)


class QdrantSessionManager:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._client: Optional[QdrantClient] = None

    def init(self):
        try:
            self._client = QdrantClient(
                url=settings.qdrant.QDRANT_HOST, api_key=settings.qdrant.QDRANT_API_KEY
            )
        except Exception as e:
            logger.exception(f"Could not connect to qdrant: {e}")
            raise
        logger.info("Connected to qdrant")

    def _create_collection(self, name: str):
        collections = [c.name for c in self._client.get_collections().collections]

        if name not in collections:
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=1, distance=models.Distance.DOT),
                on_disk_payload=True,
            )

    def get_collection_info(self, collection: str):
        self.init()

        try:
            info = self._client.get_collection(collection_name=collection)
            return info.points_count
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return None
        finally:
            self.close()

    def _generate_id(self, text):
        """Generate a unique integer ID from text."""
        return int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**63)

    def scroll(self, collection: str, offset):
        self.init()

        response = self._client.scroll(
            collection_name=collection,
            limit=100,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )

        self.close()

        return response

    def close(self):
        if self._client:
            self._client.close()
            logger.info("Connection was closed.")


manager = QdrantSessionManager()
