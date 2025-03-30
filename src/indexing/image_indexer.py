import uuid
from typing import Any, Dict, List

from PIL import Image
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from src.core.utils.images.convert import pil_image_to_base64
from src.db.qdrant import manager


class ImageIndexer:
    COLLECTION_NAME = "fashion_images"
    VECTOR_SIZE = 512

    def __init__(self):
        self._manager = manager
        self._init_collection()

    def _init_collection(self):
        """Initialize the image collection with proper vector configuration."""
        self._manager.init()
        collections = [
            c.name for c in self._manager._client.get_collections().collections
        ]

        if self.COLLECTION_NAME not in collections:
            self._manager._client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config={
                    "image": VectorParams(
                        size=self.VECTOR_SIZE, distance=Distance.COSINE
                    )
                },
            )

    def index_images(
        self,
        image_embeddings: List[List[float]],
        images: List[Image.Image],
        ids: List[str],
        display_names: List[str],
        categories: List[str] = None,
        batch_size: int = 100,
    ) -> None:
        """
        Index a batch of images with their embeddings and metadata.

        Args:
            image_embeddings: List of image embedding vectors
            images: List of PIL Image objects
            ids: List of product IDs
            display_names: List of product display names
            categories: List of product categories (optional)
            batch_size: Number of images to process in each batch
        """
        self._manager.init()

        try:
            for i in range(0, len(ids), batch_size):
                batch_end = min(i + batch_size, len(ids))
                points = []

                for j, (embedding, image, id_, name) in enumerate(
                    zip(
                        image_embeddings[i:batch_end],
                        images[i:batch_end],
                        ids[i:batch_end],
                        display_names[i:batch_end],
                    )
                ):
                    payload = {
                        "id": id_,
                        "productDisplayName": name,
                        "base64_image": pil_image_to_base64(image),
                    }

                    if categories:
                        category_idx = i + j
                        if category_idx < len(categories):
                            payload["category"] = categories[category_idx]

                    point_id = str(uuid.uuid4())
                    points.append(
                        PointStruct(
                            id=point_id, payload=payload, vector={"image": embedding}
                        )
                    )

                self._manager._client.upsert(
                    collection_name=self.COLLECTION_NAME, wait=True, points=points
                )
        finally:
            self._manager.close()

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get information about the image collection."""
        return self._manager.get_collection_info(self.COLLECTION_NAME)
