import logging
import time
import uuid
from typing import Any, Dict, Optional

import PIL
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer

from src.indexing.image_indexer import ImageIndexer


class FashionImageIndexingService:
    """
    Service class responsible for indexing fashion images into a vector database.

    This class handles loading the dataset, processing images in batches,
    generating embeddings, and indexing them using the ImageIndexer.
    """

    def __init__(
        self,
        dataset_name: str = "valex95/fashion-dataset",
        split: str = "train",
        model_name: str = "clip-ViT-B-32",
        batch_size: int = 10000,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the Fashion Image Indexing Service.

        Args:
            dataset_name: Name of the dataset to load
            split: Dataset split to use
            model_name: Name of the SentenceTransformer model to use
            batch_size: Number of images to process in each batch
            logger: Logger instance to use (creates one if None)
        """
        self.dataset_name = dataset_name
        self.split = split
        self.model_name = model_name
        self.batch_size = batch_size

        # Set up logging
        if logger:
            self.logger = logger
        else:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            self.logger = logging.getLogger(__name__)

        # Initialize components
        self.data = None
        self.label_names = None
        self.model = None
        self.indexer = None

    def _load_dataset(self) -> None:
        """Load the dataset and get label names."""
        self.logger.info(f"Loading dataset: {self.dataset_name}, split: {self.split}")
        self.data = load_dataset(self.dataset_name, split=self.split)
        self.label_names = self.data.features["label"].names
        self.logger.info(f"Dataset loaded with {len(self.data)} images")
        self.logger.debug(f"Sample data point: {self.data[0]}")

    def _load_model(self) -> None:
        """Load the embedding model."""
        self.logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

    def _initialize_indexer(self) -> None:
        """Initialize the image indexer."""
        self.logger.info("Initializing image indexer")
        self.indexer = ImageIndexer()

    def _process_batch(
        self, batch: Dataset, batch_index: int, total_batches: int
    ) -> int:
        """
        Process a single batch of images.

        Args:
            batch: The batch of data to process
            batch_index: The index of the current batch
            total_batches: Total number of batches

        Returns:
            Number of images processed in the batch
        """
        batch_start_time = time.time()
        batch_size_actual = len(batch)

        self.logger.info(
            f"Processing batch {batch_index}/{total_batches} with {batch_size_actual} images"
        )

        # Extract images
        images = batch["image"]

        # Generate embeddings
        self.logger.info(f"Generating embeddings for batch {batch_index}")
        image_embeddings = self.model.encode([img for img in images])

        # Prepare IDs and display names
        ids = [str(item) if item else str(uuid.uuid4()) for item in batch["image_id"]]
        display_names = [
            f"Fashion Item {batch_index * self.batch_size + idx}"
            for idx in range(len(batch))
        ]

        # Get category names from IDs
        category_names = [self.label_names[label_id] for label_id in batch["label"]]

        # Index the batch
        self.logger.info(f"Indexing batch {batch_index} to vector database")
        self.indexer.index_images(
            image_embeddings=image_embeddings,
            images=images,
            ids=ids,
            display_names=display_names,
            categories=category_names,
            batch_size=self.batch_size,
        )

        batch_duration = time.time() - batch_start_time
        avg_per_image = (
            batch_duration / batch_size_actual if batch_size_actual > 0 else 0
        )

        self.logger.info(
            f"Batch {batch_index} completed in {batch_duration:.2f}s ({avg_per_image:.4f}s per image)"
        )

        return batch_size_actual

    def run(self) -> Dict[str, Any]:
        """
        Run the full indexing process.

        Returns:
            Dictionary containing indexing statistics
        """
        # Initialize components
        self._load_dataset()
        self._load_model()
        self._initialize_indexer()

        # Calculate batching
        total_batches = (len(self.data) + self.batch_size - 1) // self.batch_size
        self.logger.info(
            f"Starting indexing process for {len(self.data)} images in {total_batches} batches"
        )

        # Track statistics
        total_processed = 0
        start_time = time.time()

        # Process batches
        for i in range(0, len(self.data), self.batch_size):
            try:
                end_idx = min(i + self.batch_size, len(self.data))
                batch = self.data[i:end_idx]

                batch_processed = self._process_batch(
                    batch,
                    batch_index=i // self.batch_size + 1,
                    total_batches=total_batches,
                )

                total_processed += batch_processed
                self.logger.info(
                    f"Progress: {total_processed}/{len(self.data)} images ({(total_processed / len(self.data) * 100):.2f}%)"
                )

            except PIL.UnidentifiedImageError as e:
                self.logger.warning(f"Skipping corrupted image at index {i}: {e}")
                continue

        # Summarize results
        total_duration = time.time() - start_time
        avg_time_per_image = (
            total_duration / total_processed if total_processed > 0 else 0
        )

        self.logger.info(
            f"Indexing complete! Processed {total_processed} images in {total_duration:.2f}s"
        )
        self.logger.info(
            f"Average processing time: {avg_time_per_image:.4f}s per image"
        )

        # Get collection statistics
        stats = self.indexer.get_collection_stats()
        self.logger.info(f"Collection stats: {stats}")

        return {
            "total_images": len(self.data),
            "processed_images": total_processed,
            "total_duration_seconds": total_duration,
            "avg_time_per_image_seconds": avg_time_per_image,
            "collection_stats": stats,
        }


# Example usage
if __name__ == "__main__":
    indexing_service = FashionImageIndexingService()
    results = indexing_service.run()
    print(f"Indexing completed with {results['processed_images']} images processed")
