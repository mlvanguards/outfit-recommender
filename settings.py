from pathlib import Path
from typing import ClassVar, List

from pydantic_settings import BaseSettings


class BaseAppSettings(BaseSettings):
    """Base settings class with common configuration."""

    BASE_DIR: ClassVar[Path] = Path(__file__).resolve().parent.parent.parent

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class LLMSettings(BaseAppSettings):
    """OpenAI API settings."""

    OPENAI_API_URL: str = "https://api.openai.com/v1/chat/completions"
    GPT_MODEL: str = "gpt-4o"
    GPT_TEMPERATURE: float = 0.7
    GPT_MAX_TOKENS: int = 500


class QdrantSettings(BaseAppSettings):
    """Qdrant settings."""

    QDRANT_HOST: str = "http://localhost:6333"
    QDRANT_API_KEY: str = "your_api_key"


class DataSettings(BaseAppSettings):
    """Data processing and training settings."""

    DATA_TRAIN_SIZE: float = 0.7
    DATA_VAL_SIZE: float = 0.15
    DATA_TEST_SIZE: float = 0.15
    DATA_RANDOM_SEED: int = 42
    DATA_MAX_AUDIO_DURATION: float = 10.0
    DATA_AUDIO_EXTENSIONS: List[str] = [".wav", ".flac", ".mp3"]
    DATA_MIN_WORD_LENGTH: int = 2
    DATA_MAX_WORD_FREQUENCY: int = 3
    DATA_SHORT_WORD_THRESHOLD: float = 0.2
    DATA_KAGGLE_DATASET: str = "mirfan899/kids-speech-dataset"
    DATA_OUTPUT_DIR: str = "dataset_kaggle"
    HF_AUTH_TOKEN: str = None


class Settings(BaseAppSettings):
    """Main settings class that combines all specialized settings."""

    llm: LLMSettings = LLMSettings()
    data: DataSettings = DataSettings()
    qdrant: QdrantSettings = QdrantSettings()


# Create global settings instance
settings = Settings()
