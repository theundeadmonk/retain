"""Application settings via pydantic-settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = (
        "postgresql+asyncpg://retain:retain@localhost:5432/retain"
    )
    test_database_url: str = (
        "postgresql+asyncpg://retain:retain@localhost:5432/retain_test"
    )
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = "gpt-4o-mini"

    embedding_query_model: str = "BAAI/bge-large-en-v1.5"
    embedding_batch_model: str = "BAAI/bge-large-en-v1.5"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    embedding_sparse_model: str = "prithivida/Splade_PP_en_v1"

    model_config = {"env_prefix": "RETAIN_"}


settings = Settings()
