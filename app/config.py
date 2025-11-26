"""Configuration management using Pydantic Settings"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables"""

    # Database configuration
    db_uri: str = "sqlite:///./Chinook.db"

    # LLM model configuration
    model_name: str = "bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    # Query result limits
    top_k: int = 5
    rag_top_k: int = 5

    # Agent execution limits
    recursion_limit: int = 15

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Global configuration instance
settings = Settings()
