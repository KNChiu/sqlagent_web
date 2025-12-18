"""Configuration management using Pydantic Settings"""

from typing import Optional

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables"""

    # Database configuration
    db_uri: str = "sqlite:///./Chinook.db"

    # LLM model configuration
    model_name: str = "bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    # LLM authentication parameters (optional)
    model_base_url: Optional[str] = None  # Custom API endpoint
    model_api_key: Optional[str] = None  # API key for authentication

    # Query result limits
    query_limit: int = Field(
        default=20,
        description="Maximum number of rows returned from SQL queries",
        validation_alias=AliasChoices("QUERY_LIMIT", "TOP_K")
    )
    rag_top_k: int = 20

    # Agent execution limits
    recursion_limit: int = 30

    # DeepAgent feature flags
    enable_subagents: bool = True  # Enable subagent delegation

    # Subagent configuration
    subagent_model: str = "anthropic:claude-sonnet-4-20250514"  # Faster model for subagents

    # Schema RAG configuration
    schema_json_path: str = Field(
        default="./data/schema_descriptions.json",
        validation_alias="SCHEMA_OUTPUT_PATH"
    )
    index_path: str = Field(
        default="./data/faiss_index",
        validation_alias="INDEX_OUTPUT_PATH"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def dialect(self) -> str:
        """Auto-detect SQL dialect from database URI

        Returns:
            SQL dialect name for use in Agent prompts (e.g., SQLite, T-SQL, PostgreSQL)
        """
        uri_lower = self.db_uri.lower()

        # Map database URI prefixes to SQL dialect names
        if uri_lower.startswith("sqlite"):
            return "SQLite"
        elif uri_lower.startswith("postgresql") or uri_lower.startswith("postgres"):
            return "PostgreSQL"
        elif uri_lower.startswith("mysql"):
            return "MySQL"
        elif uri_lower.startswith("mssql"):
            return "T-SQL"
        elif uri_lower.startswith("oracle"):
            return "Oracle"
        else:
            # Fallback to generic SQL if dialect cannot be determined
            return "SQL"


# Global configuration instance
settings = Settings()
