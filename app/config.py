"""Configuration management using Pydantic Settings"""

from typing import Optional

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
