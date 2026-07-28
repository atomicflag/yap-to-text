"""Configuration loading and validation via Pydantic BaseModel."""

from pathlib import Path

import commentjson
from pydantic import BaseModel, Field


class TwitchConfig(BaseModel):
    """Twitch chat credentials and settings."""

    app_id: str
    app_secret: str
    target_channel: str
    refresh_token: str | None = None


class Config(BaseModel):
    """Application configuration loaded from config.jsonc."""

    twitch: TwitchConfig | None = None
    hallucinations: list[str] = Field(default_factory=lambda: ["The.", "."])
    erase_keyword: str | None = "not what i said"
    replacements: dict[str, str] = Field(default_factory=dict)
    audio_input_device: str | None = None
    audio_output_device: str | None = None
    obs_ws_host: str = "localhost"
    obs_ws_port: int = 4444


def load_config(path: Path = Path("config.jsonc")) -> Config:
    """Load and validate configuration from a JSONC file.

    If the config file does not exist, returns a Config instance with all
    default values (twitch disabled, built-in hallucinations/erase_keyword).

    Args:
        path: Path to the config.jsonc file. Defaults to current directory.

    Returns:
        Validated Config instance.

    """
    if not path.exists():
        return Config()

    with path.open("r", encoding="utf-8") as fp:
        data = commentjson.load(fp)
    return Config.model_validate(data)


def save_config(config: Config, path: Path) -> None:
    """Save a Config instance to a JSONC file.

    Args:
        config: The validated configuration to write.
        path: File path to write JSONC to.

    """
    with path.open("w", encoding="utf-8") as fp:
        commentjson.dump(config.model_dump(), fp, indent=2)
