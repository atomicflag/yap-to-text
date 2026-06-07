"""Configuration loading and validation via Pydantic BaseModel."""

from pathlib import Path

from pydantic import BaseModel, Field


class TwitchConfig(BaseModel):
    """Twitch chat credentials and settings."""

    app_id: str
    app_secret: str
    target_channel: str
    refresh_token: str | None = None


class Config(BaseModel):
    """Application configuration loaded from config.json."""

    twitch: TwitchConfig | None = None
    hallucinations: list[str] = Field(default_factory=lambda: ["The.", "."])
    erase_keyword: str = "not what i said"
    replacements: dict[str, str] = Field(default_factory=dict)
    audio_input_device: str | None = None
    audio_output_device: str | None = None


def load_config(path: Path = Path("config.json")) -> Config:
    """Load and validate configuration from a JSON file.

    If the config file does not exist, returns a Config instance with all
    default values (twitch disabled, built-in hallucinations/erase_keyword).

    Args:
        path: Path to the config.json file. Defaults to current directory.

    Returns:
        Validated Config instance.

    """
    if not path.exists():
        return Config()

    data = path.read_text(encoding="utf-8")
    return Config.model_validate_json(data)


def save_config(config: Config, path: Path) -> None:
    """Save a Config instance to a JSON file.

    Args:
        config: The validated configuration to write.
        path: File path to write JSON to.

    """
    json_str = config.model_dump_json(indent=2)
    path.write_text(json_str + "\n", encoding="utf-8")
