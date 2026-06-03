# qwen-asr-vrc

Real-time speech-to-text captions for VRChat and Twitch, powered by Qwen3-ASR.

Turn your voice into text displayed inside VRChat's chatbox, or send transcriptions to a Twitch channel — no typing required.

**Status:** Beta · Requires an NVIDIA GPU with 8+ GB VRAM


## Features

- **Live VRChat captions** — sends real-time (and final) transcriptions directly to VRC's in-game chatbox via OSC
- **Twitch mode** — stream your own live captions to Twitch chat instead (`--twitch`)
- **Intermediate text** — see words appear as they're recognized, before speech ends
- **OBS browser source** — built-in web server serves live subtitles for OBS Studio overlays (run with `--twitch`)
- **Clipboard copy** — dump full transcriptions to your clipboard (`--clipboard`)
- **Hallucination & erase detection** — filters ASR hallucinations and clears the buffer on an erase keyword


## System Requirements

| Requirement | Details |
|---|---|
| OS | Linux or Windows |
| GPU | NVIDIA with 8+ GB VRAM (CUDA 12.8+) |
| Python | 3.12 |
| Package manager | [uv](https://docs.astral.sh/uv/) |


## Installation

```bash
# Clone the repo
git clone https://github.com/atomicflag/qwen-asr-vrc.git
cd qwen-asr-vrc

# Create a virtual environment and install dependencies
uv sync
```

That's it — `uv` resolves everything including the PyTorch and flash-attn wheels. The Qwen3-ASR model (1.7B) downloads automatically on first run.

> **Note:** If you're not familiar with Python or the uv ecosystem, see [uv getting started](https://docs.astral.sh/uv/getting-started/) for a quick intro.


## Configuration (optional)

Copy the example config and fill in your values:

```bash
cp config.json.example config.json
```

### VRChat (OSC) mode — no extra setup needed

Default mode sends text to VRChat's chatbox over localhost OSC. Just make sure VRChat is running.

### Twitch mode — requires a Twitch Developer app

1. [Register an app](https://dev.twitch.tv/docs/authentication/register-app) on the Twitch Developer Console
2. Note your **Client ID** and **Client Secret**
3. Add them to `config.json` under `twitch`, along with your channel name
4. On first run with `--twitch`, a browser window opens for OAuth authorization — grant permissions and you're done

```jsonc
{
  "twitch": {
    "app_id": "your_client_id",
    "app_secret": "your_client_secret",
    "target_channel": "your_channel_name"
  },
  "hallucinations": ["The.", "."],
  "erase_keyword": "not what i said",
  "audio_input_device": null,
  "audio_output_device": null
}
```


## Usage

### VRChat mode (default)

```bash
uv run python -m stt
```

Transcriptions appear in VRC's chatbox in real-time.

### Twitch mode

```bash
uv run python -m stt --twitch
```

Sends final transcriptions to Twitch chat and serves live captions via a local web server (port `9098`) for OBS browser source. A blip sound plays when each speech segment ends.

To add the OBS browser source:
1. Open OBS → Sources → Browser
2. Set URL to `http://localhost:9098`
3. Adjust width/height as needed

### Clipboard mode (works with either VRChat or Twitch)

```bash
uv run python -m stt --clipboard    # or --twitch --clipboard
```

Copies the full running transcription to your system clipboard whenever a segment is finalized.


## Options

Run `uv run python -m stt --help` to see all available command-line flags:

| Flag | Description |
|---|---|
| `--twitch` | Enable Twitch chat mode (sends to Twitch instead of VRC) |
| `--clipboard` | Copy final transcriptions to the system clipboard |
| `--config PATH` | Custom path to `config.json` (default: `config.json`) |

## License

MIT — see [LICENSE](LICENSE) for details.
