# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] — 2026-10-17

### Added

- `silero-vad` dependency for voice activity detection

### Changed

- CUDA stack upgraded to target CUDA 13.x systems: PyTorch wheel index moved from `cu128` to `cu132`, torch 2.11.0+cu128 to 2.14.1+cu132, triton 3.6.0 to 3.8.0, CUDA runtime libraries moved to the cu13 series (cuDNN 9.24, NCCL 2.30.7, cuBLAS 13.4), `cuda-python` 12.9.7 to 13.4.4
- flash-attn pre-built wheels updated from `2.8.3+cu128torch2.11` to `2.8.3+cu132torch2.14`
- Silero VAD is now loaded from the `silero-vad` package instead of `torch.hub`
- Dependency bumps: `fastapi` 0.121 to 0.143, `typer` 0.15 to 0.27, `bitsandbytes` 0.50.0 to 0.50.2, `numba` 0.66 to 0.68
- Documented GPU requirement as CUDA 13.2+ in the README

### Removed

- `torchaudio` dependency (unused; upstream releases stopped at 2.11.0)

## [0.3.1] — 2026-10-09

### Added

- 4-bit quantized KV cache (quanto backend) during generation to further reduce VRAM usage
- `optimum-quanto` dependency for KV cache quantization

### Changed

- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce CUDA allocator fragmentation

## [0.3.0] — 2026-07-28

### Added

- OBS WebSocket closed captions support in Twitch mode
- Optional OBS WebSocket configuration fields (`obs_ws_host`, `obs_ws_port`)

### Changed

- Twitch mode now sends transcriptions to both Twitch chat and OBS Studio (when configured)

## [0.2.0] — 2026-07-28

### Added

- 4-bit quantization support via `bitsandbytes` (nf4 + bfloat16 compute) to significantly reduce VRAM usage

### Changed

- `max_inference_batch_size` reduced from 32 to 1

## [0.1.0] — 2026-06-03

### Added

- Real-time speech transcription for VRChat (OSC) and Twitch integration
- Clipboard output mode
- Configurable hallucination filters and erase keyword
- Case-insensitive phrase replacements to auto-correct common ASR mistakes
- `--verbose` / `-v` CLI flag for increased log verbosity
- Debug logging for OSC message delivery
- Make `erase_keyword` optional (set to `null` to disable)
