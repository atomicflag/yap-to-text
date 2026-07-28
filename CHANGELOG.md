# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
