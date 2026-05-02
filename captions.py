#!/usr/bin/env python3
"""Generate SRT subtitles and a plain transcript from a local video file."""

from __future__ import annotations

import argparse
import importlib
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable


MODEL_CHOICES = ("tiny", "base", "small", "medium", "large")


class CaptionError(RuntimeError):
    """Raised for expected CLI errors that should be shown cleanly."""


class PercentProgress:
    """Small tqdm-compatible progress reporter for Whisper's transcribe loop."""

    def __init__(self, *args, total: int | float | None = None, disable: bool = False, **kwargs) -> None:
        self.total = float(total or 0)
        self.current = 0.0
        self.disable = disable
        self.last_percent = -1

    def __enter__(self) -> "PercentProgress":
        self._print(0, force=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self._print(100, force=True)
        if not self.disable:
            print(file=sys.stderr)

    def update(self, amount: int | float) -> None:
        self.current += float(amount)
        if self.total <= 0:
            return
        percent = min(100, int(round((self.current / self.total) * 100)))
        self._print(percent)

    def close(self) -> None:
        self._print(100, force=True)

    def _print(self, percent: int, force: bool = False) -> None:
        if self.disable:
            return
        if force or percent != self.last_percent:
            self.last_percent = percent
            print(f"\rTranscribing: {percent:3d}%", end="", file=sys.stderr, flush=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate .srt captions and a .txt transcript from a local video file."
    )
    parser.add_argument("input", type=Path, help="Path to a local video or audio file.")
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="small",
        help="Whisper model to use. Default: small.",
    )
    return parser.parse_args(argv)


def require_input_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise CaptionError(f"Input file not found: {path}")
    if not resolved.is_file():
        raise CaptionError(f"Input path is not a file: {path}")
    return resolved


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise CaptionError(
            "FFmpeg was not found on PATH. Install FFmpeg and make sure the "
            "'ffmpeg' command is available before running this tool."
        )


def import_whisper():
    try:
        return importlib.import_module("whisper")
    except ImportError as exc:
        raise CaptionError(
            "The 'whisper' Python package is not installed. Install it with "
            "'pip install -U openai-whisper'."
        ) from exc


def patch_whisper_progress():
    """Replace Whisper's tqdm progress bar with a clean percentage indicator.

    OpenAI Whisper decodes video/audio through FFmpeg and tracks frame progress
    internally. The local package exposes that through tqdm; this patch keeps the
    same interface while printing simple percentages. If a future Whisper version
    changes this internal detail, transcription still works without the patch.
    """

    try:
        transcribe_module = importlib.import_module("whisper.transcribe")
        original_tqdm = transcribe_module.tqdm.tqdm
        transcribe_module.tqdm.tqdm = PercentProgress
        return transcribe_module.tqdm, original_tqdm
    except (AttributeError, ImportError):
        return None, None


def restore_whisper_progress(patch_info) -> None:
    tqdm_module, original_tqdm = patch_info
    if tqdm_module is not None and original_tqdm is not None:
        tqdm_module.tqdm = original_tqdm


def seconds_to_srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def format_srt(segments: Iterable[dict]) -> str:
    blocks: list[str] = []
    index = 1
    for segment in segments:
        text = clean_text(str(segment.get("text", "")))
        if not text:
            continue

        start = seconds_to_srt_timestamp(float(segment["start"]))
        end = seconds_to_srt_timestamp(float(segment["end"]))
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
        index += 1

    return "\n\n".join(blocks) + ("\n" if blocks else "")


def format_transcript(segments: Iterable[dict]) -> str:
    lines = [clean_text(str(segment.get("text", ""))) for segment in segments]
    transcript = " ".join(line for line in lines if line)
    return transcript + ("\n" if transcript else "")


def transcribe(input_file: Path, model_name: str) -> dict:
    require_ffmpeg()
    whisper = import_whisper()

    print(f"Loading Whisper model: {model_name}", file=sys.stderr)
    try:
        model = whisper.load_model(model_name)
    except Exception as exc:
        raise CaptionError(f"Failed to load Whisper model '{model_name}': {exc}") from exc

    patch_info = patch_whisper_progress()
    try:
        return model.transcribe(str(input_file), verbose=False)
    except FileNotFoundError as exc:
        raise CaptionError(f"Could not read input file: {input_file}") from exc
    except Exception as exc:
        raise CaptionError(f"Transcription failed: {exc}") from exc
    finally:
        restore_whisper_progress(patch_info)


def write_outputs(input_file: Path, result: dict) -> tuple[Path, Path]:
    segments = list(result.get("segments", []))
    srt_path = input_file.with_suffix(".srt")
    txt_path = input_file.with_suffix(".txt")

    srt_path.write_text(format_srt(segments), encoding="utf-8")
    txt_path.write_text(format_transcript(segments), encoding="utf-8")
    return srt_path, txt_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        input_file = require_input_file(args.input)
        result = transcribe(input_file, args.model)
        srt_path, txt_path = write_outputs(input_file, result)
    except CaptionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130

    print(f"Wrote subtitles: {srt_path}")
    print(f"Wrote transcript: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
