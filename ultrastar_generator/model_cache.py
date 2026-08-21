"""Process-wide cache for expensive-to-load ASR/alignment models, keyed by (model_name, device, compute_type, ...)."""

from __future__ import annotations

_whisperx_asr_cache: dict = {}
_whisperx_align_cache: dict = {}
_faster_whisper_cache: dict = {}


def get_whisperx_asr_model(model_name: str, device: str = "cuda",
                            compute_type: str = "float16", language: str = "en",
                            vad_options: dict = None):
    """`vad_options` is part of the cache key."""
    key = (model_name, device, compute_type, language, tuple(sorted((vad_options or {}).items())))
    if key not in _whisperx_asr_cache:
        import whisperx
        print(f"  Loading whisperx ASR model '{model_name}' (first use this run)"
              f"{f' with vad_options={vad_options}' if vad_options else ''}...", flush=True)
        _whisperx_asr_cache[key] = whisperx.load_model(
            model_name, device=device, compute_type=compute_type, language=language,
            vad_options=vad_options,
        )
    return _whisperx_asr_cache[key]


def get_whisperx_align_model(language_code: str = "en", device: str = "cuda"):
    key = (language_code, device)
    if key not in _whisperx_align_cache:
        import whisperx
        print(f"  Loading whisperx alignment model ({language_code}, first use this run)...", flush=True)
        _whisperx_align_cache[key] = whisperx.load_align_model(language_code=language_code, device=device)
    return _whisperx_align_cache[key]


def get_faster_whisper_model(model_name: str, device: str = "cuda", compute_type: str = "float16"):
    key = (model_name, device, compute_type)
    if key not in _faster_whisper_cache:
        from faster_whisper import WhisperModel
        print(f"  Loading faster-whisper model '{model_name}' (first use this run)...", flush=True)
        _faster_whisper_cache[key] = WhisperModel(model_name, device=device, compute_type=compute_type)
    return _faster_whisper_cache[key]


def reset() -> None:
    """Clears all cached models. Used by tests, not the pipeline itself."""
    _whisperx_asr_cache.clear()
    _whisperx_align_cache.clear()
    _faster_whisper_cache.clear()
