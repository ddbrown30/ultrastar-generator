"""Process-wide cache for expensive-to-load ASR/alignment models.

Without this, a single pipeline run could load the same model multiple
times: once for the main transcription pass (transcription.py), again for
verify_words (verification.py) -- each of which used to call
whisperx.load_model()/load_align_model() or
faster_whisper.WhisperModel() fresh. Loading is a meaningful fraction of
total runtime for a large model (e.g. --whisper-model large-v3), so this
caches by (model_name, device, compute_type, ...) and hands back the
already-loaded model on every call after the first.

Deliberately a plain module-level dict, not a class -- there's exactly one
pipeline running per process, so a process-wide cache is the right scope;
no need for anything more elaborate.
"""

from __future__ import annotations

_whisperx_asr_cache: dict = {}
_whisperx_align_cache: dict = {}
_faster_whisper_cache: dict = {}


def get_whisperx_asr_model(model_name: str, device: str = "cuda",
                            compute_type: str = "float16", language: str = "en",
                            vad_options: dict = None):
    """`vad_options` (e.g. {"vad_onset": 0.01, "vad_offset": 0.01}) is part
    of the cache key -- a near-zero onset/offset makes pyannote VAD treat
    almost everything as speech, the closest whisperx's API gets to
    "disabled" (it has no true off switch; vad_model/vad_method is
    mandatory in whisperx.load_model())."""
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
    """Clears all cached models. Not used by the pipeline itself (one
    process = one run = no reason to evict) -- exists for tests that swap
    out the underlying whisperx/faster_whisper modules between cases, so a
    later fake doesn't silently reuse an earlier case's cached instance."""
    _whisperx_asr_cache.clear()
    _whisperx_align_cache.clear()
    _faster_whisper_cache.clear()
