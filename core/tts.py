import httpx

try:
    import dashscope
    from dashscope.audio.tts_v2 import SpeechSynthesizer
except Exception:  # optional dependency
    dashscope = None
    SpeechSynthesizer = None

class QwenTTS:
    def __init__(self, api_key: str, base_url: str, model: str, voice: str, language_type: str, timeout: float):
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or "").strip().rstrip("/")
        self.model = (model or "qwen3-tts-flash").strip() or "qwen3-tts-flash"
        self.voice = (voice or "Cherry").strip() or "Cherry"
        self.language_type = (language_type or "Chinese").strip() or "Chinese"
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key) and bool(self.base_url)

    def synthesize_wav(self, text: str) -> bytes:
        if not self.available():
            raise RuntimeError("QwenTTS not configured")

        url = self.base_url + "/services/aigc/multimodal-generation/generation"
        payload = {
            "model": self.model,
            "input": {
                "text": text,
                "voice": self.voice,
                "language_type": self.language_type,
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=httpx.Timeout(self.timeout, connect=10), follow_redirects=True) as c:
            r = c.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            audio_url = (((data or {}).get("output") or {}).get("audio") or {}).get("url")
            if not audio_url:
                raise RuntimeError(f"QwenTTS returned no audio url: {str(data)[:200]}")

            r2 = c.get(audio_url)
            r2.raise_for_status()
            return bytes(r2.content)


class AliyunTTS:
    def __init__(self, api_key: str, model: str, voice: str, volume: int, fallback_model: str, fallback_voice: str):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.volume = volume
        self.fallback_model = fallback_model
        self.fallback_voice = fallback_voice

    def available(self) -> bool:
        return bool(self.api_key) and dashscope is not None and SpeechSynthesizer is not None

    def synthesize_mp3(self, text: str) -> bytes:
        # DashScope SDK uses env var or dashscope.api_key
        dashscope.api_key = self.api_key
        try:
            synthesizer = SpeechSynthesizer(model=self.model, voice=self.voice, volume=self.volume)
            audio = synthesizer.call(text)
            return bytes(audio)
        except Exception:
            # Fallback to cheaper model/voice
            if self.fallback_model and self.fallback_voice:
                synthesizer = SpeechSynthesizer(model=self.fallback_model, voice=self.fallback_voice, volume=self.volume)
                audio = synthesizer.call(text)
                return bytes(audio)
            raise
