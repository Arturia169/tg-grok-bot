import httpx

class OpenAICompat:
    def __init__(self, base_url: str, api_key: str, timeout: float):
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout, connect=10),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def images_generate(
        self,
        model: str,
        prompt: str,
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
    ) -> dict:
        payload = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": response_format,
            "stream": False,
        }
        r = await self._client.post("/images/generations", json=payload)
        r.raise_for_status()
        return r.json()

    async def images_edits(
        self,
        model: str,
        prompt: str,
        image_bytes: bytes,
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
    ) -> dict:
        files = {
            "image": ("image.png", image_bytes, "image/png"),
        }
        data = {
            "model": model,
            "prompt": prompt,
            "n": str(n),
            "size": size,
            "response_format": response_format,
            "stream": "false",
        }
        # multipart/form-data, do not set Content-Type header manually
        headers = {k: v for k, v in self._client.headers.items() if k.lower() != "content-type"}
        r = await self._client.post("/images/edits", data=data, files=files, headers=headers)
        r.raise_for_status()
        return r.json()

    async def close(self):
        await self._client.aclose()

    async def chat(self, model: str, user_text: str, system_prompt: str = "你是一个乐于助人的助手。", max_tokens: int = None) -> str:
        return await self.chat_messages(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            max_tokens=max_tokens,
        )

    async def chat_messages(self, model: str, messages: list[dict], temperature: float = 0.7, max_tokens: int = None) -> str:
        payload = {
            "model": model,
            "stream": False,
            "temperature": temperature,
            "messages": messages,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        r = await self._client.post("/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except Exception:
            return ""

    async def decide_tts(self, model: str, user_text: str, assistant_text: str) -> bool:
        # Minimal token usage classifier. Output must be exactly VOICE or TEXT.
        prompt = (
            "你是一个分类器。只输出一个单词：VOICE 或 TEXT。\n"
            "规则：\n"
            "- 如果内容包含代码块/命令/大量链接/表格/长清单，选 TEXT。\n"
            "- 如果是简短日常对话、情绪安慰、鼓励、结论、提醒，选 VOICE。\n"
            "- 默认偏向 TEXT（省额度）。\n\n"
            f"用户：{user_text}\n"
            f"助手拟回复：{assistant_text}\n"
        )
        payload = {
            "model": model,
            "stream": False,
            "temperature": 0,
            "max_tokens": 1,
            "messages": [
                {"role": "system", "content": "You are a strict classifier."},
                {"role": "user", "content": prompt},
            ],
        }
        r = await self._client.post("/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        out = ""
        try:
            out = (data["choices"][0]["message"]["content"] or "").strip().upper()
        except Exception:
            out = ""
        return out == "VOICE"
