"""Gemini generateContent transport; credentials only in the request header."""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any


class GeminiError(RuntimeError):
    pass


class GeminiClient:
    def __init__(self, *, api_key: str | None = None,
                 base_url: str = 'https://generativelanguage.googleapis.com/v1beta'):
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            raise GeminiError('GEMINI_API_KEY is required')
        if base_url.rstrip('/') != 'https://generativelanguage.googleapis.com/v1beta':
            raise GeminiError('Only the official Gemini generateContent endpoint is supported')
        self.base_url = base_url.rstrip('/')

    def _request(self, *, model: str, system_prompt: str, user_prompt: str,
                 max_tokens: int, json_mode: bool = False) -> str:
        if not re.fullmatch(r'gemini-[A-Za-z0-9.-]+', model) or max_tokens <= 0:
            raise GeminiError('Invalid Gemini model or output budget')
        generation = {'maxOutputTokens': int(max_tokens)}
        if model == 'gemini-3.8-flash':
            generation['thinkingConfig'] = {'thinkingLevel': 'low'}
        if json_mode:
            generation['responseMimeType'] = 'application/json'
        request = urllib.request.Request(f'{self.base_url}/models/{model}:generateContent',
            data=json.dumps({'systemInstruction': {'parts': [{'text': system_prompt}]},
                'contents': [{'role': 'user', 'parts': [{'text': user_prompt}]}],
                'generationConfig': generation}).encode(),
            headers={'Content-Type': 'application/json', 'x-goog-api-key': self.api_key}, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise GeminiError(f'Gemini HTTP {exc.code}') from None
        except (urllib.error.URLError, TimeoutError, ValueError):
            raise GeminiError('Gemini network or response decoding failure') from None
        candidates = result.get('candidates')
        if not candidates or candidates[0].get('finishReason') != 'STOP':
            raise GeminiError('Gemini did not return a complete candidate')
        parts = candidates[0].get('content', {}).get('parts', [])
        text = ''.join(p.get('text', '') for p in parts if not p.get('thought', False)).strip()
        if not text:
            raise GeminiError('Gemini returned no answer text')
        return text

    def text(self, *, model: str, system_prompt: str, user_prompt: str, max_tokens: int = 4000) -> str:
        return self._request(model=model, system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=max_tokens)

    def json_object(self, *, model: str, system_prompt: str, user_prompt: str,
                    max_tokens: int = 4000) -> dict[str, Any]:
        text = self._request(model=model, system_prompt=system_prompt, user_prompt=user_prompt,
                             max_tokens=max_tokens, json_mode=True)
        try:
            result = json.loads(text)
        except ValueError:
            raise GeminiError('Gemini JSON response was invalid') from None
        if not isinstance(result, dict):
            raise GeminiError('Gemini JSON response must be an object')
        return result
