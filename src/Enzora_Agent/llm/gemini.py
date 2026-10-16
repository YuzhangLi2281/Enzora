"""Configured LLM transports; production MiniMax supports native tool calls."""
from __future__ import annotations

import json
import http.client
import os
import re
import time
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


class MiniMaxClient:
    """Official Chat Completions transport, including bounded native tool use.

    Full assistant messages stay in the private in-flight conversation as required
    by MiniMax. Reasoning is never returned to the browser or written to audits.
    """
    def __init__(self, *, api_key: str | None = None, base_url: str = 'https://api.minimax.cn/v1'):
        self.api_key = api_key or os.environ.get('MINIMAX_API_KEY')
        if not self.api_key:
            raise ValueError('MINIMAX_API_KEY is required')
        if base_url.rstrip('/') not in {'https://api.minimax.cn/v1', 'https://api.minimax.io/v1'}:
            raise ValueError('Use an official MiniMax API endpoint')
        self.base_url = base_url.rstrip('/')

    def complete(self, *, model: str, messages: list[dict], max_tokens: int, tools: list[dict] | None = None) -> dict:
        if not re.fullmatch(r'MiniMax-[A-Za-z0-9.-]+', model) or max_tokens <= 0:
            raise ValueError('Invalid MiniMax model or output budget')
        body = {'model': model, 'messages': messages, 'max_tokens': max_tokens, 'temperature': 1.0}
        if tools:
            body['tools'] = tools
        request = urllib.request.Request(self.base_url + '/chat/completions',
            data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.api_key})
        # A reasoning response can exceed two minutes. Retry only transport
        # failures: tool execution happens after a complete, validated response.
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    result = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise RuntimeError(f'MiniMax HTTP {exc.code}') from None
            except (urllib.error.URLError, OSError, http.client.HTTPException):
                if attempt == 2:
                    raise RuntimeError('MiniMax network request failed after three attempts') from None
            except ValueError:
                raise RuntimeError('MiniMax response decoding failure') from None
            time.sleep(2 ** attempt)
        if result.get('base_resp', {}).get('status_code', 0) != 0:
            raise RuntimeError('MiniMax rejected the request')
        choices = result.get('choices', [])
        if not choices or choices[0].get('finish_reason') not in {'stop', 'tool_calls'}:
            raise RuntimeError('MiniMax returned an incomplete response; check the output budget')
        message = choices[0].get('message')
        if not isinstance(message, dict):
            raise RuntimeError('MiniMax returned an invalid assistant message')
        return message

    @staticmethod
    def answer(message: dict) -> str:
        text = re.sub(r'<think>.*?</think>', '', message.get('content') or '', flags=re.S).strip()
        if '<think>' in text or not text:
            raise RuntimeError('MiniMax returned no complete answer text')
        return text

    def text(self, *, model: str, system_prompt: str, user_prompt: str, max_tokens: int = 8192) -> str:
        message = self.complete(model=model, max_tokens=max_tokens, messages=[
            {'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}])
        return self.answer(message)

    def json_object(self, *, model: str, system_prompt: str, user_prompt: str, max_tokens: int = 8192) -> dict:
        text = self.text(model=model, system_prompt=system_prompt + '\nReturn one JSON object only, without Markdown.',
                         user_prompt=user_prompt, max_tokens=max_tokens)
        if text.startswith('```json\n') and text.endswith('```'):
            text = text[8:-3].strip()
        try:
            value = json.loads(text)
        except ValueError:
            raise ValueError('MiniMax returned invalid JSON') from None
        if not isinstance(value, dict):
            raise ValueError('MiniMax JSON response must be an object')
        return value

    def tool_loop(self, *, model: str, system_prompt: str, user_prompt: str,
                  tools: list[dict], execute, max_tokens: int = 8192, max_steps: int = 4) -> dict:
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        allowed = {tool['function']['name'] for tool in tools}
        completed = []
        for _ in range(max_steps):
            try:
                message = self.complete(model=model, messages=messages, tools=tools, max_tokens=max_tokens)
            except RuntimeError:
                if not completed:
                    raise
                return {'message':'The listed tools completed successfully. The final Assistant explanation is temporarily unavailable; inspect the returned tool outputs.',
                        'calls':completed,'explanation_available':False}
            calls = message.get('tool_calls', [])
            if not calls:
                return {'message': self.answer(message), 'calls': completed}
            # Validate the whole batch before executing the first action.
            decoded = []
            for call in calls:
                name = call.get('function', {}).get('name')
                if name not in allowed or not call.get('id'):
                    raise ValueError('Assistant requested an unavailable tool')
                arguments = json.loads(call['function']['arguments'])
                if not isinstance(arguments, dict):
                    raise ValueError('Tool arguments must be an object')
                decoded.append((call, name, arguments))
            messages.append(message)
            for call, name, arguments in decoded:
                result = execute(name, arguments)
                completed.append({'tool': name, 'arguments': arguments, 'result': result})
                messages.append({'role': 'tool', 'tool_call_id': call['id'],
                                 'content': json.dumps(result, ensure_ascii=False, allow_nan=False)})
        raise RuntimeError('Assistant tool step limit reached')


def create_client(config: dict, api_key: str | None = None):
    provider = config['provider']
    if provider == 'minimax':
        return MiniMaxClient(api_key=api_key, base_url=config['base_url'])
    if provider == 'gemini':
        return GeminiClient(api_key=api_key, base_url=config['base_url'])
    raise ValueError('llm.provider must be minimax or gemini')
