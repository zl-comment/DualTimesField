"""Local System One client. No shell, file, search, or arbitrary tool execution."""
from __future__ import annotations

import json
import math
import time
import urllib.request
import urllib.error


class KevRequestError(ValueError):
    def __init__(self, record):
        super().__init__(record['error'])
        self.record = record


class KevClient:
    def __init__(self, base_url='http://127.0.0.1:8009', timeout=120):
        if base_url != 'http://127.0.0.1:8009':
            raise ValueError('This release only permits the pinned local Kev endpoint')
        self.base_url, self.timeout = base_url, timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def identity(self):
        with self.opener.open(self.base_url + '/v1/models', timeout=self.timeout) as r:
            identity = json.load(r)
        card = identity['models'][0]
        if (card['run'] != '/root/workspace/services/kev/models/kev-4b'
                or card['base'] != 'Qwen/Qwen3.5-4B-Base' or card['device'] != 'cuda'):
            raise ValueError('Unexpected Kev service identity')
        return identity

    def choose(self, state, instructions, criteria):
        body = {'model': 'kev-latest', 'state': state, 'questions': {
            'action': {'type': 'choice', 'instructions': instructions, 'criteria': criteria}}}
        data = json.dumps(body, allow_nan=False).encode()
        if len(data) > 24000:
            raise ValueError('Agent request exceeds the frozen byte budget')
        start = time.monotonic()
        request = urllib.request.Request(self.base_url + '/v1/systemone', data=data,
                                         headers={'Content-Type': 'application/json'})
        record = {'request': body}
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                result = json.load(response)
            record['response'] = result
            choice, probabilities = validate_choice(result, criteria)
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, TypeError) as exc:
            record.update(error=type(exc).__name__ + ': ' + str(exc)[:500],
                          elapsed_seconds=time.monotonic() - start)
            raise KevRequestError(record) from exc
        record.update(elapsed_seconds=time.monotonic() - start,
                      selected_probability=probabilities[choice])
        return choice, record


def validate_choice(result, criteria):
    if result.get('truncated', False):
        raise ValueError('Truncated decision evidence')
    if result.get('model') != 'kev-latest':
        raise ValueError('Unexpected response model')
    answer = result['answers']['action']
    choice, probabilities = answer['choice'], answer['probabilities']
    if answer.get('type') != 'choice' or choice not in criteria or set(probabilities) != set(criteria):
        raise ValueError('Decision outside the registered action space')
    if any(isinstance(p, bool) or not isinstance(p, (float, int)) or not math.isfinite(p)
           or not 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError('Invalid decision probabilities')
    if abs(sum(probabilities.values()) - 1) > 0.001:
        raise ValueError('Decision probabilities do not sum to one')
    if probabilities[choice] < max(probabilities.values()):
        raise ValueError('Choice contradicts the returned distribution')
    return choice, probabilities
