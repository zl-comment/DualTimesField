"""Verify the real local Kev endpoint; save identity and full smoke responses."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:8009')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(args.base_url + path, data=data,
                                     headers={'Content-Type': 'application/json'})
        with opener.open(req, timeout=120) as response:
            return json.load(response)

    identity = request('/v1/models')
    card = identity['models'][0]
    assert card['run'] == '/root/workspace/services/kev/models/kev-4b', card
    assert card['base'] == 'Qwen/Qwen3.5-4B-Base', card
    assert card['device'] == 'cuda', card
    body = {
        'model': 'kev-latest',
        'state': {'tool_status': 'failed', 'validated_base_forecast_available': True,
                  'policy': 'When the tool fails, keep the validated base forecast.'},
        'questions': {'next_action': {
            'type': 'choice',
            'instructions': 'Choose the action required by the stated policy.',
            'criteria': {'keep_base': 'Keep the validated base forecast.',
                         'use_failed_tool': 'Use the failed tool result.'}}}}
    started = time.monotonic()
    responses = [request('/v1/systemone', body) for _ in range(2)]
    for response in responses:
        answer = response['answers']['next_action']
        assert answer['choice'] == 'keep_base', response
        probs = answer['probabilities']
        assert set(probs) == {'keep_base', 'use_failed_tool'}, response
        assert all(isinstance(p, (float, int)) and 0 <= p <= 1 for p in probs.values())
        assert abs(sum(probs.values()) - 1) < 1e-5, probs
        assert response.get('truncated', False) is False, response
    assert responses[0]['answers'] == responses[1]['answers']
    report = {'status': 'PASS', 'base_url': args.base_url, 'identity': identity,
              'request_sha256': hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
              'request': body, 'responses': responses,
              'elapsed_seconds': time.monotonic() - started,
              'scope': 'deployment smoke only; not forecasting quality validation'}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'report': str(out),
                      'elapsed_seconds': report['elapsed_seconds']}))


if __name__ == '__main__':
    main()
