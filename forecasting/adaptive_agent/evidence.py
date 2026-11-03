"""Frozen evidence warehouse. Visibility is filtered before ranking/version handling."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _timestamp(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(name + ' must be integer Unix seconds')
    return value


class EvidenceStore:
    def __init__(self, records=()):
        self._records = {}
        for source in records:
            record = deepcopy(source)
            for name in ('evidence_id', 'source_id', 'version', 'availability_basis'):
                if not isinstance(record.get(name), str) or not record[name]:
                    raise ValueError('Missing evidence ' + name)
            if record.get('first_seen_role', 'availability') not in ('availability', 'archive_fetch'):
                raise ValueError('Unknown first-seen timestamp role')
            if record.get('first_seen_role') == 'archive_fetch' and record['availability_basis'] != 'verified':
                raise ValueError('Archive-fetch mode requires independently verified publication')
            if record['availability_basis'] not in ('verified', 'archive_proxy', 'unknown'):
                raise ValueError('Unknown evidence availability basis')
            for name in ('published_at', 'first_seen_at', 'valid_from', 'valid_to'):
                if record.get(name) is not None:
                    _timestamp(record[name], name)
            if record['availability_basis'] == 'verified' and record.get('published_at') is None:
                raise ValueError('Verified evidence requires publication time')
            if record.get('first_seen_at') is not None and record.get('published_at') is not None and record['first_seen_at'] < record['published_at']:
                raise ValueError('First seen cannot precede publication')
            if record.get('valid_from') is not None and record.get('valid_to') is not None and record['valid_to'] <= record['valid_from']:
                raise ValueError('Empty evidence validity interval')
            if not isinstance(record.get('payload'), dict):
                raise ValueError('Evidence payload must be an object')
            text = record['payload'].get('text', '')
            if not isinstance(text, str):
                raise ValueError('Evidence text must be a string')
            actual_hash = _hash(record['payload'])
            if record.get('content_hash', actual_hash) != actual_hash:
                raise ValueError('Evidence content hash mismatch')
            record['content_hash'] = actual_hash
            region = record.get('region', 'ALL')
            if not isinstance(region, (str, list)):
                raise ValueError('Evidence region must be string or list')
            if not isinstance(record.get('supersedes', []), list) or any(not isinstance(v, str) for v in record.get('supersedes', [])):
                raise ValueError('Supersedes must contain evidence IDs')
            if record['evidence_id'] in record.get('supersedes', []):
                raise ValueError('Evidence cannot replace itself')
            key = record['evidence_id']
            if key in self._records and self._records[key] != record:
                raise ValueError('Conflicting immutable evidence ID')
            self._records[key] = record
        # Verify explicit relation graph without inventing missing predecessors.
        def visit(key, active, done):
            if key in active:
                raise ValueError('Cyclic evidence replacement chain')
            if key in done or key not in self._records:
                return
            active.add(key)
            for previous in self._records[key].get('supersedes', []):
                visit(previous, active, done)
            active.remove(key)
            done.add(key)
        completed = set()
        for key in self._records:
            visit(key, set(), completed)
        self.fingerprint = _hash(sorted(self._records.values(), key=lambda x: x['evidence_id']))

    @classmethod
    def from_jsonl(cls, path):
        records = []
        with Path(path).open() as stream:
            for line in stream:
                if line.strip():
                    records.append(json.loads(line))
        return cls(records)

    def _visible(self, context):
        out = []
        for row in self._records.values():
            regions = row.get('region', 'ALL')
            if isinstance(regions, str):
                regions = [regions]
            if context.region not in regions and 'ALL' not in regions:
                continue
            # Verified publication is required. A first-seen receipt defaults to
            # actual availability and delays visibility. An explicitly tagged
            # archive_fetch is merely acquisition metadata for independently
            # verified historical publication. No proxy clock is promoted.
            if row['availability_basis'] != 'verified':
                continue
            if row['published_at'] > context.cutoff or (row.get('first_seen_role', 'availability') != 'archive_fetch' and row.get('first_seen_at') is not None and row['first_seen_at'] > context.cutoff):
                continue
            out.append(row)
        return out

    def search(self, context, query='', limit=8):
        if not 1 <= limit <= 8 or not isinstance(query, str):
            raise ValueError('Invalid evidence search budget/query')
        visible = self._visible(context)
        # Only a visible explicit replacement can suppress its predecessor.
        superseded = {previous for row in visible for previous in row.get('supersedes', [])}
        terms = set(re.findall(r'\w+', query.lower()))
        ranked = []
        for row in visible:
            if row['evidence_id'] in superseded:
                continue
            words = set(re.findall(r'\w+', row['payload'].get('text', '').lower()))
            score = len(terms & words)
            if terms and not score:
                continue
            ranked.append((score, row['published_at'], row['evidence_id'], row))
        ranked.sort(key=lambda x: (-x[0], -x[1], x[2]))
        items, duplicate_groups = [], {}
        for _, _, _, row in ranked:
            # Reposted identical content does not become independent evidence.
            # Preserve explicit version-chain heads for conflict review.
            duplicate_key = _hash({'content': row['content_hash'], 'region': row.get('region', 'ALL'),
                                   'valid_from': row.get('valid_from'), 'valid_to': row.get('valid_to'),
                                   'status': row.get('status'),
                                   'explicit_chain_version': row['evidence_id'] if row.get('chain_id') else None})
            if duplicate_key in duplicate_groups:
                duplicate_groups[duplicate_key]['duplicate_evidence_ids'].append(row['evidence_id'])
                continue
            result = self._with_relations(row, visible)
            result['duplicate_evidence_ids'] = []
            duplicate_groups[duplicate_key] = result
            items.append(result)
        items = items[:limit]
        return {'status': 'available' if items else 'real_data_unavailable', 'items': items,
                'warehouse_fingerprint': self.fingerprint,
                'visible_count': len(visible), 'verified_publication_required': True}

    def read(self, context, evidence_id):
        row = next((r for r in self._visible(context) if r['evidence_id'] == evidence_id), None)
        if row is None:
            raise ValueError('Evidence is unavailable at this cutoff/region')
        return self._with_relations(row, self._visible(context))

    def _with_relations(self, row, visible):
        result = self._public(row)
        superseded = {previous for item in visible for previous in item.get('supersedes', [])}
        heads = [item for item in visible if item['evidence_id'] not in superseded
                 and row.get('chain_id') and item.get('chain_id') == row.get('chain_id')]
        result['version_status'] = ('superseded' if row['evidence_id'] in superseded else
                                    'conflict_unresolved' if len(heads) > 1 else 'visible')
        result['conflicting_evidence_ids'] = [item['evidence_id'] for item in heads
                                               if item['evidence_id'] != row['evidence_id']] if len(heads) > 1 else []
        result['missing_predecessor_ids'] = [key for key in row.get('supersedes', [])
                                             if key not in {item['evidence_id'] for item in visible}]
        return result

    @staticmethod
    def _public(row):
        # Do not expose paths or arbitrary payload instructions/metadata.
        fields = ('evidence_id', 'source_id', 'version', 'published_at', 'first_seen_at',
                  'availability_basis', 'first_seen_role', 'valid_from', 'valid_to', 'region', 'content_hash',
                  'supersedes', 'status', 'chain_id')
        result = {k: deepcopy(row[k]) for k in fields if k in row}
        result['payload'] = {'text': row['payload'].get('text', '')[:8000]}
        return result
