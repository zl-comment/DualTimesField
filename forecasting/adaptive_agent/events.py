"""Quoted event candidates for bounded Kev choices; no invented price adjustment."""
from __future__ import annotations

import re

_PATTERNS = {
    'outage': r'\b(outage|out of service|unavailable|tripped)\b',
    'restoration': r'\b(restored|restoration|return to service|returned to service)\b',
    'reserve': r'\b(reserve|LOR[123]|lack of reserve)\b',
    'weather': r'\b(weather|temperature|storm|wind|heatwave)\b',
    'demand': r'\b(demand|load forecast)\b',
}


class EventResolver:
    def resolve(self, record, choice=None):
        if record.get('availability_basis') != 'verified':
            return {'status': 'unavailable', 'facts': [], 'choices': {'unknown': 'Abstain: unverified source'}}
        text = record.get('payload', {}).get('text', '')
        candidates = {}
        for kind, pattern in _PATTERNS.items():
            match = re.search(pattern, text, re.I)
            if match:
                # Exact local excerpt with offsets into the source text.
                start = max(text.rfind('\n', 0, match.start()), text.rfind('.', 0, match.start())) + 1
                stops = [p for p in (text.find('\n', match.end()), text.find('.', match.end())) if p >= 0]
                end = min(stops) + 1 if stops else len(text)
                start, end = max(start, match.start() - 200), min(end, match.end() + 300)
                quote = text[start:end]
                quantities = [{'value_literal': m.group(1), 'unit': m.group(2),
                               'quote_start': start + m.start(), 'quote_end': start + m.end()}
                              for m in re.finditer(r'(?<!\w)(-?\d+(?:\.\d+)?)\s*(MW|MWh|kV|°C|degrees C|%)\b', quote, re.I)]
                # Status remains unknown unless explicitly expressed; cancellation
                # wording can refer to another event, so retain a quoted cue rather
                # than silently treating it as a numeric/physical fact.
                status = 'unknown'
                status_match = re.search(r'\b(cancelled|canceled|withdrawn|supersedes|planned|scheduled|restored)\b', quote, re.I)
                if status_match:
                    status = {'cancelled': 'cancelled', 'canceled': 'cancelled', 'withdrawn': 'withdrawn',
                              'supersedes': 'replacement', 'planned': 'planned', 'scheduled': 'planned',
                              'restored': 'restoration_reported'}[status_match.group(0).lower()]
                candidates[kind] = {'event_type': kind, 'status': status,
                    'region': record.get('region', 'ALL'), 'evidence_id': record['evidence_id'],
                    'quote_span': {'start': start, 'end': end, 'text': quote},
                    'related_versions': list(record.get('supersedes', [])),
                    'explicit_quantities': quantities,
                    'valid_from': record.get('valid_from'), 'valid_to': record.get('valid_to'),
                    'validation_status': 'quoted_candidate_requires_semantic_review',
                    'version_status': record.get('version_status', 'unknown'),
                    'conflicting_evidence_ids': list(record.get('conflicting_evidence_ids', [])),
                    'numerical_price_effect': None}
        choices = {key: 'Select quoted ' + key + ' interpretation' for key in candidates}
        choices['unknown'] = 'Abstain: text does not support a reliable interpretation'
        if choice is None:
            return {'status': 'needs_choice' if candidates else 'unknown',
                    'choices': choices, 'facts': list(candidates.values()),
                    'evidence_id': record['evidence_id']}
        if choice not in choices:
            raise ValueError('Event interpretation is not an available typed choice')
        facts = [] if choice == 'unknown' else [dict(candidates[choice], validation_status='quoted_llm_selected')]
        return {'status': 'unknown' if not facts else 'interpreted', 'choices': choices,
                'facts': facts, 'evidence_id': record['evidence_id'], 'selected_choice': choice}
