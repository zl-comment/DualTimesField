"""Verify upstream large-file checksums and freeze every downloaded runtime file."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path('/root/workspace/services/kev')
    revisions = json.loads((root / 'model_revisions.json').read_text())
    official = json.loads((root / 'official_weight_hashes.json').read_text())
    manifest = {}
    for repo, revision in revisions.items():
        directory = root / 'models' / repo.split('/')[-1]
        for filename in revision['files']:
            if filename == 'training_metrics.json' or not (
                    filename.endswith(('.json', '.safetensors', '.pt', '.txt', '.model', '.jinja'))
                    or filename == 'LICENSE'):
                continue
            path = directory / filename
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 << 20), b''):
                    digest.update(block)
            sha = digest.hexdigest()
            expected = official.get(repo, {}).get(filename)
            if expected and (sha != expected['sha256'] or path.stat().st_size != expected['size']):
                raise ValueError('Upstream checksum mismatch: ' + str(path))
            if path.suffix == '.json':
                json.loads(path.read_text())
            manifest[str(path)] = {'sha256': sha, 'size': path.stat().st_size,
                                  'revision': revision['revision'],
                                  'upstream_lfs_verified': bool(expected)}
    (root / 'verified_weight_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('PASS: verified and fingerprinted', len(manifest), 'model files')


if __name__ == '__main__':
    main()
