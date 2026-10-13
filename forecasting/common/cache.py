"""Read and verify a single frozen aligned split without loading other periods."""
import numpy as np
from ..aligned_inputs import INPUT_KEYS, AlignedSplit
from ..aligned_experiment import sha, read_json


def load_split(root, manifest, region, name):
    """Verify and map only the requested split; test is opened after selection."""
    cache = root / 'data_cache' / region
    if sha(cache / 'hashes.json') != manifest['cache_hashes'][region]:
        raise ValueError('Shared cache manifest changed')
    hashes = read_json(cache / 'hashes.json')
    paths = [*(f'input__{key}.npy' for key in INPUT_KEYS), 'actual.npy', 'target.npy',
             'origin_unix.npy', 'target_time_unix.npy', 'metadata.json']
    for path in paths:
        if sha(cache / name / path) != hashes[f'{name}/{path}']:
            raise ValueError(f'Cached {region}/{name}/{path} changed')
    def array(key):
        return np.load(cache / name / f'{key}.npy', mmap_mode='r', allow_pickle=False)
    return AlignedSplit({key: array(f'input__{key}') for key in INPUT_KEYS}, array('actual'),
                        array('origin_unix'), array('target_time_unix'),
                        read_json(cache / name / 'metadata.json'), array('target'))
