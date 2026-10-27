"""Archive a completed frozen upper-scale audit without changing its artifacts."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

source=Path(sys.argv[1]).resolve()
destination=Path(sys.argv[2]).resolve()
verification=json.loads((source/'verification.json').read_text())
if verification['completed']!=['audit']:
    raise ValueError('Only completed audits can be archived')
for relative, expected in verification['artifact_sha256'].items():
    if hashlib.sha256((source/relative).read_bytes()).hexdigest()!=expected:
        raise ValueError('Completed artifact changed: '+relative)
independent=json.loads((source/'independent_verification.json').read_text())
if independent.get('status')!='passed':
    raise ValueError('Require independent verification before archiving')
destination.mkdir(parents=True,exist_ok=False)
files={'manifest.json':'run_manifest.json','verification.json':'verification.json',
       'independent_verification.json':'independent_verification.json'}
for name in ('summary.json','decision.json','definitions.json','prediction_manifest.json','sources.json','scores.csv','regional.csv','macro.csv'):
    files['audit/'+name]=name
for relative,name in files.items():
    shutil.copyfile(source/relative,destination/name)
shutil.copyfile(__file__,destination/'archive_results.py')
manifest=dict(source_output=str(source),run_id=verification['run_id'],
    copies={name:dict(source=str(source/relative),sha256=hashlib.sha256((destination/name).read_bytes()).hexdigest())
            for relative,name in files.items()})
(destination/'copy_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(destination)
