"""Copy verified event-allocation evidence; leave large predictions at source."""
from pathlib import Path
import hashlib,json,shutil,sys
source=Path(sys.argv[1]).resolve()
destination=Path(sys.argv[2]).resolve()
def read(p): return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
v=read(source/'verification.json')
assert set(v['completed'])=={'audit','episodes'}
for name in ('independent_verification.json','independent_episode_verification.json'):
    assert read(source/name)['status']=='passed',name
for name,digest in v['artifact_sha256'].items():
    assert sha(source/name)==digest,name
destination.mkdir(parents=True,exist_ok=False)
files={'run_manifest.json':'manifest.json','verification.json':'verification.json',
       'independent_verification.json':'independent_verification.json',
       'independent_episode_verification.json':'independent_episode_verification.json'}
for name in ('summary.json','decision.json','definitions.json','prediction_manifest.json','sources.json','excluded.json','scores.csv','regional.csv','macro.csv'):
    files[name]='audit/'+name
files.update({'episode_summary.json':'episodes/summary.json',
              'hourly_and_episode_metrics.csv':'episodes/hourly_and_episode_metrics.csv'})
records={}
for name,relative in files.items():
    path=source/relative
    shutil.copyfile(path,destination/name)
    records[name]={'source':str(path),'sha256':sha(path)}
shutil.copyfile(__file__,destination/'archive_results.py')
(destination/'copy_provenance.json').write_text(json.dumps({'source_output':str(source),'run_id':v['run_id'],'copies':records},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'copied':len(records),'output':str(destination)},indent=2))
