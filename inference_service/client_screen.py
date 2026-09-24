"""Batch a SMILES text library against one FASTA file; keep all API results."""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.request
import urllib.error
import core as c

def main():
    p=argparse.ArgumentParser();p.add_argument('--fasta',type=Path,required=True);p.add_argument('--smiles',type=Path,required=True)
    p.add_argument('--url',default='http://127.0.0.1:8000');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=512);p.add_argument('--allow-truncation',action='store_true');a=p.parse_args()
    if not 1<=a.batch_size<=4096:raise ValueError('Batch size must be 1..4096')
    if a.output.exists():raise RuntimeError('Output exists; choose a new output path')
    lines=a.fasta.read_text().splitlines()
    if sum(line.startswith('>') for line in lines)>1:raise ValueError('Provide exactly one FASTA record')
    fasta=c.sequence(''.join(line for line in lines if not line.startswith('>')))
    library=[s.strip() for s in a.smiles.read_text().splitlines() if s.strip()]
    if not library:raise ValueError('Empty compound library')
    headers={'Content-Type':'application/json'}
    if os.environ.get('API_KEY'):headers['X-API-Key']=os.environ['API_KEY']
    output=[]
    for start in range(0,len(library),a.batch_size):
        pairs=[dict(fasta=fasta,smiles=s,sample_id='compound_%08d'%(start+j+1)) for j,s in enumerate(library[start:start+a.batch_size])]
        request=urllib.request.Request(a.url.rstrip('/')+'/predict',data=json.dumps(dict(pairs=pairs,allow_truncation=a.allow_truncation)).encode(),headers=headers)
        for attempt in range(6):
            try:
                with urllib.request.urlopen(request,timeout=600) as response:result=json.load(response)
                break
            except urllib.error.HTTPError as e:
                if e.code not in (429,503) or attempt==5:raise
                time.sleep(min(2**attempt,16))
        for r in result['results']:
            output.append(r)
        c.atomic_json(a.output.with_suffix('.progress.json'),dict(completed=len(output),total=len(library),last_response=result['timing']))
    output.sort(key=lambda r:(-r['predicted_pkoff'],r['sample_id']))
    for i,r in enumerate(output,1):r['rank']=i
    c.atomic_csv(a.output,output);print('Saved',len(output),'ranked compounds to',a.output)

if __name__=='__main__':main()
