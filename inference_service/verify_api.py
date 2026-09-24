import argparse
import json
import math
import os
from pathlib import Path
import urllib.request

def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8000');a=p.parse_args()
    reference=json.loads((Path(__file__).parent/'examples/verification.json').read_text())
    expected=reference.pop('expected_prediction');headers={'Content-Type':'application/json'}
    if os.environ.get('API_KEY'):headers['X-API-Key']=os.environ['API_KEY']
    request=urllib.request.Request(a.url.rstrip('/')+'/predict',json.dumps(reference).encode(),headers)
    with urllib.request.urlopen(request,timeout=600) as response:result=json.load(response)
    if len(result['results'])!=2:raise RuntimeError('Unexpected result count')
    worst=0.0
    for i,row in enumerate(result['results']):
        if row['sample_id']!=reference['pairs'][i]['sample_id']:raise RuntimeError('Result order/ID mismatch')
        value=float(row['predicted_pkoff']);difference=abs(value-expected[i]);worst=max(worst,difference)
        if not math.isfinite(value) or difference>1e-4:raise RuntimeError('Numerical smoke mismatch: row %d / %g'%(i,difference))
    print('PASS: seed-43 checkpoint matches CPU export reference; max abs difference',worst)

if __name__=='__main__':main()
