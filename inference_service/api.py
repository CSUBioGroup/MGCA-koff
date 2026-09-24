"""Single-worker FastAPI, ready only after lifespan model load and warmup."""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
import hmac
import os
from pathlib import Path
import time
import uuid
from typing import Optional,List
from fastapi import FastAPI,HTTPException,Depends,Header
from pydantic import BaseModel,Field,ConfigDict,constr
from starlette.concurrency import run_in_threadpool

MAX_PAIRS=int(os.environ.get('MAX_PAIRS','4096'))
if not 1<=MAX_PAIRS<=4096:raise ValueError('MAX_PAIRS must be 1..4096')

class Pair(BaseModel):
    model_config=ConfigDict(extra='forbid')
    fasta:str=Field(min_length=1,max_length=10000)
    smiles:str=Field(min_length=1,max_length=4096)
    sample_id:Optional[str]=Field(default=None,max_length=200)

class PredictRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    pairs:List[Pair]=Field(min_length=1,max_length=MAX_PAIRS)
    allow_truncation:bool=False

class ScreenRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    fasta:str=Field(min_length=1,max_length=10000)
    smiles:List[constr(min_length=1,max_length=4096)]=Field(min_length=1,max_length=MAX_PAIRS)
    top_k:Optional[int]=Field(default=None,ge=1,le=MAX_PAIRS)
    allow_truncation:bool=False

@asynccontextmanager
async def lifespan(app):
    from engine import Engine
    app.state.ready=False;app.state.busy=asyncio.Lock()
    app.state.engine=await run_in_threadpool(Engine,Path(os.environ.get('BUNDLE_DIR',Path(__file__).parent)),
                                           Path(os.environ['ESM2_PATH']),os.environ.get('DEVICE','cuda:0'))
    app.state.ready=True
    try:yield
    finally:
        app.state.ready=False;app.state.engine.close()

app=FastAPI(title='Frozen MGCA KinetX screening',version='1.0',lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)

# Limit actual bytes, not only Content-Length, before JSON parsing.
class BodyLimit:
    def __init__(self,app):self.app=app;self.limit=int(os.environ.get('MAX_BODY_BYTES','8388608'))
    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        messages=[];size=0
        while True:
            message=await receive()
            if message['type']=='http.disconnect':return
            size+=len(message.get('body',b''))
            if size>self.limit:
                await send(dict(type='http.response.start',status=413,headers=[(b'content-type',b'application/json')]))
                await send(dict(type='http.response.body',body=b'{"detail":"Request body too large"}'));return
            messages.append(message)
            if not message.get('more_body',False):break
        async def replay():
            if messages:return messages.pop(0)
            return await receive()
        await self.app(scope,replay,send)

app.add_middleware(BodyLimit)

async def authorize(x_api_key:Optional[str]=Header(default=None)):
    expected=os.environ.get('API_KEY','')
    if expected and (not x_api_key or not hmac.compare_digest(x_api_key,expected)):
        raise HTTPException(status_code=401,detail='Invalid API key')

@app.get('/healthz')
async def health():return dict(status='alive')

@app.get('/readyz',dependencies=[Depends(authorize)])
async def ready():
    if not getattr(app.state,'ready',False):raise HTTPException(503,'Warming up')
    return dict(status='ready',models=1,seed=43,dataset='KinetX_clean_5446',startup_seconds=app.state.engine.startup_seconds)

async def infer(pairs,allow):
    if not getattr(app.state,'ready',False):raise HTTPException(503,'Not ready')
    if app.state.busy.locked():raise HTTPException(429,'GPU busy; retry with backoff')
    request_id=str(uuid.uuid4());started=time.perf_counter()
    async with app.state.busy:
        try:
            result=await run_in_threadpool(app.state.engine.predict,pairs,allow)
        except ValueError as e:raise HTTPException(422,str(e))
        except Exception:
            import logging
            logging.exception('Inference failure request_id=%s',request_id)
            raise HTTPException(500,'Inference failed; see server log, request_id='+request_id)
    print('request_id=%s pairs=%d elapsed=%.4f'%(request_id,len(pairs),time.perf_counter()-started),flush=True)
    result['request_id']=request_id;return result

@app.post('/predict',dependencies=[Depends(authorize)])
async def predict(request:PredictRequest):
    return await infer([r.model_dump() for r in request.pairs],request.allow_truncation)

@app.post('/screen',dependencies=[Depends(authorize)])
async def screen(request:ScreenRequest):
    pairs=[dict(fasta=request.fasta,smiles=s,sample_id='compound_%06d'%(i+1)) for i,s in enumerate(request.smiles)]
    # Enforce same per-field limits as /predict before model work.
    pairs=[Pair(**r).model_dump() for r in pairs]
    result=await infer(pairs,request.allow_truncation)
    result['results'].sort(key=lambda r:(-r['predicted_pkoff'],r['sample_id']))
    for i,r in enumerate(result['results'],1):r['rank']=i
    result['total_screened']=len(result['results'])
    if request.top_k is not None:result['results']=result['results'][:request.top_k]
    result['ranking']='pKoff descending: slower predicted dissociation first';return result
