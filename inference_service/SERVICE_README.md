# MGCA KinetX seed-43 screening service

This deployable archive contains one full-clean-KinetX checkpoint (seed 43, 55 epochs), the frozen MGCA implementation, a precomputed KinetX protein-feature warm cache, and the FastAPI service. It does not contain optimizer/resume state or training data.

Use the same model environment as training, then install only the service layer:

```bash
cd mgca_kinetx_api
python -m pip install -r requirements_service.txt
export ESM2_PATH=/root/private_data/DP/pretrained_model/esm2_t36
bash start_service.sh
```

The service starts accepting requests only after it has verified hashes, loaded ESM2 and the seed-43 MGCA checkpoint, restored the KinetX protein cache, and run real warmup forwards. Keep one worker; multiple workers duplicate model memory.

Verify the live service from a second terminal:

```bash
curl http://127.0.0.1:8000/readyz
python verify_api.py
curl -X POST http://127.0.0.1:8000/screen \
  -H 'Content-Type: application/json' --data-binary @examples/screen.json
```

For a large line-delimited SMILES library and one FASTA record:

```bash
python client_screen.py --fasta target.fasta --smiles library.smi \
  --batch-size 512 --output screening_results.csv
```

The client saves every scored molecule and then ranks by descending predicted pKoff. A first request for an unseen protein runs ESM2; repeat requests reuse the in-memory feature cache. Invalid inputs fail the whole request and are not silently discarded.

Default binding is local-only. Prefer SSH port forwarding. Non-loopback binding is refused unless `API_KEY` is set; production exposure additionally requires a trusted TLS reverse proxy, network access control and rate limiting.

This is a single-checkpoint research prediction, so no ensemble variance or calibrated confidence interval is available. Larger pKoff means slower predicted dissociation. The model was trained on the entire clean KinetX cohort; its original benchmark test split is therefore no longer an independent evaluation set.
