# ml4paleo v2: architecture and build plan

## Context

v1 (`webapp/` Flask + three polling runner scripts + `volume/jobs.json`, plus the `ml4paleo/` library) works for one user on one box. A code review and the open issues show structural limits: no auth, heavy work in HTTP requests on one sync worker, a JSON file as the job store, stuck jobs after crashes, a random-forest-only model with no ignore label, an annotator that serves random 512² patches with no provenance or undo, and local-disk-only storage. v2 is a ground-up re-architecture on branch `v2` (created from `master` at `e97ceab`, after #74–#76 merged). Goals: accounts and sharing, one annotation app (brush, polygon, auto-annotation) with provenance that scales from small training cubes to hand-annotating a whole volume, swappable neural-net segmentation, a real job system that can offload to other machines or the cloud, and storage that works on local disk, S3, or GCS.

## Decisions

| Area | Decision |
|---|---|
| Repo | Monorepo uv workspace: `ml4paleo/` (library), `server/` (FastAPI), `worker/`, `web/` (SvelteKit), `deploy/`, `tests/`. Delete `webapp/` once the v1 importer works. Python ≥ 3.12. |
| Database | Postgres only: app data, op log, and the job queue. No Redis. Alembic migrations. |
| Auth | Local username/password (argon2id via pwdlib, Postgres sessions, CSRF, rate limits, optional TOTP; TOTP required for admins) + admin-configured OIDC (Authlib). |
| Signup | Open signup (admin can switch to invite-only). Optional SMTP for verification, reset, job-done notices. |
| Sharing | Owner + collaborators, full access, no role tiers. |
| Project | One volume per project (optionally multichannel), multi-class label set. |
| Quotas | User-facing limits are plain: **storage** (default 10 GB) and **trained models** (count of models a user keeps; deleting one frees a slot). Every dimension (also optional CPU-hours, GPU-hours) is independently configurable per deploy; unset = unlimited. Unlimited projects within quota. "Request more" button → short form → email (or admin inbox when no SMTP). |
| Scheduling | FIFO within a priority tier (interactive > normal > background). Resources, not quotas, limit concurrency. Background-tier jobs never trigger scale-up; they run on any idle compatible worker ("drain backlogs on idle capacity"), and burst nodes drain matching queued work before shutting down. |
| Compute | One worker image (CPU and CUDA variants), three ways to run it: local containers (GPU worker auto-enabled when a GPU is detected), remote pull workers (outbound HTTPS only), SkyPilot burst (AWS/GCP/K8s/SSH, scale to zero). |
| Burst limits | Per job type pools instead of cost caps: `pools.<jobtype>.node_type` (accelerator/instance) × `pools.<jobtype>.max_nodes` (and optional `min_nodes`, `idle_minutes`). |
| Storage | S3 API everywhere. Single box runs SeaweedFS (MinIO CE archived Apr 2026); AWS S3, GCS, R2 by config. zarr-python 3 + obstore, one code path chosen by URL. Images: OME-Zarr 0.5, axes `c,z,y,x`, 64³ chunks in 512³ shards, multiscale. |
| Frontend | SvelteKit 2 / Svelte 5 + TS static SPA served by the API. WebGL2 tri-planar viewer + 3D preview + self-hosted Neuroglancer on the same origin. |
| Labels | Raster, content-addressed 64³ chunks + Postgres op log (provenance, undo/redo, audit). Unlabeled = ignored in training unless inside an ROI marked complete. |
| Training unit | User-placed ROIs (cubes/slices) + active-learning suggestions after each training run. Hand-annotating the whole volume stays possible; human labels override predictions. |
| Models | RF (CPU), nnU-Net v2, MONAI SegResNet behind one plugin protocol. Interactive: nnInteractive (default; non-commercial weights, shown in UI) and SAM 2.1 (admin-switchable). |
| Auto-annotation | Model-proposal overlay, click-to-segment (GPU), slice propagation (CPU), classic tools (threshold brush, flood fill/wand, morphology). |
| v1 import | v2 replaces v1 on the same origin. Logged-in visit to `/job/<6hex>` claims and imports it (first visitor wins). Bulk importer reads v1 `localStorage["jobs"]`. Rate limit + admin reassign. |
| IaC | OpenTofu modules `deploy/tofu/modules/{aws,gcp}`. |
| Phasing | M1 vertical slice on one box → M2 GPU models, remote workers, suggestions → M3 interactive, propagation, SkyPilot, OpenTofu. |

## Architecture

### Core invariants
1. **Projects have no status enum.** State is derived from committed artifact heads and active pipelines, so the web app and workers can never overwrite each other's status (in v1, saving an annotation could cancel a queued training run, and a retrain could hide finished downloads).
2. **Committed artifacts are immutable.** Each lives at `projects/<pid>/artifacts/<aid>/`; "current" is a DB pointer (`artifact_heads`). Commit is one DB transaction after the worker writes `_MANIFEST.json` last. Partial outputs never become "latest"; export caches keyed by source artifact can't go stale (v1 could cache a partial segmentation zip forever).
3. **One axis boundary.** Storage is `c,z,y,x`. The only XYZ↔ZYX transpose is in `ml4paleo/ome.py`, covered by golden tests (v1 shipped both a transposed PNG export and mirrored meshes).
4. **Heavy work never runs in a request.** The API only enqueues, streams bytes, or applies small label ops.

### Layout
```
ml4paleo/  storage/ (StorageGrant, object_store, zarr_store, proxy_store)  ome.py  blocks.py
           labels/ (value conventions, chunk codec, mask deltas)  protocol.py (pydantic models shared by server+worker)
           volume_providers/ (keep imagevp, dicomvp w/ spacing, numpyvp; port zarrvp to zarr 3, still reads v1 zarr v2)
           segmentation/ (plugin protocol; plugins/rf.py, nnunet.py, monai.py; predict_block with halo)
           meshing/ (mesh_block with 1-voxel overlap, merge_meshes; keeps xyz/mm + mesh_info.json)
server/ml4paleo_server/  app.py settings.py cli.py db/ migrations/ auth/ api/ jobs/ storage/broker.py
                         gateway.py (zarr/data proxy) housekeeper.py scaler/ quotas.py audit.py email.py
worker/ml4paleo_worker/  main.py client.py context.py caps.py handlers/{ingest,pyramid,labels,train,predict,mesh,export,v1import,propagate,interactive}.py
web/src/  routes/{login,signup,projects,p/[pid]/{annotate,rois,models,results,history,settings},admin,import}  lib/{viewer,chunks,labels,tools,ops}
deploy/   compose/ (caddy, api, housekeeper, postgres, seaweedfs, worker-cpu, worker-gpu[profile], scaler[profile])
          skypilot/worker.yaml  tofu/modules/{aws,gcp} + examples/
```
Library extras keep the server light: the base install (numpy, zarr≥3, obstore, pydantic, pillow, zstandard) is all the API server gets; `[dicom]`, `[rf]`, `[mesh]`, and later `[torch]` are for workers, and `[v1]` holds the old Flask app's dependencies until it is deleted. `requirements.txt`, intern, and the `numcodecs<0.16` pin are gone.

Reused from v1: `_extract_zip_archive`, `_should_ignore_source_file`, `_get_volume_provider` (`webapp/conversionrunner.py`) → `handlers/ingest.py`; `DicomVolumeProvider` spacing logic; `ChunkedMesher` axis/winding fix and `mesh_info.json` (`ml4paleo/meshing/__init__.py`); `_default_features_func` (`ml4paleo/segmentation/rf.py`); `_foreground_metrics` (`webapp/segmentrunner.py`) for per-class metrics; `normalize_annotation_volume` windowing (without the 8-bit cast); v1 sample metadata helpers `get_annotation_pairs`, `load_annotation_sample_metadata`, `annotation_sample_metadata_for_z` (`webapp/apputils.py`) for the importer; bossypaints' viewport loader (`ImageCache.ts`, `getViewportChunkWindow`, `getVisibleChunksForWindow`, progressive levels, `cancelRequestsExcept`) and propagation registry (`server/bossypaints/propagation.py`), both Apache-2.0.

### Data model (Postgres, UUIDv7 keys)
- Identity: `users` (status, is_admin, quota_override jsonb), `sessions` (hashed token), `oidc_providers`, `oidc_identities`, `auth_tokens` (verify/reset/invite), `rate_limits` (unlogged), `site_settings`, `quota_requests`.
- Projects: `projects` (owner, `v1_job_id` unique), `project_members`, `volumes` (shape_czyx, dtype, voxel_size_zyx), `channels` (window), `label_sets`, `label_classes` (0 = unlabeled, 1 = background, 2–254 classes, never reused).
- Labels: `label_chunk(layer, cz, cy, cx, version, class_sha, source_sha, n_labeled, class_counts)`, `op(op_id, op_seq, user/worker, client_op_id, kind, source, tool_params, model_id, job_id, bbox, reverts, conflicts)`, `op_chunk(op_id, chunk, base_version, new_version, new_sha, claim)`, `roi(bbox, kind, status open|complete|skipped, split train|val, origin, score)`, `roi_suggestion`, `training_set(id = sha256 manifest)`, `model` (plugin, version, params, training_set_id, parent, class_map, metrics, artifact).
- Artifacts: `artifacts(kind, state staging|committed|superseded|failed|deleted, uri, bytes, inputs, produced_by_job, cache_key, expires_at)`, `artifact_heads(project_id, slot, artifact_id)`, `uploads` (S3 multipart).
- Queue: `jobs(root_id, parent_id, kind, tier, status blocked|queued|leased|succeeded|failed|cancelled, payload, required_labels[], min_vram_gb, weight, attempts, lease_worker_id, lease_token, lease_expires_at, progress, idempotency_key, scale_trigger bool)`, `job_deps`, `job_attempts`, `workers(token_hash, pool local|remote|burst, labels[], vram_gb, slots, status, last_seen)`, `burst_nodes`.
- Accounting: `usage_ledger`, `user_usage(storage_bytes, trained_models)`, `audit_log`, `v1_claims`, `email_outbox`.

### Worker protocol (`/api/worker/v1`, `Bearer m4pw_<random>`, server stores sha256)
`POST /hello` · `POST /claim` (long-poll ≤25 s, woken by LISTEN/NOTIFY with a 5 s poll as backup; returns job + StorageGrants) · `POST /jobs/{id}/heartbeat` (30 s; 120 s lease; carries progress; returns cancel flag, refreshed grants) · `POST /jobs/{id}/complete` (idempotent) · `POST /jobs/{id}/fail` (`retryable` or not) · `POST /jobs/{id}/release` (a stopping worker hands a job back without using an attempt) · proxy object endpoints for workers without direct storage credentials (step 9). Messages live in `ml4paleo/protocol.py`, shared by server and worker. Every claim carries the worker's caps (kinds, labels, VRAM), so processes that share a token (replicas, or the CPU and GPU workers on one box) can differ; the worker row keeps the last caps seen, and "online" is derived from `last_seen_at`. Every state change is conditional on the lease token (stored hashed); stale leases get 409 and discard output. The housekeeper reaper requeues expired leases with backoff (30 s doubling, max 3 attempts), then fails the job and cancels the rest of its pipeline; it also queues any blocked job whose dependencies all succeeded, as a safety net. Claim query: `FOR UPDATE SKIP LOCKED`, filtered by `kind = ANY(caps.kinds)`, `required_labels <@ caps.labels`, and VRAM, ordered by tier then pipeline submission time (FIFO). Dependencies stay within one pipeline. When parents finish together, each locks the waiting children before checking them, so the last parent to commit always releases the child. A pipeline is cancelled by setting its root job's flag: `cancel_pipeline` waits only for the root row and skips busy rows, and claims, reports, heartbeats, and the reaper all honor the root's flag, so nothing slips through. Locks follow one order (the job, then its pipeline root, then waiting children; workers last) and use `FOR NO KEY UPDATE`, which foreign-key checks don't block. `enqueue` re-reads dependencies under a share lock. Admin API: worker tokens (`/api/admin/workers`), the job list, cancel, and a `noop` diagnostic job; `ml4paleo-server check-workers` runs one from the command line.

### Pipelines (DAG in `jobs` + `job_deps`, progress = Σ weight·progress over `root_id`, SSE to the browser)
- **Upload:** presigned S3 multipart direct from the browser, resumable (upload id in IndexedDB), quota checked at create.
- **Ingest:** `ingest.probe` (zip-bomb guards, DICOM sort, spacing, channels) → N `ingest.slab` (each writes whole shards) → `pyramid.level` per level → `artifact.finalize` (OME metadata, histogram, specimen mask on coarsest level).
- **Train + predict:** `training_set` manifest (REPEATABLE READ snapshot: image id, label `op_seq`, ROIs, free labeled regions, chunk shas) → `train` (plugin caps choose CPU/GPU) → `predict.shard` fan-out (512³ shards + halo, outputs classes + uncertainty + 64³ uncertainty grid) → pyramid → finalize → `suggest_rois`. `predict.region` (ROI-scoped, high tier) feeds the proposal overlay.
- **Final segmentation:** `compose_final` per shard = prediction → per-class dust removal (`cc_block` → `cc_merge` union-find → `cc_apply`) → human-label override (`where(label≠0, label, pred)`, complete-ROI unlabeled = background).
- **Mesh:** `mesh.block` (1-voxel overlap; downsample mode/max/gaussian, #22/#24) → `mesh.merge` per class (STL/OBJ/GLB + Neuroglancer precomputed).
- **Export:** zarr-zip, TIFF/PNG stack (in upload orientation), mesh bundle; cached by `(source_aid, fmt, params)`, 7-day expiry, presigned GET.
- **v1 import:** single job on a worker labelled `v1-volume` (v1 volume mounted read-only).

### Label edit path
- Client rasterizes every committed action into per-chunk bit-packed mask deltas (geometry kept in `tool_params` for audit); one op = one stroke/polygon/fill/accept. Ops queue in IndexedDB, flush within 250 ms, retried idempotently by `client_op_id`. Status chip: Saved / Saving n / Offline n.
- Server writer (one transaction per op): idempotency check → lock chunk rows in sorted order → rebase on stale `base_version` (strict ops 409 on overlap; brush ops last-writer-wins per voxel) → write new content-addressed blobs `labels/{layer}/blobs/{sha}` → log `op` + `op_chunk` → NOTIFY (collaborators refetch via ETag) → coalesced `label_pyramid` job.
- Each applied delta records a **claim**: the voxels the op actually wrote (after `only_if`) and the values it wrote there. A chunk's labels are always the overlay of every live op's claim in op order (each voxel takes the value of the last live claim covering it, else unlabeled). Undo and redo are new ops (`reverts=op_id`) that flip an op between live and undone and recompute only that op's voxels from the remaining live claims, so the result never depends on the order of undos and never disturbs voxels a later op also wrote. A property test checks that incremental state always equals a full overlay. Workers (propagation, interactive, imports) submit ops through the same writer with a worker token.
- Everything decoded from a client (masks, values) is decompressed with a hard output limit equal to what its box allows. Label chunks are fixed at 64³ uint8 (sparse and highly compressible); the step 14 viewer spike only decides image chunk sizes.
- The data gateway serves a virtual zarr for labels (`/zarr/{pid}/labels/...` → current blob, ETag = sha) and streams image shards with Range passthrough (`Cache-Control: private, immutable`). Optional `IMAGE_READS=presigned` for cloud deploys; labels are always proxied.

### Annotator (web)
- Routes: `/p/[pid]` overview, `/annotate?roi=`, `/rois` (queue, suggestions, gallery with delete #60, label-image import #28/#31), `/models`, `/results`, `/history` (op log, revert), `/settings` (label set, collaborators, default plugin); `/import` (bulk v1 claim); `/admin`.
- WebGL2, one context per plane: CPU extracts 64×64 cross-sections from cached chunks, GPU does uint16 windowing (no 8-bit cast), multichannel additive blend, label palette texture + opacity slider (#51), proposal hatch, uncertainty heatmap. three.js 3D preview.
- Chunk loader ported from bossypaints: visible window + 1 chunk padding, nearest-first, coarse levels first, abort out-of-view requests, decode in Web Workers, LRU by bytes (pin chunks with pending ops).
- State with Svelte 5 runes; zoom/pan never persisted (#37); brush size, windows, opacity persisted per user.
- Tools: brush/eraser (any plane, optional 3D sphere), polygon/lasso (+subtract), threshold brush, flood fill / wand (2D; 3D in a Worker ≤256³), copy to next slice, 2D fill-holes/open/close — client side. Random walker propagation and 3D morphology — CPU worker jobs. Click-to-segment (M3) — `interactive_session` job on a warm GPU worker that opens an outbound WebSocket relayed by the API; previews commit as strict ops.
- Single keymap table drives handlers and the `?` overlay.

### ML plugins
```python
class SegmentationPlugin(Protocol):
    name: ClassVar[str]
    version: ClassVar[str]
    # devices, min_vram_gb, halo, block, probs, prompt
    caps: ClassVar[PluginCaps]

    def params_schema(self) -> dict: ...
    def train(
        self, ds: SparseTrainingSet, params: dict, out: Path, ctx: JobContext
    ) -> TrainResult: ...

    # Predictor.predict_block(C,Z,Y,X) -> (K+1,Z,Y,X) probs; optional prompt()
    def load(self, artifact: Path, device: str) -> Predictor: ...
```
- Plugin label space: background 0, classes 1..K, IGNORE 255 (from unlabeled voxels outside complete ROIs).
- RF: labeled voxels only, balanced per-class sampling (replaces the 1/500 negative stride), 3D features, halo = 4·σmax.
- nnU-Net v2: crops as cases, `ignore` label in `dataset.json`, pre-normalized `noNorm` channel with a global window, time-budgeted trainer, warm start from parent model.
- MONAI SegResNet: CE with `ignore_index` + masked Dice, label-balanced patch sampling, sliding-window inference.
- Metrics: per-class Dice/IoU on validation ROIs (20% of non-suggested ROIs once ≥5 exist), not on training data.
- Active learning: score 64³ cells by top-10% entropy (from the uncertainty grid, or sampled crops inside the specimen mask), NMS + diversity, 20% random exploration.

### Scheduling, compute, and burst
- Local workers via compose; `setup.sh` detects `nvidia-smi` and the NVIDIA container runtime and sets `COMPOSE_PROFILES=gpu`. Local workers share one token (`secrets/worker_token`, registered by `migrate`) and reach only the worker API, through an unpublished Caddy port on the `jobs` network.
- Remote workers: admin creates a worker token in `/admin/workers`, runs `docker run ghcr.io/j6k4m8/ml4paleo-worker-gpu --server ... --token-file ...`.
- Scaler (compose profile `burst`, sole holder of cloud credentials, runs the SkyPilot API server with Postgres state): every 30 s, per pool: demand = queued `scale_trigger` jobs whose requirements match the pool and no non-burst worker can take soon → target nodes = clamp(ceil(backlog/30 min), `min_nodes`, `max_nodes`) → `sky.launch` worker clusters (`down=True`, idle autostop, per-node expiring worker token) → reconcile and tear down idle/orphaned/silent nodes. Background-tier jobs set `scale_trigger=false`.
- Pool config example: `pools: {train: {node_type: L4, max_nodes: 2}, predict: {node_type: L4, max_nodes: 4}, interactive: {node_type: A10G, max_nodes: 1, idle_minutes: 20}}`.

### Storage and credentials
- `StorageGrant(url, access, backend s3|gcs|file|proxy, credentials, endpoint, expires_at)`; `object_store(grant)` / `zarr_store(grant)` with refresh via heartbeat.
- Broker: AWS STS session policy scoped to `projects/<pid>/*`; GCS downscoped tokens; SeaweedFS static keys for local trusted workers, proxy mode for remote and burst.
- Housekeeper GC (staging/failed after 48 h, expired exports, superseded per retention) is the lifecycle mechanism (#1); cloud lifecycle rules only as backstop.

### Security (nothing is public without strong authentication)
`__Host-` session cookie (HttpOnly, Secure, SameSite=Lax), CSRF header + Origin check, every project route through one membership dependency (404 for non-members), route-matrix test over `app.routes`, password length ≥12 + offline common-password list, dummy-hash timing, rate limits (login, signup, reset, v1 claim), bootstrap admin with random password printed once + forced change + TOTP, no API CORS (Neuroglancer self-hosted same-origin), strict CSP, no third-party scripts or analytics, Caddy body cap 8 MB on `/api` (uploads go direct to S3), untrusted files parsed only in non-root workers with Pillow/zip limits, secrets via `*_FILE`, TOTP/OIDC secrets encrypted with a data key.

### Deploy
- Single box: `deploy/compose` with Caddy as the only published service (80/443, HTTPS for `M4P_DOMAIN`, including `localhost` with Caddy's local CA; the API refuses a plain-HTTP public URL), `migrate` one-shot (also creates the first `admin` from `secrets/initial_admin_password`), `seaweedfs-init` one-shot (creates the bucket), `api` (uvicorn, 4 workers, serves SPA), `housekeeper`, `postgres` (17), `backup` (nightly `pg_dump`, 14 days kept), `seaweedfs`, `worker-cpu`, `worker-gpu` (profile), `scaler` (profile). Networks: `public` (Caddy, API, housekeeper; internet access), `private` (internal: Postgres, SeaweedFS, and the server processes), and `jobs` (Caddy's worker-only port `:8080` and the workers). CI checks what each network can reach. `setup.sh DOMAIN` generates every secret. Images non-root, explicit `.dockerignore`; pinning images by digest is still to do.
- Settings: pydantic-settings, prefix `M4P_` (PUBLIC_URL, DATABASE_URL, STORAGE__*, AUTH__*, QUOTA__*, SMTP__*, POOLS__*, V1__VOLUME_PATH).
- OpenTofu `aws`: EC2 + EBS + SG (80/443, SSM), S3 (Block Public Access, SSE, lifecycle), instance role + scoped worker role + SkyPilot policy, optional RDS, Route53. `gcp`: GCE + disk, GCS (uniform access), service accounts (incl. token creator for downscoping), SkyPilot permissions, optional Cloud SQL, Cloud DNS.

## Workflow
- Work happens on the `v2` branch. Each numbered step below is one PR with base `v2`, and CI must be green before it merges.
- When M1 is complete, one PR merges `v2` into `master`; v1 keeps running from `master` until then.

## Milestones

**M1 — vertical slice on one box (PR-sized steps, in order)**
1. uv workspace (root lib + `server` + `worker`), pytest, ruff, pyright, CI that runs tests; drop `requirements.txt`.
2. `ml4paleo.storage` (zarr 3 + obstore, URL backends) + `blocks.py` + zarrvp port (reads v1 zarr v2).
3. `ml4paleo.ome` (czyx boundary, OME-Zarr 0.5 sharded writer, pyramid) + hardened imagevp/dicomvp; golden axis tests.
4. `ml4paleo.labels` value conventions, chunk codec, mask deltas (Python + TS shared fixtures).
5. Server skeleton: settings, SQLAlchemy models, Alembic baseline, SPA serving; compose with Caddy, Postgres, SeaweedFS.
6. Auth: sessions, CSRF, password policy, rate limits, bootstrap admin, signup mode, SMTP outbox (verify/reset), route-matrix test.
7. Projects, collaborators, audit log, quotas (storage, trained models) + request-more form.
8. Queue + worker protocol + worker loop (`noop` kind) + reaper + chaos tests; GPU auto-detect in compose.
9. Artifacts, heads, atomic commit, credential broker (static + proxy), GC.
10. Presigned resumable multipart uploads.
11. Ingest pipeline + SSE progress.
12. Data gateway (Range passthrough, virtual label zarr) + self-hosted Neuroglancer.
13. Label writer (ops, idempotency, rebase/strict, undo/redo, SSE chunk events) + ROI model.
14. SPA scaffold, ChunkStore, zarrita adapter, decode workers, WebGL2 XY plane — **spike on a real 50 GB+ volume** to settle chunk size and proxy throughput.
15. Tri-planar views, crosshair, keymap, label overlay/palette/opacity.
16. Brush/eraser/polygon → OpQueue → writer; undo/redo; ROI tool, queue, complete flag, gallery.
17. Training-set manifests, plugin protocol, RF plugin rewrite, `train` job on the CPU worker.
18. Proposal overlay (`predict.region`, accept/reject), `predict.shard` fan-out with halo + uncertainty, `compose_final` with override + dust removal, per-class meshing with overlap, exports.
19. v1 importer: `/job/<6hex>` claim, bulk claim from localStorage, admin reassign; dry run on a copy of the production volume; then delete `webapp/`.
20. Hardening pass and docs (install, admin, worker setup).

**M2:** CUDA worker image; nnU-Net v2 and MONAI plugins; remote worker tokens UI; validation metrics on models page; active-learning `suggest_rois`; multichannel UI polish (#63).

**M3:** `interactive_session` (nnInteractive, SAM 2.1) with WebSocket relay; random-walker propagation and 3D morphology jobs; SkyPilot scaler with per-job-type pools; OpenTofu `aws` and `gcp` modules + examples.

## Open issues → where they land
#64, #16, #32, #66, #11, #29, #37, #51, #60 → annotator + op log (M1). #65, #1 → storage + GC (M1). #59 → ingest pyramid (M1). #35, #33 → exports and meshing as jobs with progress (M1). #28, #31 → label-image import on `/rois` (M1). #12, #22, #24 → compose_final + mesh options (M1). #14 → optional SMTP (M1). #63 → M2. #17 → superseded by plugins + validation metrics (M2). #57 → polygon tool (M1). #4 → closed by v1 fix + czyx boundary. Close or relabel these on GitHub as each lands.

## Risks to watch
- Viewer bandwidth through the proxy (step 14 spike decides 32³ vs 64³ inner chunks for levels 0–1).
- Label writer correctness under concurrent edits and selective undo (property tests).
- TS/Python coordinate conventions (shared golden fixtures).
- Open signup + compute abuse: storage quota + trained-model quota; burst nodes run only `scale_trigger` jobs; admin can disable users and switch to invite-only.
- v1 claim enumeration (24-bit IDs, previously leakable): accepted; rate limits, claim window, admin reassign.
- SeaweedFS has no prefix-scoped temp credentials → remote/burst workers use proxy mode; recommend a cloud bucket when remote compute is heavy.
- `weed server` serves its master, filer, and volume ports without authentication. Compose keeps Postgres and SeaweedFS on an internal `private` network that only the server processes join (CI checks this), and workers never join it. Step 9 must give local workers storage access without that network: proxy mode through the API, or SeaweedFS JWT signing.
- nnU-Net on small sparse crops (patch size vs crop size), interactive latency to remote GPUs (M2/M3).
- FIFO lets one large pipeline delay others; tiers mitigate, and fair-share can be added as a config switch later.
- Unverified details to pin with integration tests: zarr 3 partial-shard behaviour, obstore + GCS S3 API, SkyPilot secrets and Postgres state config.

## Verification
- Every PR: `uv run pytest` (unit + testcontainers integration for Postgres and SeaweedFS), `ruff`, `pyright`, `alembic upgrade/downgrade/check`, `pnpm -C web check && pnpm -C web test`.
- Queue chaos tests: kill a worker mid-job → requeued; stale-lease completion → 409; duplicate complete → idempotent; cancel a running pipeline.
- Axis golden tests: a non-square, anisotropic synthetic volume round-trips upload → OME-Zarr → export TIFF/PNG → mesh with identical orientation and mm scale (extends `tests/test_data_paths.py`).
- Label writer property tests: concurrent random ops + undo/redo converge to the same state as a serial replay of the op log.
- End to end (Playwright on compose, nightly and on main): sign up → create project → upload stack → ingest → annotate brush + polygon in two planes → mark ROI complete → train RF → accept proposal → final segmentation → mesh → download; then log in as a collaborator and see the same labels with provenance in `/history`.
- v1 importer: dry run against a copy of the production `volume/` directory; check that each claimed project's image, annotations (as complete slice ROIs), and latest model match v1.
- CI keeps the Docker build + Trivy scan green.
