# Hosted demo

ModelMesh uses the Render Blueprint in `render.yaml` for the public demo shape:

- one public gateway
- two private workers
- one managed cache

The hosted gateway exposes `/health`, `/predict`, `/metrics`, `/workers`, and `/dashboard`.
Worker delay and failure controls are disabled in the Blueprint with
`GATEWAY_CONTROLS_ENABLED=false`.

## Deploy from Render

1. Open Render and choose **New** > **Blueprint**.
2. Connect this repository.
3. Select `render.yaml`.
4. Review the service list before applying:
   - `modelmesh-gateway` uses the `starter` plan.
   - `modelmesh-worker-a` and `modelmesh-worker-b` use private services.
   - `modelmesh-cache` uses Render Key Value.
5. Apply the Blueprint.
6. Wait until the gateway and both workers finish their first deploy.

Private worker services may cost money on Render. Review the pricing screen before applying
the Blueprint.

## Validate the live gateway

After Render gives the gateway a public URL, run:

```bash
python scripts/smoke_check.py --base-url https://your-render-gateway-url
```

The smoke check verifies:

- `/health`
- `/predict`
- `/metrics`
- `/workers`
- `/dashboard`

Only add the live URL to the README or GitHub repo homepage after this smoke check passes.
