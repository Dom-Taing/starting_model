FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U pip

RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir voxcpm soundfile flask

RUN python - <<'PY'
import pathlib

path = pathlib.Path("/usr/local/lib/python3.11/site-packages/voxcpm/modules/minicpm4/model.py")
txt = path.read_text()

needle = "attn_mask = torch.arange(key_cache.size(2), device=key_cache.device) <= position_id"
if needle not in txt:
    raise SystemExit("Patch target not found; voxcpm code changed.")

replacement = """# 1D mask over key length
        attn_mask_1d = torch.arange(key_cache.size(2), device=key_cache.device) <= position_id

        # Expand to 4D for SDPA broadcast: [B, H, Q, K]
        bsz, num_heads, _, _ = query_states.shape
        attn_mask = attn_mask_1d.view(1, 1, 1, -1).expand(bsz, num_heads, 1, -1)"""

path.write_text(txt.replace(needle, replacement))
print("✅ Patched:", path)
PY

WORKDIR /app

COPY server.py .

EXPOSE 8080

CMD ["python", "server.py"]