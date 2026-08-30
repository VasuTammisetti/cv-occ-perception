# CPU-only image for the occupancy pipeline. Device is auto-detected at runtime.
# nuScenes data is NOT baked in; it is mounted at runtime as a volume.

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Torch's CPU build from the PyTorch wheel index, added ALONGSIDE PyPI
# (--extra-index-url, not --index-url) so torch comes from the CPU index while
# its build dependencies (flit_core, typing_extensions, ...) still resolve from
# PyPI. An exclusive --index-url would hide PyPI and break that resolution.
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY occperc/ ./occperc/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

RUN mkdir -p /data /app/gt_out

CMD ["python", "-c", "print('Occupancy pipeline image. Run GT: python -m scripts.generate_gt --dataroot /data --out gt_out --max-samples 3 ; Run DL: python -m scripts.train --gt-root gt_out --epochs 10')"]