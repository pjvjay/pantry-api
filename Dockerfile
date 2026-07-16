FROM python:3.12-slim

WORKDIR /app

# gcc + libc headers: psutil (via burr) has no arm64 wheel for this
# base and builds from source. Purged again after pip install — the
# whole dance lives in one layer so the compilers never ship.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY pantry_planner/ ./pantry_planner/
COPY seeds/ ./seeds/

RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && pip install --no-cache-dir -e . \
    && apt-get purge -y gcc libc6-dev && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Seed the DB at build time so the container ships ready-to-serve
RUN python -m pantry_planner.db seed || true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "pantry_planner.api:app", "--host", "0.0.0.0", "--port", "8000"]
