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

# The release this image is. build.yml passes them (RELEASING.md in
# pantry-platform); pantry_planner/version.py reads the env, and the
# labels let anyone holding only the image find its commit. Last, so a
# new version reuses every layer above. A plain `docker build` leaves
# them empty and the API reports "unknown".
ARG APP_VERSION=""
ARG GIT_SHA=""
ARG BUILD_TIME=""
ENV PANTRY_API_VERSION=$APP_VERSION \
    PANTRY_API_REVISION=$GIT_SHA \
    PANTRY_API_BUILD_TIME=$BUILD_TIME
LABEL org.opencontainers.image.title="pantry-api" \
      org.opencontainers.image.source="https://github.com/pjvjay/pantry-api" \
      org.opencontainers.image.version=$APP_VERSION \
      org.opencontainers.image.revision=$GIT_SHA \
      org.opencontainers.image.created=$BUILD_TIME

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "pantry_planner.api:app", "--host", "0.0.0.0", "--port", "8000"]
