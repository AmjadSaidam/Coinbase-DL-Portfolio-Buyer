# --- INFO ---
# build instruction file used to create images and containers 
# note: If you import/download any new module you must rebuild the docker image and container, this file remains the same

# Terminal/Command-Line:
# 1) to build image (do for each .py file)
    # >>> docker build -f DockerFile -t {image_name} .
# 2) to build container:
    # >>> docker run -d --restart always -p 8000:8000 --name {container_name} {image_name}
# for ports, the boilerplate is -p {host port}:{container port from EXPOSE}

# Docker Desktop:
# When building the trading-strategy container include a volume (v) mount before --name {image_name}:
    # >>> -v "$(pwd)":/app
# this is so when building our container, we are able to read and write to our local database files, otherwise we get a sqlight3.OperationalError
# The Volume mount allows dynamic updates, overiding the COPY . . snapshot
# also mount an anonymous volume over /app/.venv (so the bind mount doesn't hide the dependencies installed during the image build):
    # >>> (-v /app/.venv) 

# ---- Base Image ----
# pinned to match .python-version / pyproject.toml so dependency resolution
# is identical to what the lock file (uv.lock) was generated against
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base 

# ---- Environment ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# ---- Install uv (pinned for reproducible, deterministic dependency installs) ----
COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /uvx /usr/local/bin/

# ---- Create Non-Privileged User ----
# UID/GID default to the host dev user so files written inside the container (sqlite dbs, live
# state/logs) under the .:/app bind mount stay owned by a user the host account can also write to -
# otherwise appuser's own uid/gid never match the pre-existing host-owned files and every write hits
# sqlite3.OperationalError: attempt to write a readonly database. Override at build time for other machines.
ARG UID=10001
ARG GID=10001
# GID may already belong to a base-image group (e.g. debian's dialout=20) - reuse it by name instead
# of creating a duplicate group, since useradd only needs an existing gid, not a matching group name
RUN group_name="$(getent group "${GID}" | cut -d: -f1)"; \
    if [ -z "$group_name" ]; then \
        addgroup --gid "${GID}" appuser; \
        group_name=appuser; \
    fi; \
    adduser --disabled-password --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    --ingroup "$group_name" \
    appuser

# ---- Create and Set Permissions for /app/db ----
RUN mkdir -p /app/db && \
    chown -R appuser:"${GID}" /app/db && \
    chmod -R 770 /app/db

# ---- Install Python Dependencies ----
# uses uv.lock so every machine resolves the exact same dependency versions
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# ---- Copy Application Code ----
# snapshot copy of our repo (overwritten in .yaml)
COPY . .

# ---- Switch to Non-Privileged User ----
USER appuser

# ---- Expose (Container) Port ----
EXPOSE 8000

# ---- Default Command ----
# run first then comment in, and uncomment out 'CMD ["python", "-m", "live_trading.run_live"]' for the trading-strategy container
CMD ["python", "telegram_bot.py"]
#CMD ["python", "-m", "live_trading.run_live"]
