ARG PYTHON_IMAGE=python:3.11-slim
FROM ${PYTHON_IMAGE}

ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_DEFAULT_INDEX=${PIP_INDEX_URL} \
    PATH="/app/.venv/bin:$PATH" \
    CHAINLIT_APP_ROOT=/data \
    MPLCONFIGDIR=/tmp/matplotlib \
    PRIZM_USAGE_LEDGER_PATH=/data/usage.sqlite3

# CJK 字体供 matplotlib 图表使用；避免中文显示方框。
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --index-url "$PIP_INDEX_URL" uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev \
    && useradd --create-home --uid 10001 prizm \
    && mkdir -p /data /tmp/matplotlib \
    && chown -R prizm:prizm /app /data /tmp/matplotlib

USER prizm

EXPOSE 8000
VOLUME ["/data"]

CMD ["chainlit", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
