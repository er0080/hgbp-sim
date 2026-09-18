# ---- build the React frontend
FROM node:20-alpine AS web
WORKDIR /web
COPY webui/frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY webui/frontend ./
RUN npm run build

# ---- python runtime with the simulator and the API
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml README.md LICENSE ./
COPY hgbp_sim ./hgbp_sim
COPY webui/backend/requirements.txt ./webui/backend/requirements.txt
RUN pip install . CoolProp -r webui/backend/requirements.txt
COPY webui/backend ./webui/backend
COPY --from=web /web/dist ./webui/frontend/dist
EXPOSE 8000
CMD ["uvicorn", "webui.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
