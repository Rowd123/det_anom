FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY pmu_simulator ./pmu_simulator
COPY examples/rpad_detection.json ./examples/rpad_detection.json
RUN pip install --no-cache-dir .
ENV PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
ENTRYPOINT ["rpad-detection", "phase-stream"]
CMD ["--config", "/app/examples/rpad_detection.json"]
