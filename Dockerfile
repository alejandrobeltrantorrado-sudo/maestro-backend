FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip

RUN pip install torch==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install \
    transformers==4.40.0 \
    scipy==1.13.0 \
    numpy==1.26.4 \
    accelerate==0.30.0

RUN pip install \
    fastapi==0.111.0 \
    "uvicorn[standard]==0.30.1" \
    python-multipart==0.0.9 \
    pydantic==2.7.1

COPY main.py .

RUN mkdir -p /tmp/maestro

ENV MODEL_ID=facebook/musicgen-small
ENV MAX_DURATION=30
ENV TEMP_DIR=/tmp/maestro
ENV ALLOWED_ORIGINS=*

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "300"]
