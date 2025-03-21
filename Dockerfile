FROM python:3.12-slim

ENV API_URL_PREFIX http://host.docker.internal:1234
ENV MODEL_NAME orpheus-3b-0.1-ft-q4_k_m

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y libportaudio2
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
CMD python gguf_orpheus.py \
    --host 0.0.0.0 \
    --port 5000 \
    --api-url-prefix $API_URL_PREFIX \
    --model $MODEL_NAME \
    --debug
