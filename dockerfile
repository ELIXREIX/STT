FROM python:3.8-slim-buster

# ติดตั้งแพ็กเกจพื้นฐาน
RUN apt-get update && apt-get install -y \
    ffmpeg git gcc curl unzip wget make build-essential cmake \
    python3-dev swig portaudio19-dev libsndfile1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ติดตั้ง Python packages
RUN pip install --upgrade pip
RUN pip install faster-whisper==0.9.0 scipy sounddevice numpy jiwer

# ติดตั้ง Vosk เวอร์ชัน ARM64 จาก wheel
WORKDIR /tmp
RUN wget https://github.com/alphacep/vosk-api/releases/download/v0.3.30/vosk-0.3.30-py3-none-linux_aarch64.whl && \
    pip install vosk-0.3.30-py3-none-linux_aarch64.whl

# กลับเข้ามา app
WORKDIR /app
COPY . .

# โหลด Vosk model ขนาดเล็ก
RUN mkdir -p /models/vosk && \
    curl -L -o vosk-model-small-en-us-0.15.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip && \
    unzip vosk-model-small-en-us-0.15.zip -d /models/vosk && \
    mv /models/vosk/vosk-model-small-en-us-0.15 /models/vosk/model && \
    rm vosk-model-small-en-us-0.15.zip

# โหลด Whisper model ขนาด tiny.en ล่วงหน้า
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('tiny.en', download_root='/models/whisper')"

CMD ["python3", "benchmark.py"]