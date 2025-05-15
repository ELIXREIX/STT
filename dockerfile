FROM python:3.10-slim

# ติดตั้งของจำเป็น
RUN apt-get update && apt-get install -y ffmpeg git gcc portaudio19-dev libsndfile1

# ติดตั้งไลบรารี
RUN pip install --upgrade pip
RUN pip install faster-whisper vosk scipy sounddevice numpy jiwer psutil

# เตรียม working dir
WORKDIR /app
COPY . .

# โหลด Vosk Model (ขนาดเล็ก)
RUN apt-get update && apt-get install -y curl unzip && \
    mkdir -p /models/vosk && \
    curl -L -o vosk-model-small-en-us-0.15.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip && \
    unzip vosk-model-small-en-us-0.15.zip -d /models/vosk && \
    mv /models/vosk/vosk-model-small-en-us-0.15 /models/vosk/model && \
    rm vosk-model-small-en-us-0.15.zip

# โหลด Whisper model ล่วงหน้า
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('tiny.en', download_root='/models/whisper')"

CMD ["python3", "benchmark.py"]
