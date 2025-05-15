import os
import time
import wave
import json
import subprocess
import psutil

from vosk import Model, KaldiRecognizer
from faster_whisper import WhisperModel
from jiwer import wer, Compose, RemovePunctuation, ToLowerCase, RemoveWhiteSpace

# === เตรียมวัดทรัพยากร ===
process = psutil.Process(os.getpid())

def measure_resource(fn):
    start_time = time.time()
    cpu_start = process.cpu_percent(interval=None)
    mem_start = process.memory_info().rss / 1024 / 1024  # MB

    result = fn()

    cpu_end = process.cpu_percent(interval=None)
    mem_end = process.memory_info().rss / 1024 / 1024  # MB
    end_time = time.time()

    latency = end_time - start_time
    cpu_used = cpu_end
    mem_used = mem_end - mem_start
    return result, latency, cpu_used, mem_used

# === เลือกไฟล์เสียง ===
wav_files = [f for f in os.listdir(".") if f.endswith(".wav")]
if not wav_files:
    print("❌ ไม่พบไฟล์ .wav")
    exit()

print("📂 เลือกไฟล์เสียง:")
for i, f in enumerate(wav_files):
    print(f"[{i}] {f}")

idx = int(input("กรุณาเลือกหมายเลขไฟล์: "))
orig_wav = wav_files[idx]
txt_file = orig_wav.replace(".wav", ".txt")
converted_wav = "converted_temp.wav"

if not os.path.exists(txt_file):
    print(f"❌ ไม่พบไฟล์ {txt_file} สำหรับ ground truth")
    exit()

# === แปลงไฟล์ .wav เป็น PCM ===
print(f"🔄 แปลงไฟล์ {orig_wav} → PCM format...")
subprocess.run([
    "ffmpeg", "-y", "-i", orig_wav,
    "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
    converted_wav
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# === โหลด ground truth ===
with open(txt_file, "r", encoding="utf-8") as f:
    ground_truth = f.read().strip().lower()

# === ฟังก์ชัน normalize สำหรับ WER ===
transform = Compose([
    RemovePunctuation(),
    ToLowerCase(),
    RemoveWhiteSpace(replace_by_space=True),
])

# === รัน Vosk ===
def run_vosk():
    model = Model("/models/vosk/model")
    wf = wave.open(converted_wav, "rb")
    audio_data = wf.readframes(wf.getnframes())
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.AcceptWaveform(audio_data)
    result = rec.Result()
    return json.loads(result)["text"]

print("\n🧠 Running Vosk...")
vosk_text, vosk_time, vosk_cpu, vosk_mem = measure_resource(run_vosk)

# === รัน Whisper ===
def run_whisper():
    model = WhisperModel("tiny.en", download_root="/models/whisper")
    segments, _ = model.transcribe(converted_wav)
    return " ".join([seg.text for seg in segments])

print("\n🧠 Running Whisper...")
whisper_text, whisper_time, whisper_cpu, whisper_mem = measure_resource(run_whisper)

# === แสดงผล ===
print("\n🧾 Ground Truth:", ground_truth)

print("\n🎙️ Whisper Output:", whisper_text.strip().lower())
print("🕐 Latency: %.2f sec | ⚙️ CPU: %.2f%% | 📦 RAM: %.2f MB" % (whisper_time, whisper_cpu, whisper_mem))
print("📉 WER: %.2f%%" % (wer(transform(ground_truth), transform(whisper_text)) * 100))

print("\n🎙️ Vosk Output:", vosk_text.strip().lower())
print("🕐 Latency: %.2f sec | ⚙️ CPU: %.2f%% | 📦 RAM: %.2f MB" % (vosk_time, vosk_cpu, vosk_mem))
print("📉 WER: %.2f%%" % (wer(transform(ground_truth), transform(vosk_text)) * 100))

# === ลบไฟล์แปลงชั่วคราว
os.remove(converted_wav)
