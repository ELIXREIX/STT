import os
import time
import wave
import json
import subprocess
from vosk import Model, KaldiRecognizer
from faster_whisper import WhisperModel
from jiwer import wer, Compose, RemovePunctuation, ToLowerCase, RemoveWhiteSpace


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

# === แปลงไฟล์ .wav เป็น PCM ด้วย ffmpeg ===
print(f"🔄 แปลงไฟล์ {orig_wav} → PCM format...")
subprocess.run([
    "ffmpeg", "-y", "-i", orig_wav,
    "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
    converted_wav
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# === โหลด ground truth ===
with open(txt_file, "r", encoding="utf-8") as f:
    ground_truth = f.read().strip().lower()

# === โหลด Vosk Model ===
vosk_model = Model("/models/vosk/model")
wf = wave.open(converted_wav, "rb")
audio_data = wf.readframes(wf.getnframes())
rec = KaldiRecognizer(vosk_model, wf.getframerate())

start_vosk = time.time()
rec.AcceptWaveform(audio_data)
vosk_result = rec.Result()
end_vosk = time.time()

vosk_text = json.loads(vosk_result)["text"]
vosk_time = end_vosk - start_vosk

# === Whisper Model ===
whisper_model = WhisperModel("tiny.en", download_root="/models/whisper")
start_whisper = time.time()
segments, _ = whisper_model.transcribe(converted_wav)
whisper_text = " ".join([seg.text for seg in segments])
end_whisper = time.time()
whisper_time = end_whisper - start_whisper

transform = Compose([
    RemovePunctuation(),
    ToLowerCase(),
    RemoveWhiteSpace(replace_by_space=True),
])

# === แสดงผล ===
print("\n🧾 Ground Truth:", ground_truth)
print("\n🎙️ Whisper Output:", whisper_text.strip().lower())
print("🕐 Whisper Latency: %.2f sec" % whisper_time)
print("📉 Whisper WER: %.2f%%" % (wer(transform(ground_truth), transform(whisper_text)) * 100))

print("\n🎙️ Vosk Output:", vosk_text.strip().lower())
print("🕐 Vosk Latency: %.2f sec" % vosk_time)
print("📉 Vosk WER: %.2f%%" % (wer(transform(ground_truth), transform(vosk_text)) * 100))

# === ลบไฟล์แปลงชั่วคราว
os.remove(converted_wav)
