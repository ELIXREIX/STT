import sounddevice as sd
from scipy.io.wavfile import write

duration = int(input("⏱️ ระบุความยาวเสียง (วินาที): "))
fs = 16000
print("🎤 กำลังอัดเสียง...")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
write("test.wav", fs, recording)
print("✅ บันทึก .wav สำเร็จ")
