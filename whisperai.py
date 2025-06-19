

import whisper
import sounddevice as sd
import numpy as np
import queue
import threading
import time

def record_audio(q, stop_event, samplerate=16000):
    """Record audio from microphone and put chunks in queue until stop_event is set."""
    def callback(indata, frames, time_, status):
        if status:
            print(f"Sounddevice status: {status}")
        q.put(indata.copy())

    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
        while not stop_event.is_set():
            sd.sleep(100)

def calculate_speech_speed(transcription, duration_seconds):
    """Calculate words per minute (WPM)."""
    words = transcription.strip().split()
    word_count = len(words)
    if duration_seconds > 0:
        wpm = (word_count / duration_seconds) * 60
    else:
        wpm = 0
    return wpm, word_count

def calculate_accuracy(reference_text, transcribed_text):
    """Calculate clarity score and count correct words."""
    reference_words = reference_text.strip().split()
    transcribed_words = transcribed_text.strip().split()

    if not reference_words:
        return 0.0, 0

    correct = 0
    for i in range(min(len(reference_words), len(transcribed_words))):
        if reference_words[i].lower() == transcribed_words[i].lower():
            correct += 1

    errors = len(reference_words) - correct + abs(len(reference_words) - len(transcribed_words))
    wer = errors / len(reference_words)
    clarity_score = max(0.0, 1.0 - wer)
    return clarity_score, correct

def detect_dyslexia(wpm, clarity_score):
    """Simple heuristic-based dyslexia symptom detection."""
    if wpm < 80 and clarity_score < 0.6:
        return "⚠️ Possible signs of dyslexia: slow and inaccurate reading."
    elif wpm < 80:
        return "⚠️ Reading is slow; might indicate dyslexia symptoms."
    elif clarity_score < 0.6:
        return "⚠️ Pronunciation mismatch; potential dyslexia symptoms."
    else:
        return "✅ Speech speed and clarity seem normal."

def main():
    print("Loading Whisper model...")
    model = whisper.load_model("base.en")

    q = queue.Queue()
    stop_event = threading.Event()

    print("\nPress Enter to start recording.")
    input()
    print("Recording... Press Enter again to stop.")
    
    record_thread = threading.Thread(target=record_audio, args=(q, stop_event))
    record_thread.start()
    input()
    stop_event.set()
    record_thread.join()
    print("Recording stopped.")

    # Combine audio
    audio_chunks = []
    while not q.empty():
        audio_chunks.append(q.get())

    if not audio_chunks:
        print("No audio recorded.")
        return

    audio_np = np.concatenate(audio_chunks, axis=0).flatten()
    if audio_np.dtype != np.float32:
        audio_np = audio_np.astype(np.float32)

    duration_seconds = len(audio_np) / 16000
    print(f"\nAudio duration: {duration_seconds:.2f} seconds")

    result = model.transcribe(audio_np, language='en')
    transcription = result['text'].strip()
    print(f"\n📝 Transcription:\n{transcription}")

    # Calculate speech speed
    wpm, word_count = calculate_speech_speed(transcription, duration_seconds)
    print(f"📊 Words Per Minute: {wpm:.2f} WPM | Total Words Spoken: {word_count}")

    # Reference text
    ref_text = input("\n📖 Enter the reference sentence (what the user was supposed to say):\n").strip()
    if ref_text:
        clarity_score, correct_words = calculate_accuracy(ref_text, transcription)
        print(f"✅ Accuracy: {clarity_score*100:.2f}% | Correct Words: {correct_words} out of {len(ref_text.strip().split())}")

        # Detect possible dyslexia symptoms
        diagnosis = detect_dyslexia(wpm, clarity_score)
        print(f"\n🩺 Dyslexia Detection Result:\n{diagnosis}")
    else:
        print("⚠️ No reference text provided. Skipping accuracy and diagnosis.")

if __name__ == "__main__":
    main()
