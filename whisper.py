import argparse
import os
import subprocess
import threading
import wave
from datetime import datetime
from queue import Queue, Empty, Full

import mlx_whisper
import numpy as np
from openai import OpenAI
import tkinter as tk
from tkinter import ttk

# Set parameters
SAMPLE_RATE = 16000
CHUNK_DURATION = 3  # 2 seconds
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
BYTES_PER_SAMPLE = 2  # 16-bit = 2 bytes

# Argument parsing
parser = argparse.ArgumentParser(description='Audio transcription with optional recording')
parser.add_argument('-f', '--file', nargs='?', const=True, default=False,
                    help='Save audio to file. Optionally specify filename')
args = parser.parse_args()

# WAV file setup
wav_file = None
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if args.file:
    filename = args.file if isinstance(args.file, str) else f'audio_capture_{timestamp}.wav'
    wav_file = wave.open(filename, 'wb')
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(SAMPLE_RATE)

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Subtitle window class
class SubtitleWindow:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("即時字幕")
        
        # Set window transparency
        self.window.attributes('-alpha', 0.8)
        # Set window always on top
        self.window.attributes('-topmost', True)
        
        # Set window size and position
        window_width = 600
        window_height = 100
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = screen_height - window_height - 50  # 50 pixels from bottom
        self.window.geometry(f'{window_width}x{window_height}+{x}+{y}')
        
        # Add a list to store recent subtitles
        self.recent_subtitles = []
        self.max_lines = 3
        self.current_translation = ""  # Track current streaming translation
        
        # Create label to display subtitles
        self.label = ttk.Label(
            self.window, 
            wraplength=580,
            justify="center",
            font=("Arial", 14)
        )
        self.label.pack(expand=True)
    
    def update_text(self, text, is_final=False):
        if is_final:
            # When translation is complete
            self.recent_subtitles.append(self.current_translation)
            if len(self.recent_subtitles) > self.max_lines:
                self.recent_subtitles.pop(0)
            self.current_translation = ""  # Reset current translation
        else:
            # During streaming, update only the current translation
            self.current_translation = text
            
        # Combine previous subtitles with current translation
        display_lines = self.recent_subtitles[-self.max_lines:] if self.recent_subtitles else []
        if self.current_translation:
            display_lines.append(self.current_translation)
            
        display_text = "\n".join(display_lines)
        self.label.config(text=display_text)
        self.window.update()

def translate_text(text):
    """Translate English text to Chinese, using streaming."""
    # Send the original text to the GUI queue
    gui_queue.put((text, False))
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a translator. Translate the given text to Traditional Chinese briefly and accurately."},
            {"role": "user", "content": text}
        ],
        stream=True
    )
    
    full_response = ""
    print("Translating: ", end="", flush=True)
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_response += content
            print(content, end="", flush=True)
            # Send streaming update with is_final=False
            gui_queue.put((full_response, False))
    print()
    # Send final update with is_final=True
    gui_queue.put((full_response, True))
    return full_response

class AudioProcessor(threading.Thread):
    def __init__(self, queue, gui_queue):
        super().__init__()
        self.queue = queue
        self.gui_queue = gui_queue
        self.running = True

    def run(self):
        while self.running:
            try:
                audio_data = self.queue.get(timeout=1)
                result = mlx_whisper.transcribe(
                    audio_data,
                    path_or_hf_repo="mlx-community/whisper-large-v3-turbo"
                )
                if result and "text" in result:
                    text = result["text"].strip()
                    if text:
                        translate_text(text)
                        # Place the final translation into the GUI queue
                        # self.gui_queue.put(translated_text)
            except Empty:
                continue
            except Exception as e:
                print(f"Error during processing: {e}")

    def stop(self):
        self.running = False

class AudioCapture(threading.Thread):
    def __init__(self, process_queue, wav_file=None):
        super().__init__()
        self.process_queue = process_queue
        self.wav_file = wav_file
        self.running = True
        self.swift_process = subprocess.Popen(['./.build/release/SystemAudioCapture'], 
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   bufsize=0)
        self.audio_buffer = bytearray()

    def run(self):
        try:
            while self.running:
                chunk = self.swift_process.stdout.read(CHUNK_SIZE * BYTES_PER_SAMPLE)
                if not chunk:
                    break
                if self.wav_file:
                    self.wav_file.writeframes(chunk)
                self.audio_buffer.extend(chunk)
                if len(self.audio_buffer) >= CHUNK_SIZE * BYTES_PER_SAMPLE:
                    audio_data = np.frombuffer(
                        self.audio_buffer[:CHUNK_SIZE * BYTES_PER_SAMPLE], 
                        dtype=np.int16
                    ).astype(np.float32) / 32768.0
                    try:
                        self.process_queue.put_nowait(audio_data)
                    except Full:
                        print("Warning: Processing queue is full, skipping chunk")
                    self.audio_buffer = self.audio_buffer[CHUNK_SIZE * BYTES_PER_SAMPLE:]
        except Exception as e:
            print(f"Error during audio capture: {e}")
        finally:
            self.swift_process.terminate()

    def stop(self):
        self.running = False
        self.swift_process.terminate()

if __name__ == "__main__":
    try:
        # Create the subtitle window
        subtitle_window = SubtitleWindow()

        # Create queues
        process_queue = Queue(maxsize=15)
        gui_queue = Queue()

        # Initialize and start threads
        audio_capture = AudioCapture(process_queue, wav_file)
        processor = AudioProcessor(process_queue, gui_queue)
        audio_capture.start()
        processor.start()

        # Function to process GUI queue
        def process_gui_queue():
            try:
                while True:
                    text, is_final = gui_queue.get_nowait()
                    subtitle_window.update_text(text, is_final)
            except Empty:
                pass
            subtitle_window.window.after(100, process_gui_queue)

        # Start processing the GUI queue
        subtitle_window.window.after(100, process_gui_queue)

        # Function to handle window closing
        def on_closing():
            audio_capture.stop()
            processor.stop()
            audio_capture.join()
            processor.join()
            if wav_file:
                wav_file.close()
                print(f"\nAudio saved to: {filename}")
            subtitle_window.window.destroy()

        # Bind the close event
        subtitle_window.window.protocol("WM_DELETE_WINDOW", on_closing)

        # Start the Tkinter main loop
        subtitle_window.window.mainloop()

    except KeyboardInterrupt:
        print("\nStopped listening.")
    finally:
        # Ensure threads are stopped
        if 'audio_capture' in locals():
            audio_capture.stop()
            audio_capture.join()
        if 'processor' in locals():
            processor.stop()
            processor.join()
        if wav_file:
            wav_file.close()
            print(f"\nAudio saved to: {filename}")

