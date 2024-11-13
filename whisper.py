import argparse
import os
import sys
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
import pyaudio  # Add this import at the top

# Set parameters
SAMPLE_RATE = 16000
CHUNK_DURATION = 3  # 2 seconds
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
BYTES_PER_SAMPLE = 2  # 16-bit = 2 bytes

# Argument parsing
parser = argparse.ArgumentParser(description='Audio transcription with optional recording')
parser.add_argument('-f', '--file', nargs='?', const=True, default=False,
                    help='Save audio to file. Optionally specify filename')
parser.add_argument('-d', '--debug', action='store_true',
                    help='Enable debug output including processing times')
parser.add_argument('-m', '--mic', action='store_true',
                    help='Use microphone input instead of system audio')
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
        
        # Set window transparency and always on top
        self.window.attributes('-alpha', 0.8, '-topmost', True)
        
        # Set window size and position
        window_width = 600
        window_height = 150  # Reduced height
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = screen_height - window_height - 50
        self.window.geometry(f'{window_width}x{window_height}+{x}+{y}')
        
        # Add checkbox for display mode
        self.show_original = tk.BooleanVar()
        self.checkbox = ttk.Checkbutton(
            self.window,
            text="顯示原文",
            variable=self.show_original,
            command=self.refresh_display
        )
        self.checkbox.pack(anchor='w', padx=10, pady=5)
        
        # Create frame for text widgets
        self.frame = ttk.Frame(self.window)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Create frames for each column
        self.original_frame = ttk.Frame(self.frame)
        self.translated_frame = ttk.Frame(self.frame)
        
        # Create labels for columns
        self.original_label = ttk.Label(
            self.original_frame,
            text="原文",
            font=("Arial", 12, "bold"),
            anchor="center"
        )
        self.original_label.pack(pady=(0, 5))
        
        self.translated_label = ttk.Label(
            self.translated_frame,
            text="翻譯",
            font=("Arial", 12, "bold"),
            anchor="center"
        )
        self.translated_label.pack(pady=(0, 5))
        
        # Create text widgets in their respective frames
        self.original_text = tk.Text(
            self.original_frame,
            wrap=tk.WORD,
            font=("Arial", 14),
            background=self.window.cget('bg'),
            relief=tk.FLAT,
            padx=10,
            width=30
        )
        self.original_text.pack(fill=tk.BOTH, expand=True)
        
        self.translated_text = tk.Text(
            self.translated_frame,
            wrap=tk.WORD,
            font=("Arial", 14),
            background=self.window.cget('bg'),
            relief=tk.FLAT,
            padx=10,
            width=30
        )
        self.translated_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure text alignment
        self.original_text.tag_configure('center', justify='center')
        self.translated_text.tag_configure('center', justify='center')
        
        # Store subtitles
        self.original_subtitles = []
        self.translated_subtitles = []
        self.current_original = ""
        self.current_translation = ""
    
    def update_text(self, text, is_final=False, is_original=False):
        if is_original:
            if is_final:
                self.original_subtitles.append(self.current_original)
                self.current_original = ""
            else:
                self.current_original = text
        else:
            if is_final:
                self.translated_subtitles.append(self.current_translation)
                self.current_translation = ""
            else:
                self.current_translation = text
        
        self.refresh_display()
    
    def refresh_display(self):
        # Clear both text widgets
        self.original_text.delete('1.0', tk.END)
        self.translated_text.delete('1.0', tk.END)
        
        if self.show_original.get():
            # Show both original and translated text side by side
            self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.translated_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Update original text
            original_display = "\n\n".join(self.original_subtitles)
            if self.current_original:
                if original_display:
                    original_display += "\n\n"
                original_display += self.current_original
            self.original_text.insert('1.0', original_display, 'center')
            
            # Update translated text
            translated_display = "\n\n".join(self.translated_subtitles)
            if self.current_translation:
                if translated_display:
                    translated_display += "\n\n"
                translated_display += self.current_translation
            self.translated_text.insert('1.0', translated_display, 'center')
        else:
            # Hide original text and show only translation
            self.original_frame.pack_forget()
            self.translated_frame.pack(fill=tk.BOTH, expand=True)
            
            # Update only translated text
            translated_display = "\n\n".join(self.translated_subtitles)
            if self.current_translation:
                if translated_display:
                    translated_display += "\n\n"
                translated_display += self.current_translation
            self.translated_text.insert('1.0', translated_display, 'center')
        
        # Auto-scroll to bottom
        if self.show_original.get():
            self.original_text.see(tk.END)
        self.translated_text.see(tk.END)
        self.window.update()

def translate_text(text):
    """Translate English text to Chinese, using streaming."""
    # Send the original text to the GUI queue
    gui_queue.put((text, False, True))  # Added is_original=True
    gui_queue.put((text, True, True))   # Final update for original text
    
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
            gui_queue.put((full_response, False, False))  # Added is_original=False
    print()
    # Send final update with is_final=True
    gui_queue.put((full_response, True, False))  # Added is_original=False
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
                
                if args.debug:
                    start_time = datetime.now()
                    
                result = mlx_whisper.transcribe(
                    audio_data,
                    path_or_hf_repo="mlx-community/whisper-large-v3-turbo"
                )
                
                if args.debug:
                    process_time = (datetime.now() - start_time).total_seconds()
                    print(f"[DEBUG] Whisper transcription took {process_time:.2f} seconds")
                
                if result and "text" in result:
                    text = result["text"].strip()
                    if text:
                        if args.debug:
                            print(f"[DEBUG] Transcribed text: {text}")
                        translate_text(text)
                        # Place the final translation into the GUI queue
                        # self.gui_queue.put(translated_text)
            except Empty:
                continue
            except Exception as e:
                print(f"Error during processing: {e}")

    def stop(self):
        self.running = False

def get_executable_path(file_name):
    # 若程式是以打包形式執行，使用 sys._MEIPASS 路徑
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, file_name)
    else:
        # 若非打包執行，直接使用原始路徑
        return os.path.join(os.path.dirname(__file__), '.build/release', file_name)


class AudioCapture(threading.Thread):
    def __init__(self, process_queue, wav_file=None):
        super().__init__()
        self.process_queue = process_queue
        self.wav_file = wav_file
        self.running = True
        self.audio_buffer = bytearray()
        if args.mic:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=1024,  # 使用較小的緩衝區
                input_device_index=None,  # 使用默認輸入設備
                stream_callback=None,
                start=True,  # 立即開始錄音
            )
        else:
            system_audio_path = get_executable_path('SystemAudioCapture')
            self.swift_process = subprocess.Popen([system_audio_path], 
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       bufsize=0)

    def run(self):
        try:
            while self.running:
                if args.mic:
                    # Microphone input handling
                    chunk = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                else:
                    # System audio input handling
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
            if args.mic:
                self.stream.stop_stream()
                self.stream.close()
                self.p.terminate()
            else:
                self.swift_process.terminate()

    def stop(self):
        self.running = False
        if args.mic:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
        else:
            self.swift_process.terminate()

# Add this after the imports
def initialize_whisper():
    """Pre-initialize whisper model"""
    print("Initializing Whisper model...")
    # Load model with a dummy inference
    dummy_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    mlx_whisper.transcribe(
        dummy_audio,
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo"
    )
    print("Whisper model initialized!")

if __name__ == "__main__":
    try:
        # Pre-initialize whisper model
        initialize_whisper()
        
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
                    text, is_final, is_original = gui_queue.get_nowait()
                    subtitle_window.update_text(text, is_final, is_original)
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

