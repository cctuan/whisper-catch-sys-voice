import argparse
import os
import sys
import subprocess
import threading
import wave
from datetime import datetime
from queue import Queue, Empty, Full
import asyncio

import mlx_whisper
import numpy as np
from openai import OpenAI
import flet as ft
import pyaudio

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
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "即時字幕"
        
        # Update deprecated window properties
        self.page.window_width = 600
        self.page.window_height = 200
        self.page.window_opacity = 0.8
        self.page.window_always_on_top = True
        
        # Store subtitles
        self.original_subtitles = []
        self.translated_subtitles = []
        self.current_original = ""
        self.current_translation = ""
        
        # Create text controls with scroll
        self.original_text = ft.Text("", size=14)
        self.translated_text = ft.Text("", size=14)
        
        # Create scrollable containers using ListView
        self.original_scroll = ft.ListView(
            [self.original_text],
            expand=True,
            auto_scroll=True,
            spacing=10,
            padding=10,
        )
        
        self.translated_scroll = ft.ListView(
            [self.translated_text],
            expand=True,
            auto_scroll=True,
            spacing=10,
            padding=10,
        )
        
        # Create UI elements
        self.show_original = ft.Checkbox(
            label="顯示原文",
            value=True,
            on_change=lambda e: self.refresh_display()
        )
        
        self.auto_scroll = ft.Checkbox(
            label="自動滾動",
            value=True,
            on_change=self.toggle_auto_scroll
        )
        
        # Settings panel
        self.api_key_field = ft.TextField(
            label="OpenAI API Key",
            value=os.environ.get("OPENAI_API_KEY", ""),
            password=True,  # 隱藏 API key
            expand=True
        )
        
        self.api_url_field = ft.TextField(
            label="API URL",
            value=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
            expand=True
        )
        
        self.settings_panel = ft.Column([
            ft.Row([self.api_key_field], expand=True),
            ft.Row([self.api_url_field], expand=True),
            ft.Row([
                ft.ElevatedButton("儲存設定", 
                    on_click=self.save_settings
                )
            ], alignment=ft.MainAxisAlignment.END),
        ], spacing=10)
        
        # 替代 ExpansionTile 的版本
        self.settings_button = ft.ElevatedButton(
            "⚙️ 設定",
            on_click=lambda _: self.toggle_settings()
        )
        
        self.settings_dialog = ft.AlertDialog(
            title=ft.Text("設定"),
            content=self.settings_panel,
        )
        
        # Create checkbox row with settings
        self.control_row = ft.Row([
            self.show_original,
            self.auto_scroll,
            ft.Container(expand=True),  # 推動設定按鈕到右側
            self.settings_button,
        ])
        
        self.original_column = ft.Column([
            ft.Text("原文", size=16, weight=ft.FontWeight.BOLD),
            self.original_scroll
        ], expand=True, alignment=ft.MainAxisAlignment.CENTER)
        
        self.translated_column = ft.Column([
            ft.Text("翻譯", size=16, weight=ft.FontWeight.BOLD),
            self.translated_scroll
        ], expand=True, alignment=ft.MainAxisAlignment.CENTER)
        
        self.columns_row = ft.Row(
            [self.original_column, self.translated_column],
            expand=True,
            alignment=ft.MainAxisAlignment.CENTER,
        )
        
        # Add elements to page with bottom padding
        self.page.add(
            ft.Column([
                self.control_row,
                self.columns_row,
                ft.Container(height=20),
            ], 
            expand=True,
            spacing=10,
            )
        )
    
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
        if self.show_original.value:
            self.original_column.visible = True
            
            # Update original text
            original_display = "\n\n".join(self.original_subtitles)
            if self.current_original:
                if original_display:
                    original_display += "\n\n"
                original_display += self.current_original
            self.original_text.value = original_display
            
        else:
            self.original_column.visible = False
        
        # Update translated text
        translated_display = "\n\n".join(self.translated_subtitles)
        if self.current_translation:
            if translated_display:
                translated_display += "\n\n"
            translated_display += self.current_translation
        self.translated_text.value = translated_display
        
        # Force scroll to bottom if auto_scroll is enabled
        if self.auto_scroll.value:
            self.original_scroll.scroll_to(offset=float('inf'), duration=0)  # 立即滾動到底部
            self.translated_scroll.scroll_to(offset=float('inf'), duration=0)  # 立即滾動到底部
        
        self.page.update()
    
    def toggle_auto_scroll(self, e):
        """Toggle auto scroll for both scrollable areas"""
        self.original_scroll.auto_scroll = self.auto_scroll.value
        self.translated_scroll.auto_scroll = self.auto_scroll.value
        self.page.update()
    
    def save_settings(self, e):
        """Save settings and update OpenAI client"""
        api_key = self.api_key_field.value
        api_base = self.api_url_field.value
        
        # Update environment variables
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = api_base
        
        # Update OpenAI client
        global client
        client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        
        # Show success message
        self.page.show_snack_bar(
            ft.SnackBar(content=ft.Text("設定已儲存"))
        )
        self.page.update()
    
    def toggle_settings(self):
        self.settings_dialog.open = True
        self.page.update()

def translate_text(text, gui_queue):
    """Translate English text to Chinese, using streaming."""
    gui_queue.put((text, False, True))
    gui_queue.put((text, True, True))
    
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
                        translate_text(text, self.gui_queue)
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

async def main(page: ft.Page):
    try:
        # Pre-initialize whisper model
        initialize_whisper()
        
        # Create the subtitle window
        subtitle_window = SubtitleWindow(page)
        
        # Create queues
        process_queue = Queue(maxsize=15)
        gui_queue = Queue()

        # Initialize and start threads
        audio_capture = AudioCapture(process_queue, wav_file=wav_file)
        processor = AudioProcessor(process_queue, gui_queue)
        audio_capture.start()
        processor.start()

        async def process_gui_queue():
            while True:
                try:
                    text, is_final, is_original = gui_queue.get_nowait()
                    subtitle_window.update_text(text, is_final, is_original)
                except Empty:
                    pass
                await asyncio.sleep(0.1)

        # Start processing the GUI queue
        task = asyncio.create_task(process_gui_queue())

        async def cleanup():
            task.cancel()
            audio_capture.stop()
            processor.stop()
            audio_capture.join()
            processor.join()
            if wav_file:
                wav_file.close()
                print(f"\nAudio saved to: {filename}")

        page.on_close = cleanup
        await page.update_async()

    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    ft.app(target=main)

