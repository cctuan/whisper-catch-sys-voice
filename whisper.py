import argparse
import os
import subprocess

# from mlx_llm.chat import ChatSetup, LLMChat
import threading
import wave
from datetime import datetime
from queue import Empty
from queue import Queue

import mlx_whisper
import numpy as np
from openai import OpenAI

# 設置參數
SAMPLE_RATE = 16000
CHUNK_DURATION = 2  # 2秒
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
BYTES_PER_SAMPLE = 2  # 16-bit = 2 bytes

# 新增參數解析
parser = argparse.ArgumentParser(description='Audio transcription with optional recording')
parser.add_argument('-f', '--file', nargs='?', const=True, default=False,
                    help='Save audio to file. Optionally specify filename')
args = parser.parse_args()

# 修改 WAV 檔案創建邏輯
wav_file = None
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if args.file:
    filename = args.file if isinstance(args.file, str) else f'audio_capture_{timestamp}.wav'
    wav_file = wave.open(filename, 'wb')
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(SAMPLE_RATE)

# 啟動 Swift 程式
swift_process = subprocess.Popen(['./.build/release/SystemAudioCapture'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               bufsize=0)

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def translate_text(text):
    """將英文文本翻譯成中文"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a translator. Translate the given text to Traditional Chinese briefly and accurately."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

# 新增一個處理音訊的線程類
class AudioProcessor(threading.Thread):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.running = True

    def run(self):
        while self.running:
            try:
                # 從隊列中獲取音訊數據，等待1秒
                audio_data = self.queue.get(timeout=1)

                # 使用 MLX Whisper 處理音訊
                result = mlx_whisper.transcribe(
                    audio_data,
                    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                )

                if result and "text" in result:
                    text = result["text"].strip()
                    if text:
                        text = translate_text(text)
                        print(f"Transcription: {text}")

            except Empty:
                continue  # 如果隊列為空，繼續等待
            except Exception as e:
                print(f"Error during processing: {e}")

    def stop(self):
        self.running = False

# 主程序修改
try:
    audio_buffer = bytearray()
    # 創建一個隊列來存放待處理的音訊數據
    process_queue = Queue(maxsize=10)  # 限制隊列大小為10

    # 啟動處理線程
    processor = AudioProcessor(process_queue)
    processor.start()

    while True:
        # 讀取音訊數據
        chunk = swift_process.stdout.read(CHUNK_SIZE * BYTES_PER_SAMPLE)
        if not chunk:
            break

        # 條件性寫入 WAV 文件
        if wav_file:
            wav_file.writeframes(chunk)

        # 將數據添加到緩衝區
        audio_buffer.extend(chunk)

        # 當緩衝區達到目標大小時進行處理
        if len(audio_buffer) >= CHUNK_SIZE * BYTES_PER_SAMPLE:
            # 轉換為 numpy array
            audio_data = np.frombuffer(audio_buffer[:CHUNK_SIZE * BYTES_PER_SAMPLE],
                                     dtype=np.int16).astype(np.float32) / 32768.0

            try:
                # 非阻塞方式將數據放入處理隊列
                process_queue.put_nowait(audio_data)
            except Queue.full:
                print("Warning: Processing queue is full, skipping chunk")

            # 保留剩餘的數據
            audio_buffer = audio_buffer[CHUNK_SIZE * BYTES_PER_SAMPLE:]

except KeyboardInterrupt:
    print("\nStopped listening.")
finally:
    # 停止處理線程
    if 'processor' in locals():
        processor.stop()
        processor.join()
    if wav_file:
        wav_file.close()
        print(f"\nAudio saved to: {filename}")
    swift_process.terminate()

