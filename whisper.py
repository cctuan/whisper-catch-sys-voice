import subprocess
import numpy as np
import mlx_whisper
from mlx_llm.model import create_model
import sys
import wave
from datetime import datetime
import argparse

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

model = create_model("phi_3.5_mini_instruct")

try:
    audio_buffer = bytearray()
    
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
            audio_data = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0

            try:
                # 使用 MLX Whisper 處理音訊
                result = mlx_whisper.transcribe(
                    audio_data,
                    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                )
                
                if result and "text" in result:
                    text = result["text"].strip()
                    if text:
                        print(f"Transcription: {text}")
                
            except Exception as e:
                print(f"Error during transcription: {e}")
            
            # 清空緩衝區，為下一塊做準備
            audio_buffer = bytearray()

except KeyboardInterrupt:
    print("\nStopped listening.")
finally:
    if wav_file:
        wav_file.close()
        print(f"\nAudio saved to: {filename}")
    swift_process.terminate()
