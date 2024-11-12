# Audio Transcription System
這是一個即時音訊轉錄系統，可以捕獲系統音訊並轉換成文字。系統使用 Swift 來捕獲音訊，並使用 MLX Whisper 進行語音轉文字。

## 系統需求
- macOS 系統
- Python 3.10+
- Swift 6.0+
- MLX 框架支援

## 安裝步驟
安裝 Python 依賴：
```bash
# 使用 pip
pip install -r requirements.txt

# 或使用 uv (推薦，安裝更快)
uv pip install -r requirements.txt
```
編譯 Swift 音訊捕獲程式：
```
swift build -c release
```
重要提醒：必須先執行 swift build -c release 命令來編譯 Swift 程式，否則 ./.build/release/SystemAudioCapture 將不存在！

## 使用方法
### 基本使用
```
python whisper.py
```
### 儲存音訊檔案
使用自動生成的檔名：
```
python whisper.py -f
```
指定檔名：
```
python whisper.py -f my_recording.wav
```

-f, --file: 儲存音訊到 WAV 檔案
  不帶參數：使用自動生成的檔名（格式：audio_capture_YYYYMMDD_HHMMSS.wav）
  帶參數：使用指定的檔名

## 功能特點
- 即時系統音訊捕獲
- 即時語音轉文字
- 可選的音訊檔案儲存
使用 MLX Whisper 進行高效能語音識別
- 支援中斷操作（Ctrl+C）

## 注意事項
- 首次執行時需要下載 Whisper 模型，這可能需要一些時間
- 確保系統有足夠的權限存取音訊裝置
- 音訊轉換過程中可能會有一些延遲，這是正常現象
