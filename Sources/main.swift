import Foundation
import ScreenCaptureKit
import AVFoundation

@available(macOS 14.0, *)
func requestScreenCapturePermission() async -> Bool {
    guard let display = try? await SCShareableContent.current.displays.first else {
        return false
    }
    let filter = SCContentFilter(display: display, excludingApplications: [], exceptingWindows: [])
    return true
}

// 添加音訊輸出處理類
@available(macOS 14.0, *)
class AudioStreamOutput: NSObject, SCStreamOutput {
    private var isRecording = false
    private var audioConverter: AVAudioConverter?
    private var outputFormat: AVAudioFormat?

    func startRecording() {
        isRecording = true
    }

    func stopRecording() {
        isRecording = false
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
      guard type == .audio, isRecording else { return }

      // Get the format description from the sample buffer
      guard let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer),
            let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(formatDescription) else {
          print("Failed to get format description")
          return
      }

      // Initialize the audio converter if necessary
      if audioConverter == nil {
          // Create the input audio format
          guard let inputFormat = AVAudioFormat(streamDescription: asbd) else {
              print("Failed to create input audio format")
              return
          }
          print("Input Audio Format: \(inputFormat)")

          // Define the desired output format (16kHz, 16-bit, mono)
          outputFormat = AVAudioFormat(commonFormat: .pcmFormatInt16,
                                      sampleRate: 16000,
                                      channels: 1,
                                      interleaved: true)

          guard let outputFormat = outputFormat else {
              print("Failed to create output audio format")
              return
          }
          print("Output Audio Format: \(outputFormat)")

          // Create the audio converter
          audioConverter = AVAudioConverter(from: inputFormat, to: outputFormat)
      }

      guard let audioConverter = audioConverter,
            let outputFormat = outputFormat else {
          print("Audio converter not initialized")
          return
      }

      // Convert CMSampleBuffer to AVAudioPCMBuffer
      guard let inputPCMBuffer = sampleBufferToPCMBuffer(sampleBuffer: sampleBuffer) else {
          print("Failed to create PCM buffer from sample buffer")
          return
      }

      // Calculate the output frame capacity based on sample rate ratio
      let sampleRateRatio = outputFormat.sampleRate / inputPCMBuffer.format.sampleRate
      let outputFrameCapacity = AVAudioFrameCount(Double(inputPCMBuffer.frameLength) * sampleRateRatio)

      guard let outputPCMBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputFrameCapacity) else {
          print("Failed to create output PCM buffer")
          return
      }

      // Calculate audio level before conversion
      let decibels = calculateDecibels(from: sampleBuffer)
      // print("Decibels: \(decibels)")  // 可以保留這行來觀察分貝值
      
      // 當分貝值大於 -50 且不是負無限大時才處理音訊
      guard decibels > -50 && decibels != Float.infinity && decibels != -Float.infinity else { 
          return 
      }

      // Perform the conversion
      let inputBlock: AVAudioConverterInputBlock = { inNumPackets, outStatus in
          outStatus.pointee = .haveData
          return inputPCMBuffer
      }

      var error: NSError?
      let status = audioConverter.convert(to: outputPCMBuffer, error: &error, withInputFrom: inputBlock)

      if status == .haveData {
          // Access the interleaved audio data from the audioBufferList
          let audioBufferList = outputPCMBuffer.audioBufferList.pointee
          let audioBuffer = audioBufferList.mBuffers
          let dataSize = Int(audioBuffer.mDataByteSize)

          if let dataPointer = audioBuffer.mData {
              let bufferData = Data(bytes: dataPointer, count: dataSize)
              FileHandle.standardOutput.write(bufferData)
          }
      } else if status == .error {
          if let error = error {
              print("Audio conversion failed: \(error.localizedDescription)")
          } else {
              print("Audio conversion failed with unknown error")
          }
      }
  }

    private func sampleBufferToPCMBuffer(sampleBuffer: CMSampleBuffer) -> AVAudioPCMBuffer? {
    guard let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer),
          let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(formatDescription),
          let audioFormat = AVAudioFormat(streamDescription: asbd) else {
        return nil
    }

    let numSamples = CMSampleBufferGetNumSamples(sampleBuffer)
    guard let pcmBuffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: AVAudioFrameCount(numSamples)) else {
        return nil
    }
    pcmBuffer.frameLength = pcmBuffer.frameCapacity

    // Copy data from CMSampleBuffer to AVAudioPCMBuffer
    guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else {
        return nil
    }

    var length: Int = 0
    var totalLength: Int = 0
    var dataPointer: UnsafeMutablePointer<Int8>?

    let status = CMBlockBufferGetDataPointer(blockBuffer,
                                             atOffset: 0,
                                             lengthAtOffsetOut: &length,
                                             totalLengthOut: &totalLength,
                                             dataPointerOut: &dataPointer)

    guard status == noErr, let dataPointer = dataPointer else {
        return nil
    }

    // Copy data into the pcmBuffer using mutableAudioBufferList
    let audioBufferList = pcmBuffer.mutableAudioBufferList
    let audioBuffers = UnsafeMutableAudioBufferListPointer(audioBufferList)

    var offset = 0
    for buffer in audioBuffers {
        let dataSize = Int(buffer.mDataByteSize)
            memcpy(buffer.mData, dataPointer.advanced(by: offset), dataSize)
            offset += dataSize
        }

        return pcmBuffer
    }

    // Add this new method to calculate decibels
    private func calculateDecibels(from sampleBuffer: CMSampleBuffer) -> Float {
        guard let channelData = sampleBufferToPCMBuffer(sampleBuffer: sampleBuffer) else {
            return 0.0
        }
        
        // Get the raw audio data
        guard let data = channelData.floatChannelData else {
            return 0.0
        }
        
        // Calculate RMS (Root Mean Square)
        var rms: Float = 0.0
        let length = Int(channelData.frameLength)
        
        // Use first channel (mono)
        let samples = data[0]
        for i in 0..<length {
            rms += samples[i] * samples[i]
        }
        rms = sqrt(rms / Float(length))
        
        // Convert to decibels
        let decibels = 20 * log10(rms)
        return decibels
    }
}

@available(macOS 14.0, *)
class AudioCapture {
    private var stream: SCStream?
    private var streamOutput: AudioStreamOutput?
    
    func startCapture() async throws {
        let shareable = try await SCShareableContent.current
        guard let display = shareable.displays.first else {
            throw NSError(domain: "ScreenCapture", code: 1, userInfo: [NSLocalizedDescriptionKey: "無法獲取可分享內容"])
        }
        
        let audioConfig = SCStreamConfiguration()
        audioConfig.capturesAudio = true
        audioConfig.excludesCurrentProcessAudio = false
        
        // 設置音訊格式為 16kHz, 16-bit, 單聲道
        audioConfig.sampleRate = 16000
        audioConfig.channelCount = 1
        
        let filter = SCContentFilter(display: display, excludingApplications: [], exceptingWindows: [])
        stream = SCStream(filter: filter, configuration: audioConfig, delegate: nil)
        
        streamOutput = AudioStreamOutput()
        
        guard let stream = stream, let streamOutput = streamOutput else {
            throw NSError(domain: "ScreenCapture", code: 2, userInfo: [NSLocalizedDescriptionKey: "串流初始化失敗"])
        }
        
        try stream.addStreamOutput(streamOutput, type: .audio, sampleHandlerQueue: .global())
        try await stream.startCapture()
        streamOutput.startRecording()
    }
    
    func stopCapture() async throws {
        guard let stream = stream, let streamOutput = streamOutput else { return }
        streamOutput.stopRecording()
        try await stream.stopCapture()
        self.stream = nil
        self.streamOutput = nil
    }
}

@available(macOS 14.0, *)
func main() async {
    print("開始執行")
    
    // 請求權限
    let hasPermission = await requestScreenCapturePermission()
    guard hasPermission else {
        print("未獲得螢幕錄製權限")
        return
    }
    
    let audioCapture = AudioCapture()
    
    do {
        try await audioCapture.startCapture()
        
        // 錄製 10 秒後停止
        // try await audioCapture.stopCapture()
        // 等待直到程序被手動中斷
        try await withUnsafeThrowingContinuation { (continuation: UnsafeContinuation<Void, Error>) in
        }
        // try await audioCapture.stopCapture()

        print("錄製完成")
        // exit(0)
    } catch {
        print("錄製失敗: \(error.localizedDescription)")
        exit(1)
    }
}

// 啟動程序時進行版本檢查
if #available(macOS 14.0, *) {
    Task {
        await main()
    }
} else {
    print("此程式需要 macOS 14.0 或更新版本")
}

RunLoop.main.run()