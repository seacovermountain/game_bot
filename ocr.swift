import Vision
import AppKit

// 1. 从标准输入读取二进制数据
let data = FileHandle.standardInput.readDataToEndOfFile()

// 2. 将 Data 转换为 NSImage
guard let image = NSImage(data: data),
      let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    // 如果图片无效，静默退出
    exit(1)
}

let request = VNRecognizeTextRequest { (request, _) in
    guard let observations = request.results as? [VNRecognizedTextObservation] else { return }
    for observation in observations {
        if let candidate = observation.topCandidates(1).first {
            print(candidate.string)
        }
    }
}

request.recognitionLanguages = ["zh-Hans"]
request.recognitionLevel = .accurate

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
try? handler.perform([request])