import VisionCamera
import Foundation
import MLKitFaceDetection
import MLKitVision
import CoreML
import UIKit
import AVFoundation
import SceneKit
import Accelerate

@objc(VisionCameraFaceDetector)
public class VisionCameraFaceDetector: FrameProcessorPlugin {
  // device display data  
  private let screenBounds = UIScreen.main.bounds

  // detection props
  private var autoScale = false
  private var faceDetector: FaceDetector! = nil
  private var runLandmarks = false
  private var runClassifications = false
  private var runContours = false
  private var trackingEnabled = false

  public override init(
    proxy: VisionCameraProxyHolder, 
    options: [AnyHashable : Any]! = [:]
  ) {
    super.init(proxy: proxy, options: options)
    let config = getConfig(withArguments: options)

    // handle auto scaling
    autoScale = config?["autoScale"] as? Bool == true

    // initializes faceDetector on creation
    let minFaceSize = 0.15
    let optionsBuilder = FaceDetectorOptions()
        optionsBuilder.performanceMode = .fast
        optionsBuilder.landmarkMode = .none
        optionsBuilder.contourMode = .none
        optionsBuilder.classificationMode = .none
        optionsBuilder.minFaceSize = minFaceSize
        optionsBuilder.isTrackingEnabled = false

    if config?["performanceMode"] as? String == "accurate" {
      optionsBuilder.performanceMode = .accurate
    }

    if config?["landmarkMode"] as? String == "all" {
      runLandmarks = true
      optionsBuilder.landmarkMode = .all
    }

    if config?["classificationMode"] as? String == "all" {
      runClassifications = true
      optionsBuilder.classificationMode = .all
    }

    if config?["contourMode"] as? String == "all" {
      runContours = true
      optionsBuilder.contourMode = .all
    }

    let minFaceSizeParam = config?["minFaceSize"] as? Double
    if minFaceSizeParam != nil && minFaceSizeParam != minFaceSize {
      optionsBuilder.minFaceSize = CGFloat(minFaceSizeParam!)
    }

    if config?["trackingEnabled"] as? Bool == true {
      trackingEnabled = true
      optionsBuilder.isTrackingEnabled = true
    }

    faceDetector = FaceDetector.faceDetector(options: optionsBuilder)
  }

  func getConfig(
    withArguments arguments: [AnyHashable: Any]!
  ) -> [String:Any]! {
    if arguments.count > 0 {
      let config = arguments.map { dictionary in
        Dictionary(uniqueKeysWithValues: dictionary.map { (key, value) in
          (key as? String ?? "", value)
        })
      }

      return config
    }

    return nil
  }

  func processBoundingBox(
    from face: Face,
    scaleX: CGFloat,
    scaleY: CGFloat
  ) -> [String:Any] {
    let boundingBox = face.frame

    return [
      "width": boundingBox.width * scaleX,
      "height": boundingBox.height * scaleY,
      "x": boundingBox.origin.x * scaleX,
      "y": boundingBox.origin.y * scaleY
    ]
  }

  func processLandmarks(
    from face: Face,
    scaleX: CGFloat,
    scaleY: CGFloat
  ) -> [String:[String: CGFloat?]] {
    let faceLandmarkTypes = [
      FaceLandmarkType.leftCheek,
      FaceLandmarkType.leftEar,
      FaceLandmarkType.leftEye,
      FaceLandmarkType.mouthBottom,
      FaceLandmarkType.mouthLeft,
      FaceLandmarkType.mouthRight,
      FaceLandmarkType.noseBase,
      FaceLandmarkType.rightCheek,
      FaceLandmarkType.rightEar,
      FaceLandmarkType.rightEye
    ]

    let faceLandmarksTypesStrings = [
      "LEFT_CHEEK",
      "LEFT_EAR",
      "LEFT_EYE",
      "MOUTH_BOTTOM",
      "MOUTH_LEFT",
      "MOUTH_RIGHT",
      "NOSE_BASE",
      "RIGHT_CHEEK",
      "RIGHT_EAR",
      "RIGHT_EYE"
    ];

    var faceLandMarksTypesMap: [String: [String: CGFloat?]] = [:]
    for i in 0..<faceLandmarkTypes.count {
      let landmark = face.landmark(ofType: faceLandmarkTypes[i]);
      let position = [
        "x": landmark?.position.x ?? 0.0 * scaleX,
        "y": landmark?.position.y ?? 0.0 * scaleY
      ]
      faceLandMarksTypesMap[faceLandmarksTypesStrings[i]] = position
    }

    return faceLandMarksTypesMap
  }

  func processFaceContours(
    from face: Face,
    scaleX: CGFloat,
    scaleY: CGFloat
  ) -> [String:[[String:CGFloat]]] {
    let faceContoursTypes = [
      FaceContourType.face,
      FaceContourType.leftCheek,
      FaceContourType.leftEye,
      FaceContourType.leftEyebrowBottom,
      FaceContourType.leftEyebrowTop,
      FaceContourType.lowerLipBottom,
      FaceContourType.lowerLipTop,
      FaceContourType.noseBottom,
      FaceContourType.noseBridge,
      FaceContourType.rightCheek,
      FaceContourType.rightEye,
      FaceContourType.rightEyebrowBottom,
      FaceContourType.rightEyebrowTop,
      FaceContourType.upperLipBottom,
      FaceContourType.upperLipTop
    ]

    let faceContoursTypesStrings = [
      "FACE",
      "LEFT_CHEEK",
      "LEFT_EYE",
      "LEFT_EYEBROW_BOTTOM",
      "LEFT_EYEBROW_TOP",
      "LOWER_LIP_BOTTOM",
      "LOWER_LIP_TOP",
      "NOSE_BOTTOM",
      "NOSE_BRIDGE",
      "RIGHT_CHEEK",
      "RIGHT_EYE",
      "RIGHT_EYEBROW_BOTTOM",
      "RIGHT_EYEBROW_TOP",
      "UPPER_LIP_BOTTOM",
      "UPPER_LIP_TOP"
    ];

    var faceContoursTypesMap: [String:[[String:CGFloat]]] = [:]
    for i in 0..<faceContoursTypes.count {
      let contour = face.contour(ofType: faceContoursTypes[i]);
      var pointsArray: [[String:CGFloat]] = []

      if let points = contour?.points {
        for point in points {
          let currentPointsMap = [
            "x": point.x * scaleX,
            "y": point.y * scaleY,
          ]

          pointsArray.append(currentPointsMap)
        }

        faceContoursTypesMap[faceContoursTypesStrings[i]] = pointsArray
      }
    }

    return faceContoursTypesMap
  }

  public override func callback(
    _ frame: Frame, 
    withArguments arguments: [AnyHashable: Any]?
  ) -> Any {
    var result: [Any] = []
    var resultData: [String: Any] = [:]
    var normalizedBrightness:Float = 0.5;

    do {
      let image = VisionImage(buffer: frame.buffer)
      image.orientation = frame.orientation

      var scaleX:CGFloat
      var scaleY:CGFloat
      if autoScale {
        scaleX = screenBounds.size.width / CGFloat(frame.width)
        scaleY = screenBounds.size.height / CGFloat(frame.height)
      } else {
        scaleX = CGFloat(1)
        scaleY = CGFloat(1)
      }

      // Convert CMSampleBuffer to CVImageBuffer
      if let imageBuffer = CMSampleBufferGetImageBuffer(frame.buffer) {
          // Calculate and normalize average brightness
          let avgBrightness = calculateAverageBrightness(from: imageBuffer)
          normalizedBrightness = normalizeBrightness(avgBrightness)
          // print("Normalized Brightness: \(normalizedBrightness)")
      } else {
          // print("Failed to get CVImageBuffer from CMSampleBuffer.")
      }

      let faces: [Face] = try faceDetector!.results(in: image)
      for face in faces {
        var map: [String: Any] = [:]

        if runLandmarks {
          map["landmarks"] = processLandmarks(
            from: face,
            scaleX: scaleX,
            scaleY: scaleY
          )
        }

        if runClassifications {
          map["leftEyeOpenProbability"] = face.leftEyeOpenProbability
          map["rightEyeOpenProbability"] = face.rightEyeOpenProbability
          map["smilingProbability"] = face.smilingProbability
        }

        if runContours {
          map["contours"] = processFaceContours(
            from: face,
            scaleX: scaleX,
            scaleY: scaleY
          )
        }

        if trackingEnabled {
          map["trackingId"] = face.trackingID
        }

        map["rollAngle"] = face.headEulerAngleZ
        map["pitchAngle"] = face.headEulerAngleX
        map["yawAngle"] = face.headEulerAngleY
        map["bounds"] = processBoundingBox(
          from: face,
          scaleX: scaleX,
          scaleY: scaleY
        )

        result.append(map)
      }
    } catch let error {
      print("Error processing face detection: \(error)")
    }
    resultData["faces"] = result
    resultData["brightness"] = normalizedBrightness
    return resultData
  }

  // Function to calculate the average brightness from an image buffer
  func calculateAverageBrightness(from buffer: CVImageBuffer) -> Float {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    
    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    let baseAddress = CVPixelBufferGetBaseAddress(buffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    
    guard let baseAddress = baseAddress else {
        CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
        return 0
    }
    
    let pixelBuffer = baseAddress.assumingMemoryBound(to: UInt8.self)
    let luminanceFactors = SIMD3<Float>(0.299, 0.587, 0.114)
    
    // Multithreaded processing
    let group = DispatchGroup()
    let concurrentQueue = DispatchQueue(label: "brightnessQueue", attributes: .concurrent)
    
    var totalBrightness: Float = 0
    let pixelCount = (width / 15) * (height / 15)
    let chunkSize = height / ProcessInfo.processInfo.activeProcessorCount
    
    let totalBrightnessPointer = UnsafeMutablePointer<Float>.allocate(capacity: 1)
    totalBrightnessPointer.pointee = 0
    
    for chunkStart in stride(from: 0, to: height, by: chunkSize) {
        let chunkEnd = min(chunkStart + chunkSize, height)
        
        concurrentQueue.async(group: group) {
            var localBrightness: Float = 0
            for y in stride(from: chunkStart, to: chunkEnd, by: 20) {
                let rowPointer = pixelBuffer + y * bytesPerRow
                for x in stride(from: 0, to: width, by: 20) {
                    let pixelPointer = rowPointer + x * 4
                    let red = Float(pixelPointer[0]) / 255.0
                    let green = Float(pixelPointer[1]) / 255.0
                    let blue = Float(pixelPointer[2]) / 255.0
                    
                    let colorVector = SIMD3<Float>(red, green, blue)
                    localBrightness += dot(colorVector, luminanceFactors)
                }
            }
            
            // Safely update the totalBrightness across threads
            DispatchQueue.global().sync {
                totalBrightnessPointer.pointee += localBrightness
            }
        }
    }
    
    group.wait()
    
    let averageBrightness = totalBrightnessPointer.pointee / Float(pixelCount)
    totalBrightnessPointer.deallocate()
    
    CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
    
    return averageBrightness
}

  // Function to normalize brightness to a range [0, 1]
  func normalizeBrightness(_ brightness: Float) -> Float {
      return min(max(brightness, 0), 1)  // Ensuring the value stays within [0, 1]
  }
}
