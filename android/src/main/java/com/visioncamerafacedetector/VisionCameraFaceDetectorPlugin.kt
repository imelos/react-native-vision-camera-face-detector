package com.visioncamerafacedetector

import android.content.res.Resources
import android.graphics.Rect
import android.util.Log
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceContour
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import com.mrousavy.camera.core.FrameInvalidError
import com.mrousavy.camera.frameprocessor.Frame
import com.mrousavy.camera.frameprocessor.FrameProcessorPlugin
import com.mrousavy.camera.frameprocessor.VisionCameraProxy
import android.media.Image;
import java.nio.ByteBuffer;

private const val TAG = "FaceDetector"
class VisionCameraFaceDetectorPlugin(
  proxy: VisionCameraProxy,
  options: Map<String, Any>?
) : FrameProcessorPlugin() {
  // device display data
  private val density = Resources.getSystem().displayMetrics.density.toInt()
  private val windowWidth = Resources.getSystem().displayMetrics.widthPixels / density
  private val windowHeight = Resources.getSystem().displayMetrics.heightPixels / density

  // detection props
  private var faceDetector: FaceDetector? = null
  private var runLandmarks = false
  private var runClassifications = false
  private var runContours = false
  private var trackingEnabled = false
  private var returnOriginal = false
  private var convertFrame = false

  init {
    // initializes faceDetector on creation
    var performanceModeValue = FaceDetectorOptions.PERFORMANCE_MODE_FAST
    var landmarkModeValue = FaceDetectorOptions.LANDMARK_MODE_NONE
    var classificationModeValue = FaceDetectorOptions.CLASSIFICATION_MODE_NONE
    var contourModeValue = FaceDetectorOptions.CONTOUR_MODE_NONE
    var minFaceSize = 0.15f

    if (options?.get("performanceMode").toString() == "accurate") {
      performanceModeValue = FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE
    }

    if (options?.get("landmarkMode").toString() == "all") {
      runLandmarks = true
      landmarkModeValue = FaceDetectorOptions.LANDMARK_MODE_ALL
    }

    if (options?.get("classificationMode").toString() == "all") {
      runClassifications = true
      classificationModeValue = FaceDetectorOptions.CLASSIFICATION_MODE_ALL
    }

    if (options?.get("contourMode").toString() == "all") {
      runContours = true
      contourModeValue = FaceDetectorOptions.CONTOUR_MODE_ALL
    }

    val minFaceSizeParam = options?.get("minFaceSize").toString()
    if (
      minFaceSizeParam != "null" &&
      minFaceSizeParam != minFaceSize.toString()
    ) {
      minFaceSize = minFaceSizeParam.toFloat()
    }

    val optionsBuilder = FaceDetectorOptions.Builder()
      .setPerformanceMode(performanceModeValue)
      .setLandmarkMode(landmarkModeValue)
      .setContourMode(contourModeValue)
      .setClassificationMode(classificationModeValue)
      .setMinFaceSize(minFaceSize)

    if (options?.get("trackingEnabled").toString() == "true") {
      trackingEnabled = true
      optionsBuilder.enableTracking()
    }

    faceDetector = FaceDetection.getClient(
      optionsBuilder.build()
    )

    // also check about returing frame settings
    returnOriginal = options?.get("returnOriginal").toString() == "true"
    convertFrame = options?.get("convertFrame").toString() == "true"
  }

  private fun processBoundingBox(
    boundingBox: Rect,
  ): Map<String, Any> {
    val bounds: MutableMap<String, Any> = HashMap()
    val width = boundingBox.width().toDouble()

    bounds["width"] = width
    bounds["height"] = boundingBox.height().toDouble()
    bounds["x"] = windowWidth - (width + (
      boundingBox.left.toDouble()
    ))
    bounds["y"] = boundingBox.top.toDouble()

    return bounds
  }

  private fun processLandmarks(
    face: Face,
  ): Map<String, Any> {
    val faceLandmarksTypes = intArrayOf(
      FaceLandmark.LEFT_CHEEK,
      FaceLandmark.LEFT_EAR,
      FaceLandmark.LEFT_EYE,
      FaceLandmark.MOUTH_BOTTOM,
      FaceLandmark.MOUTH_LEFT,
      FaceLandmark.MOUTH_RIGHT,
      FaceLandmark.NOSE_BASE,
      FaceLandmark.RIGHT_CHEEK,
      FaceLandmark.RIGHT_EAR,
      FaceLandmark.RIGHT_EYE
    )
    val faceLandmarksTypesStrings = arrayOf(
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
    )
    val faceLandmarksTypesMap: MutableMap<String, Any> = HashMap()
    for (i in faceLandmarksTypesStrings.indices) {
      val landmark = face.getLandmark(faceLandmarksTypes[i])
      val landmarkName = faceLandmarksTypesStrings[i]
      Log.d(
        TAG,
        "Getting '$landmarkName' landmark"
      )
      if (landmark == null) {
        Log.d(
          TAG,
          "Landmark '$landmarkName' is null - going next"
        )
        continue
      }
      val point = landmark.position
      val currentPointsMap: MutableMap<String, Double> = HashMap()
      currentPointsMap["x"] = point.x.toDouble()
      currentPointsMap["y"] = point.y.toDouble()
      faceLandmarksTypesMap[landmarkName] = currentPointsMap
    }

    return faceLandmarksTypesMap
  }

  private fun processFaceContours(
    face: Face,
  ): Map<String, Any> {
    val faceContoursTypes = intArrayOf(
      FaceContour.FACE,
      FaceContour.LEFT_CHEEK,
      FaceContour.LEFT_EYE,
      FaceContour.LEFT_EYEBROW_BOTTOM,
      FaceContour.LEFT_EYEBROW_TOP,
      FaceContour.LOWER_LIP_BOTTOM,
      FaceContour.LOWER_LIP_TOP,
      FaceContour.NOSE_BOTTOM,
      FaceContour.NOSE_BRIDGE,
      FaceContour.RIGHT_CHEEK,
      FaceContour.RIGHT_EYE,
      FaceContour.RIGHT_EYEBROW_BOTTOM,
      FaceContour.RIGHT_EYEBROW_TOP,
      FaceContour.UPPER_LIP_BOTTOM,
      FaceContour.UPPER_LIP_TOP
    )
    val faceContoursTypesStrings = arrayOf(
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
    )
    val faceContoursTypesMap: MutableMap<String, Any> = HashMap()
    for (i in faceContoursTypesStrings.indices) {
      val contour = face.getContour(faceContoursTypes[i])
      val contourName = faceContoursTypesStrings[i]
      Log.d(
        TAG,
        "Getting '$contourName' contour"
      )
      if (contour == null) {
        Log.d(
          TAG,
          "Face contour '$contourName' is null - going next"
        )
        continue
      }
      val points = contour.points
      val pointsMap: MutableMap<String, Map<String, Double>> = HashMap()
      for (j in points.indices) {
        val currentPointsMap: MutableMap<String, Double> = HashMap()
        currentPointsMap["x"] = points[j].x.toDouble()
        currentPointsMap["y"] = points[j].y.toDouble()
        pointsMap[j.toString()] = currentPointsMap
      }

      faceContoursTypesMap[contourName] = pointsMap
    }
    return faceContoursTypesMap
  }

  override fun callback(
    frame: Frame,
    params: Map<String, Any>?
  ): Map<String, Any> {
    val resultMap: MutableMap<String, Any> = HashMap()

    try {
      val frameImage = frame.image
      val orientation = frame.orientation

      if (
        frameImage == null &&
        orientation == null
      ) {
        Log.i(TAG, "Image or orientation is null")
        return resultMap
      }

      val planes: Array<Image.Plane> = frameImage.planes
      val yPlaneBuffer: ByteBuffer = planes[0].buffer // Y plane contains the luminance information

      // Optional: You could downsample here by only reading every nth pixel
      val width: Int = frameImage.width
      val height: Int = frameImage.height
      val pixelStride: Int = planes[0].pixelStride
      val rowStride: Int = planes[0].rowStride
      val rowPadding: Int = rowStride - pixelStride * width

      var totalLuminance: Long = 0
      var pixelCount = 0

      // Loop over the Y plane buffer and calculate the total luminance
      for (y in 0 until height step 50) {
          for (x in 0 until width step 50) {
              val luminance: Int = yPlaneBuffer[y * rowStride + x * pixelStride].toInt() and 0xFF // Convert to unsigned
              totalLuminance += luminance
              pixelCount++
          }
          yPlaneBuffer.position(yPlaneBuffer.position() + rowPadding) // Skip the row padding
      }

      // Calculate the average brightness
      val averageBrightness: Float = totalLuminance.toFloat() / pixelCount
      val normalizedAverageBrightness: Float = averageBrightness / 255.0f

      val rotation = orientation!!.toDegrees()
      val image = InputImage.fromMediaImage(frameImage!!, rotation)

      val task = faceDetector!!.process(image)
      val faces = Tasks.await(task)
      val facesList = ArrayList<Map<String, Any?>>()

      faces.forEach{face ->
        val map: MutableMap<String, Any?> = HashMap()

        if (runLandmarks) {
          map["landmarks"] = processLandmarks(
            face
          )
        }

        if (runClassifications) {
          map["leftEyeOpenProbability"] = face.leftEyeOpenProbability?.toDouble() ?: -1
          map["rightEyeOpenProbability"] = face.rightEyeOpenProbability?.toDouble() ?: -1
          map["smilingProbability"] = face.smilingProbability?.toDouble() ?: -1
        }

        if (runContours) {
          map["contours"] = processFaceContours(
            face
          )
        }

        if (trackingEnabled) {
          map["trackingId"] = face.trackingId
        }

        map["rollAngle"] = face.headEulerAngleZ.toDouble()
        map["pitchAngle"] = face.headEulerAngleX.toDouble()
        map["yawAngle"] = face.headEulerAngleY.toDouble()
        map["bounds"] = processBoundingBox(
          face.boundingBox
        )
        facesList.add(map)
      }

      val frameMap: MutableMap<String, Any> = HashMap()
      if (returnOriginal) {
        frameMap["original"] = frame
      }

      if (convertFrame) {
        frameMap["converted"] = BitmapUtils.convertYuvToRgba(frameImage)
      }

      resultMap["faces"] = facesList
      // if(returnOriginal || convertFrame) {
      //   resultMap["frame"] = frameMap
      // }
      resultMap["frame"] = frameMap
      frameMap["brightness"] = normalizedAverageBrightness.toDouble()
      frameMap["width"] = width.toDouble()
      frameMap["height"] = height.toDouble()
    } catch (e: Exception) {
      Log.e(TAG, "Error processing face detection: ", e)
    } catch (e: FrameInvalidError) {
      Log.e(TAG, "Frame invalid error: ", e)
    }

    return resultMap
  }
}
