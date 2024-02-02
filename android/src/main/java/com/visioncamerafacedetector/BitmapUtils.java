package com.visioncamerafacedetector;

import android.graphics.BitmapFactory;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.media.Image;
import android.media.Image.Plane;
import android.util.Base64;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;

/** Utils functions for bitmap conversions. */
public class BitmapUtils {
  private static final String TAG = "BitmapUtils";

  private static final ByteArrayOutputStream outputStream = new ByteArrayOutputStream();

  public static String convertYuvToRgba(Image image) {
    Image.Plane[] planes = image.getPlanes();
    int width = image.getWidth();
    int height = image.getHeight();

    ByteBuffer yBuffer = planes[0].getBuffer();
    ByteBuffer uvBuffer = planes[1].getBuffer();

    int ySize = yBuffer.remaining();
    int uvSize = uvBuffer.remaining();

    byte[] nv21Data = new byte[ySize + uvSize];

    // Copy Y plane
    yBuffer.get(nv21Data, 0, ySize);

    // Copy interleaved U and V planes
    uvBuffer.get(nv21Data, ySize, uvSize);

    // Convert NV21 to RGB
    int[] rgbaArray = new int[width * height];

    outputStream.reset();
    YuvImage yuvImage = new YuvImage(nv21Data, ImageFormat.NV21, width, height, null);
    yuvImage.compressToJpeg(new Rect(0, 0, width, height), 90, outputStream);

    byte[] jpegData = outputStream.toByteArray();
    Bitmap bitmap = BitmapFactory.decodeByteArray(jpegData, 0, jpegData.length);

    // Get RGB values from the bitmap
    bitmap.getPixels(rgbaArray, 0, width, 0, 0, width, height);

    // Create Bitmap from RGB array
    Bitmap rgbaBitmap = Bitmap.createBitmap(rgbaArray, width, height, Bitmap.Config.RGB_565);

    outputStream.reset();
    rgbaBitmap.compress(Bitmap.CompressFormat.PNG, 90, outputStream);
    return Base64.encodeToString(outputStream.toByteArray(), Base64.DEFAULT);
  }
}
