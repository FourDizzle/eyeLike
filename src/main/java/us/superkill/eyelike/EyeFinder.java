package us.superkill.eyelike;

import static org.bytedeco.javacpp.opencv_core.split;
import static org.bytedeco.javacpp.opencv_imgproc.GaussianBlur;
import static us.superkill.eyelike.DetectConfig.kEyePercentHeight;
import static us.superkill.eyelike.DetectConfig.kEyePercentSide;
import static us.superkill.eyelike.DetectConfig.kEyePercentTop;
import static us.superkill.eyelike.DetectConfig.kEyePercentWidth;
import static us.superkill.eyelike.DetectConfig.kSmoothFaceFactor;
import static us.superkill.eyelike.DetectConfig.kSmoothFaceImage;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.Size;

public class EyeFinder {
	
	private static final Logger log = LogManager.getLogger(EyeFinder.class);
	
	public static Rect[] findEyes(Mat image, Rect face) throws Exception {
		log.debug("Looking for eyes.");

		MatVector rgbChannels = new MatVector();
		split(image, rgbChannels);
		Mat imageGray = rgbChannels.get(2);

		//TODO check that this is right
		Mat faceROI = new Mat(imageGray, face);
		
		if (kSmoothFaceImage) {
			double sigma = kSmoothFaceFactor * face.width();
			Size blurSize = new Size(0, 0);
		    GaussianBlur(faceROI, faceROI, blurSize, sigma);
		    blurSize.close();
		}
		
		//Find eyes
		int eyeRegionWidth = 
				(int) Math.round(face.width() * ((double) kEyePercentWidth/100));
		int eyeRegionHeight =
				(int) Math.round(face.height() * ((double) kEyePercentHeight/100));
		int eyeRegionTop = 
				(int) Math.round(face.height() * ((double) kEyePercentTop/100));
		
		Rect leftEyeRegion = 
				new Rect((int) Math.round(face.width() * ((double) kEyePercentSide/100)),
				         eyeRegionTop,eyeRegionWidth,eyeRegionHeight);
		
		Rect rightEyeRegion = 
				new Rect((int) Math.round(face.width() - eyeRegionWidth - 
						    face.width()*((double) kEyePercentSide/100)),
				         eyeRegionTop,eyeRegionWidth,eyeRegionHeight);
		
		log.debug("returning eye areas");
		
		faceROI.close();
		
		return new Rect[] {leftEyeRegion, rightEyeRegion};
	}
}