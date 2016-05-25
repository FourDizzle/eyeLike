package us.superkill.eyelike;

import static org.bytedeco.javacpp.opencv_core.CV_64F;
import static org.bytedeco.javacpp.opencv_core.minMaxLoc;
import static org.bytedeco.javacpp.opencv_core.split;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.GaussianBlur;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static us.superkill.eyelike.DetectConfig.kEnableWeight;
import static us.superkill.eyelike.DetectConfig.kFastEyeWidth;
import static us.superkill.eyelike.DetectConfig.kGradientThreshold;
import static us.superkill.eyelike.DetectConfig.kWeightBlurSize;
import static us.superkill.eyelike.DetectConfig.kWeightDivisor;
import static us.superkill.eyelike.Helper.computeDynamicThreshold;
import static us.superkill.eyelike.Helper.matrixMagnitude;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.indexer.*;

public class PupilFinder {
	
	private static final Logger log = LogManager.getLogger(PupilFinder.class);
	
	private static Point unscalePoint(Point p, Rect origSize) {
		float ratio = (((float)kFastEyeWidth)/origSize.width());
		int x = (int) Math.round(p.x() / ratio);
		int y = (int) Math.round(p.y() / ratio);
		
		return new Point(x, y);
	}
	
	private static Mat scaleToFastSize(Mat src) {
		resize(src, src, 
				new Size(kFastEyeWidth, 
						(kFastEyeWidth/src.cols()) * src.rows()));
		return src;
	}
	
	private static Mat computeMatXGradient(Mat mat) {
		mat.convertTo(mat, CV_64F);
		Mat output = new Mat(mat.rows(), mat.cols(), CV_64F);
		DoubleIndexer outIndex = output.createIndexer();
		DoubleIndexer matIndex = mat.createIndexer();
		for(int y = 0; y < mat.rows(); y++) {
			double a = matIndex.get(y,1) - matIndex.get(y,0);
//			double a = mat.ptr(y,1).get() - mat.ptr(y,0).get();
			outIndex.put(y, 0, a);
//			output.ptr(y,0).put((byte)(mat.ptr(y,1).get() - mat.ptr(y,0).get()));
			for(int x = 1; x < mat.cols() - 1; x++) {
				double b = (matIndex.get(y,x+1) - matIndex.get(y,x-1))/2.0;
				outIndex.put(y, x, b);
//				output.ptr(y,x).put((byte)((mat.ptr(y,x+1).get() - mat.ptr(y,x-1).get())/2.0));
			}
		}
		imwrite("Users/ncassiani/Projects/MeanMachine/testimg/xgrad.jpg", output);
		return output;
	}
	
	private static Mat testPossibleCentersFormula(
			int x, int y, Mat weight, double gx, double gy, Mat out) {
		//for all possible centers
		DoubleIndexer weightIndex = weight.createIndexer();
		DoubleIndexer outIndex = out.createIndexer();
		for (int cy = 0; cy < out.rows(); cy++) {
			for (int cx = 0; cx < out.cols(); cx++) {
				if ((x == cx && y == cy) == false) {
					// create a vector from the possible center 
					// to the gradient origin
					double dx = x - cx;
					double dy = y - cy;
					// normalize d
					double magnitude = Math.sqrt((dx * dx) + (dy * dy));
					dx = dx / magnitude;
					dy = dy / magnitude;
					double dotProduct = dx*gx + dy*gy;
					dotProduct = Math.max(0.0, dotProduct);
					//square and multiply by the weight
					if (kEnableWeight) {
						outIndex.put(cy, cx, outIndex.get(cy,cx) 
								+ dotProduct * dotProduct 
								* (weightIndex.get(cy, cx)/kWeightDivisor));
//						out.ptr(cy, cx).put((byte)
//								(out.ptr(cy, cx).get()
//									+ dotProduct * dotProduct
//									* (weight.ptr(cy, cx).get()/kWeightDivisor)));
					} else {
						outIndex.put(cy, cx, outIndex.get(cy,cx) 
								+ dotProduct * dotProduct);
//						out.ptr(cy, cx).put((byte) 
//								(out.ptr(cy, cx).get() + dotProduct * dotProduct));
					}
				}
			}
		}
		return out;
	}
	
	public static Point findEyeCenter(Mat face, Rect eye) {
		Mat eyeROIUnscaled = new Mat(face, eye);
		Mat eyeROI = scaleToFastSize(eyeROIUnscaled);

		imwrite("/Users/ncassiani/Projects/MeanMachine/testimg/eye.jpg", eyeROI);
		
		//Convert to grayscale
//		MatVector rgbChannels = new MatVector();
//		split(eyeROI, rgbChannels);
//		eyeROI = rgbChannels.get(2);
		cvtColor(eyeROI, eyeROI, COLOR_RGB2GRAY);
		//Find the gradient
		Mat gradientX = computeMatXGradient(eyeROI);
		Mat gradientY = computeMatXGradient(eyeROI.t().asMat()).t().asMat();
		
		// Normalize and threshold the gradient
		// compute all the magnitudes
		Mat mags = matrixMagnitude(gradientX, gradientY);
		
		// Compute the threshold
		double gradientThresh = 
				computeDynamicThreshold(mags, kGradientThreshold);
		
		//normalize
		DoubleIndexer gradXIndex = gradientX.createIndexer();
		DoubleIndexer gradYIndex = gradientY.createIndexer();
		DoubleIndexer magIndex = mags.createIndexer();
		for (int y = 0; y < eyeROI.rows(); ++y) {
			for(int x = 0; x < eyeROI.cols(); ++x) {
				long[] p = new long[] {y, x};
//				double gX = gradXIndex.getDouble(p);
				double gX = gradXIndex.get(y, x);
				double gY = gradYIndex.get(y, x);
				double magnitude = magIndex.get(y, x);
				if (magnitude > gradientThresh) {
					gradXIndex.put(y, x, gX/magnitude);
					gradYIndex.put(y, x, gY/magnitude);
				} else {
					gradXIndex.put(y, x, 0.0);
					gradYIndex.put(y, x, 0.0);
				}
			}
		}
		
		// Create a blurred and inverted image for weighting
		Mat weight = new Mat();
		GaussianBlur(eyeROI, 
				weight, new Size(kWeightBlurSize, kWeightBlurSize), 0.0);
		DoubleIndexer weightIndex = weight.createIndexer();
		for (int y = 0; y < weight.rows(); y++) {
			for (int x = 0; x < weight.cols(); x++) {
				weightIndex.put(y, x, 255 - weightIndex.get(y, x));
//				weight.ptr(y, x).put((byte)(255 - weight.ptr(y, x).get()));
			}
		}
		
		 //-- Run the algorithm!
		Mat outSum = Mat.zeros(eyeROI.rows(), eyeROI.cols(), CV_64F).asMat();
		// for each possible gradient location
		// Note: these loops are reversed from the way the paper does them
		// it evaluates every possible center for each gradient location instead of
		// every possible gradient location for every center.
		log.debug("Eye Size: " + outSum.rows() + "x" + outSum.cols());
		for (int y = 0; y < weight.rows(); y++) {
			for (int x = 0; x < weight.cols(); x++) {
				double gX = gradXIndex.get(y, x);
				double gY = gradYIndex.get(y, x);
				if ((gX == 0.0 && gY == 0.0) == false) {
					outSum = testPossibleCentersFormula(
							x, y, weight, gX, gY, outSum);
				}
			}
		}
		
		// scale all the values down, basically averaging them
		double numGradients = (weight.rows()*weight.cols());
		Mat out = new Mat();
		outSum.convertTo(out, CV_64F, 1.0/numGradients, 0.0);
		imwrite("/Users/ncassiani/Projects/MeanMachine/testimg/out.jpg", out);
		// find the maximum point
//		MinMaxLocResult outMinMaxLocResult = Core.minMaxLoc(out);
		Point minLoc = new Point(100, 200);
		Point maxLoc = new Point();
		double[] minVal = {0.0};
		double[] maxVal = {0.0};
		Mat mask = new Mat(out.size());
		minMaxLoc(out, null, maxVal, null, maxLoc, null);
		int x = maxLoc.x();
		int y = maxLoc.y();
		int minx = minLoc.x();
		int miny = minLoc.y();
//		int[] minP = {0 ,0};
//		int[] maxP = {0 ,0};
//		minMaxLoc(out, minVal, maxVal, minP, maxP);
//		Point maxP = outMinMaxLocResult.maxLoc;
//		double maxVal = outMinMaxLocResult.maxVal;
		
		return unscalePoint(maxLoc, eye);
	}
}
