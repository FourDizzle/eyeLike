package us.superkill.eyelike;

import static org.bytedeco.javacpp.opencv_core.CV_64F;

import org.bytedeco.javacpp.opencv_core.Mat;

public class Helper {
	
//	public static Mat matrixMagnitude(Mat matX, Mat matY) {
//		Mat mags = new Mat(matX.rows(), matX.cols(), CV_64F);
//		for (int y = 0; y < matX.rows(); ++y) {
//			for(int x = 0; x < matX.cols(); ++x) {
//				double gX = matX.get(y, x)[0];
//				double gY = matY.get(y, x)[0];
//				double magnitude = Math.sqrt((gX * gX) + (gY * gY));
//				mags.put(y, x, magnitude);
//			}
//		}
//		return mags;
//	}
	
	public static Mat matrixMagnitude(Mat matX, Mat matY) {
		Mat mags = new Mat(matX.rows(), matX.cols(), CV_64F);
		for (int y = 0; y < matX.rows(); ++y) {
			for(int x = 0; x < matX.cols(); ++x) {
				double gX = matX.ptr(y, x).get();
				double gY = matY.ptr(y, x).get();
				double magnitude = Math.sqrt((gX * gX) + (gY * gY));
				mags.ptr(y, x).put((byte) magnitude);
			}
		}
		return mags;
	}
	
	public static double computeDynamicThreshold(Mat mat, double stdDevFactor) {
		double meanMagnGrad = computeMean(mat);
		double stdMagnGrad = computeStdDev(mat, meanMagnGrad);
		
		double stdDev = stdMagnGrad / Math.sqrt(mat.rows()*mat.cols());
		
		return stdDevFactor * stdDev + stdMagnGrad;
	}
	
	private static double computeMean(Mat mat) {
		double sum = 0.0;
		int totalPix = mat.rows() * mat.cols();
		
		for (int y = 0; y < mat.rows() - 1; ++y) {
			for (int x = 0; x < mat.cols() - 1; ++x) {
				sum += mat.ptr(y, x).get();
			}
		}
		
		double mean = sum / totalPix;
		return mean;
	}
	
	private static double computeStdDev(Mat mat, double mean) {
		double sum = 0.0;
		int totalPix = mat.rows() * mat.cols();
		
		for (int y = 0; y < mat.rows() - 1; ++y) {
			for (int x = 0; x < mat.cols() - 1; ++x) {
				int a = mat.ptr(y, x).get();
				sum += (a - mean) * (a - mean);
			}
		}
		
		double stdDev = Math.sqrt(sum / totalPix - 1);
		
		return stdDev;
	}
}
