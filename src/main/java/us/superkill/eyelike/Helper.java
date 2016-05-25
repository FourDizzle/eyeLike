package us.superkill.eyelike;

import static org.bytedeco.javacpp.opencv_core.CV_64F;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.indexer.DoubleIndexer;

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
		DoubleIndexer xIndex = matX.createIndexer();
		DoubleIndexer yIndex = matY.createIndexer();
		DoubleIndexer magsIndex = mags.createIndexer();
		for (int y = 0; y < matX.rows(); ++y) {
			for(int x = 0; x < matX.cols(); ++x) {
				double gX = xIndex.get(y, x);
				double gY = yIndex.get(y, x);
				double magnitude = Math.sqrt((gX * gX) + (gY * gY));
				magsIndex.put(y, x, magnitude);
			}
		}
		imwrite("/Users/ncassiani/Projects/MeanMachine/testimg/matmag.jpg", mags);
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
		DoubleIndexer index = mat.createIndexer();
		for (int y = 0; y < mat.rows() - 1; ++y) {
			for (int x = 0; x < mat.cols() - 1; ++x) {
				sum += index.get(y, x);
			}
		}
		
		double mean = sum / totalPix;
		return mean;
	}
	
	private static double computeStdDev(Mat mat, double mean) {
		double sum = 0.0;
		int totalPix = mat.rows() * mat.cols();
		DoubleIndexer index = mat.createIndexer();
		for (int y = 0; y < mat.rows() - 1; ++y) {
			for (int x = 0; x < mat.cols() - 1; ++x) {
				double a = index.get(y, x);
				sum += (a - mean) * (a - mean);
			}
		}
		
		double stdDev = Math.sqrt(sum / totalPix - 1);
		
		return stdDev;
	}
}
