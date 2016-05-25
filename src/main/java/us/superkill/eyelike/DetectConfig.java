package us.superkill.eyelike;

public class DetectConfig {
	
	// Debugging
	protected final static boolean kPlotVectorField = false;

	// Size constants
	protected final static int kEyePercentTop = 25;
	protected final static int kEyePercentSide = 13;
	protected final static int kEyePercentHeight = 30;
	protected final static int kEyePercentWidth = 35;

	// Preprocessing
	protected final static boolean kSmoothFaceImage = false;
	protected final static float kSmoothFaceFactor = 0.005f;

	// Algorithm Parameters
	protected final static int kFastEyeWidth = 50;
	protected final static int kWeightBlurSize = 5;
	protected final static boolean kEnableWeight = true;
	protected final static float kWeightDivisor = 1.0f;
	protected final static double kGradientThreshold = 50.0;

	// Postprocessing
	protected final static boolean kEnablePostProcess = true;
	protected final static float kPostProcessThreshold = 0.97f;

	// Eye Corner
	protected final static boolean kEnableEyeCorner = false;
}
