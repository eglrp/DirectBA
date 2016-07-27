#if !defined( DATASTRUCTURE_H )
#define DATASTRUCTURE_H

#include <cstdlib>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define MAX_STRING_SIZE 512
#define SIFTBINS 128
#define Pi 3.1415926535897932

typedef unsigned char uchar;
struct ImgData
{
	int width, height, nchannels, frameID;
	//Also placeholder for dispairty, which will be converted later.
	float *depth;
	int *depthConf;
	Mat color;
	vector<Mat> imgPyr;
	vector<float *> depthPyr;
	vector<double> scaleFactor;
};
struct SparseFlowData
{
	Point2f *flow;
	int nimages, nfeatures;
};
struct CameraData
{
	CameraData()
	{
		valid = false;
		for (int ii = 0; ii < 9; ii++)
			R[ii] = 0.0, K[ii] = 0.0;
		for (int ii = 0; ii < 6; ii++)
			rt[ii] = 0.0;
		for (int ii = 0; ii < 5; ii++)
			intrinsic[ii] = 0.0;
		for (int ii = 0; ii < 7; ii++)
			distortion[ii] = 0.0;
		hasIntrinsicExtrinisc = 0;
	}

	vector<double> photometric; //a*I+b
	double K[9], invK[9], distortion[7], intrinsic[5], P[12], activeIntrinsic[5], activeK[9], activeinvK[9];
	double RT[12], R[9], invR[9], T[3], rt[6], wt[6], C[3];
	int LensModel, ShutterModel;
	double threshold, ninlierThresh;
	std::string filename;
	int nviews, width, height, hasIntrinsicExtrinisc;
	bool notCalibrated, valid;
};
struct DirectAlignPara
{
	DirectAlignPara(double dataWeight = 0.8, double regIntraWeight = 0.2, double regInterWeight = 0.2, double colorSigma = 5.0, double depthSigma = 2.0, double gradientThresh2 = 100, double reProjectionSigma = 0.5) :
		dataWeight(dataWeight), regIntraWeight(regIntraWeight), regInterWeight(regInterWeight), colorSigma(colorSigma), depthSigma(depthSigma), gradientThresh2(gradientThresh2), reProjectionSigma(reProjectionSigma){}

	double dataWeight, regIntraWeight, regInterWeight;
	double colorSigma, depthSigma, reProjectionSigma; //expected std of variables (grayscale, mm);
	double gradientThresh2;
};
struct Corpus
{
	int nCameras, n3dPoints;
	vector<string> filenames;
	CameraData *camera;

	vector<Point3d>  xyz;
	vector<Point3i >  rgb;
	vector<int> nEleAll3D; //3D -> # visible views
	vector<int*> viewIdAll3D; //3D -> visiable views index
	vector<int*> pointIdAll3D; //3D -> 2D index in those visible views
	vector<Point2f*>  uvAll3D; //3D -> uv of that point in those visible views
	vector<float*> scaleAll3D; //3D -> scale of that point in those visible views
};
struct VideoData
{
	VideoData()
	{
		maxFrameOffset = 0;
	}
	int nVideos, startTime, stopTime, nframesI, maxFrameOffset;
	CameraData *VideoInfo;
};
#endif 
