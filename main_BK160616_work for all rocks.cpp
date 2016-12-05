#include "DataStructure.h"
#include "Visualization.h"
#include "DataIO.h"
#include "ImgUlti.h"
#include "MathUlti.h"


#include "ceres/ceres.h"
#include "ceres/cost_function_to_functor.h"
#include "ceres/internal/scoped_ptr.h"
#include "glog/logging.h"
#include "ceres/cubic_interpolation.h"

using ceres::Grid2D;
using ceres::Grid1D;
using ceres::CubicInterpolator;
using ceres::BiCubicInterpolator;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using ceres::TukeyLoss;
using ceres::SizedCostFunction;
using ceres::CostFunctionToFunctor;

int MousePosX, MousePosY, clicked;
static void onMouse(int event, int x, int y, int, void*)
{
	if (event == EVENT_LBUTTONDBLCLK)
	{
		MousePosX = x, MousePosY = y, clicked = 1;
		printf("Selected: %d %d\n", x, y);
		cout << "\a";
	}
}

int  InitializeRT(vector<ImgData> &allImages, vector<CameraData> &allCamInfo)
{
	int PryLevel = 3;
	double *K = allCamInfo[0].K;
	int width = allImages[0].width, height = allImages[0].height;
	vector<float> err;
	vector<uchar> status;
	Size winSize(95 * width / 100, 95 * height / 100);// *min(width, height) / 100);
	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);


	Mat RefImg, nonRefImg;
	if (allImages[0].nchannels == 1)
		RefImg = allImages[0].color;
	else
		cvtColor(allImages[0].color, RefImg, CV_RGB2GRAY);

	for (int jj = 0; jj < 6; jj++)
		allCamInfo[0].rt[jj] = 0;

	vector<Mat> RefPyr, nonRefPyr;
	buildOpticalFlowPyramid(RefImg, RefPyr, winSize, PryLevel, true);

	for (int ii = 1; ii < (int)allImages.size(); ii++)
	{
		vector<Point2f>RefPt, nonRefPt;
		RefPt.push_back(Point2f(width / 2, height / 2)); nonRefPt.push_back(Point2f(width / 2, height / 2));;

		if (allImages[0].nchannels == 1)
			nonRefImg = allImages[ii].color;
		else
			cvtColor(allImages[ii].color, nonRefImg, CV_RGB2GRAY);

		buildOpticalFlowPyramid(nonRefImg, nonRefPyr, winSize, PryLevel, true);

		calcOpticalFlowPyrLK(RefPyr, nonRefPyr, RefPt, nonRefPt, status, err, winSize, PryLevel, termcrit);
		if (status[0] == true)
		{
			double ycnR = (RefPt[0].y - K[5]) / K[4], xcnR = (RefPt[0].x - K[1] * ycnR - K[2]) / K[0];
			double ycnNR = (nonRefPt[0].y - K[5]) / K[4], xcnNR = (nonRefPt[0].x - K[1] * ycnNR - K[2]) / K[0];
			double dxcn = xcnNR - xcnR, dycn = ycnNR - ycnR;

			for (int jj = 0; jj < 3; jj++)
				allCamInfo[ii].rt[jj] = 0;
			allCamInfo[ii].rt[3] = dxcn, allCamInfo[ii].rt[4] = dycn, allCamInfo[ii].rt[5] = 0; // using [dTx, dTy, dTz] = Z*(dxcn, dycn, 0). This assumes pure camera translation with no change in Z direction and the depth is as 1.
		}
	}

	return 0;
}
int ExtractDepthFromCorpus(Corpus &CorpusData, CameraData &CamI, int CamID, float *depth)
{
	//char Fname[512];
	//sprintf(Fname, "%s/Corpus/calibInfo.txt", Path);
	//Corpus CorpusData; readCalibInfo(Fname, CorpusData);

	vector<double> allDepth; allDepth.reserve(CorpusData.n3dPoints);
	double rayDir[3], ij[3] = { CamI.K[2], CamI.K[5], 1 };
	getRayDir(rayDir, CamI.invK, CamI.R, ij);

	Point3d CamCenter(CamI.C[0], CamI.C[1], CamI.C[2]), opticalAxis(rayDir[0], rayDir[1], rayDir[2]);

	for (int pid = 0; pid < CorpusData.n3dPoints; pid++)
	{
		Point3d lineofSightDist = CorpusData.xyz[pid] - CamCenter;
		if (lineofSightDist.dot(opticalAxis) > 0)
			allDepth.push_back(sqrt(lineofSightDist.dot(lineofSightDist)));
	}

	sort(allDepth.begin(), allDepth.end());

	int npts = (int)allDepth.size();
	double depthMin = allDepth[npts / 500], depthMax = allDepth[495 * npts / 500];
	depth[0] = depthMin, depth[1] = depthMax;

	return 0;
}
void NviewTriangulation(Point2d *pts, double *P, Point3d *WC, int nview, int npts, double *A, double *B)
{
	int ii, jj, kk;
	bool MenCreated = false;
	if (A == NULL)
	{
		MenCreated = true;
		A = new double[6 * nview];
		B = new double[2 * nview];
	}
	double u, v;

	for (ii = 0; ii < npts; ii++)
	{
		for (jj = 0; jj < nview; jj++)
		{
			u = pts[ii + jj*npts].x, v = pts[ii + jj*npts].y;

			A[6 * jj + 0] = P[12 * jj] - u*P[12 * jj + 8];
			A[6 * jj + 1] = P[12 * jj + 1] - u*P[12 * jj + 9];
			A[6 * jj + 2] = P[12 * jj + 2] - u*P[12 * jj + 10];
			A[6 * jj + 3] = P[12 * jj + 4] - v*P[12 * jj + 8];
			A[6 * jj + 4] = P[12 * jj + 5] - v*P[12 * jj + 9];
			A[6 * jj + 5] = P[12 * jj + 6] - v*P[12 * jj + 10];
			B[2 * jj + 0] = u*P[12 * jj + 11] - P[12 * jj + 3];
			B[2 * jj + 1] = v*P[12 * jj + 11] - P[12 * jj + 7];
		}

		QR_Solution_Double(A, B, 2 * nview, 3);
		WC[ii].x = B[0], WC[ii].y = B[1], WC[ii].z = B[2];
	}

	if (MenCreated)
		delete[]A, delete[]B;

	return;
}

void ProjectandDistort(Point3d WC, Point2d *pts, double *P, double *K, double *distortion, int nviews)
{
	int ii;
	double num1, num2, denum;

	for (ii = 0; ii < nviews; ii++)
	{
		num1 = P[ii * 12 + 0] * WC.x + P[ii * 12 + 1] * WC.y + P[ii * 12 + 2] * WC.z + P[ii * 12 + 3];
		num2 = P[ii * 12 + 4] * WC.x + P[ii * 12 + 5] * WC.y + P[ii * 12 + 6] * WC.z + P[ii * 12 + 7];
		denum = P[ii * 12 + 8] * WC.x + P[ii * 12 + 9] * WC.y + P[ii * 12 + 10] * WC.z + P[ii * 12 + 11];

		pts[ii].x = num1 / denum, pts[ii].y = num2 / denum;
		if (K != NULL)
			LensDistortionPoint(&pts[ii], K + ii * 9, distortion + ii * 7, 1);
	}

	return;
}
double TriangulatePointsFromCorpusCameras(char *Path, int distortionCorrected, int maxPts, double threshold)
{
	printf("Start clicking points for triangulation\n");
	char Fname[200];
	Corpus corpusData;
	corpusData.nCameras = 39;
	corpusData.camera = new CameraData[39];
	//sprintf(Fname, "%s/BA_Camera_AllParams_after.txt", Path);
	//if (ReadCalibInfo(Fname, corpusData))
	//return -1;
	int nviews = corpusData.nCameras;
	for (int ii = 0; ii < nviews; ii++)
	{
		corpusData.camera[ii].threshold = threshold, corpusData.camera[ii].ninlierThresh = 50, corpusData.camera[ii];
		GetrtFromRT(corpusData.camera[ii].rt, corpusData.camera[ii].R, corpusData.camera[ii].T);
		AssembleP(corpusData.camera[ii].K, corpusData.camera[ii].R, corpusData.camera[ii].T, corpusData.camera[ii].P);
		if (distortionCorrected == 1)
			for (int jj = 0; jj < 7; jj++)
				corpusData.camera[ii].distortion[jj] = 0.0;
	}

	sprintf(Fname, "%s/ImageList.txt", Path);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return -1;
	}
	int imgID;
	vector<int> ImageIDList;
	while (fscanf(fp, "%d ", &imgID) != EOF)
		ImageIDList.push_back(imgID);
	fclose(fp);

	int n3D = 1, viewID, ptsCount = 0;
	vector<Point3d> t3D;
	vector<Point2d> uv;
	vector<int> viewIDAll3D;
	vector<Point2d>uvAll3D;
	vector<Point3d>TwoPoints;

	sprintf(Fname, "%s/Points.txt", Path);
	ofstream ofs(Fname);
	if (ofs.fail())
		cerr << "Cannot write " << Fname << endl;
	for (int npts = 0; npts < maxPts; npts++)
	{
		t3D.clear(), uv.clear(), viewIDAll3D.clear(), uvAll3D.clear();
		for (int ii = 0; ii < ImageIDList.size(); ii++)
		{
			cvNamedWindow("Image", CV_WINDOW_NORMAL); setMouseCallback("Image", onMouse);
			sprintf(Fname, "%s/%d.jpg", Path, ImageIDList[ii]);
			if (IsFileExist(Fname) == 0)
				sprintf(Fname, "%s/%d.png", Path, ImageIDList[ii]);
			Mat Img = imread(Fname);
			if (Img.empty())
			{
				printf("Cannot load %s\n", Fname);
				return 1;
			}

			CvPoint text_origin = { Img.cols / 30, Img.cols / 30 };
			sprintf(Fname, "Point %d/%d of Image %d/%d", npts + 1, maxPts, ii + 1, ImageIDList.size());
			if (npts % 2 == 0)
				putText(Img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 1.0 * Img.cols / 640, CV_RGB(255, 0, 0), 2);
			else
				putText(Img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 1.0 * Img.cols / 640, CV_RGB(0, 255, 0), 2);

			imshow("Image", Img), waitKey(0); cout << "\a";
			viewIDAll3D.push_back(ImageIDList[ii]);
			uvAll3D.push_back(Point2d(MousePosX, MousePosY));
			ofs << MousePosX << " " << MousePosY << endl;
		}

		//Test if 3D is correct
		ptsCount = (int)uvAll3D.size();
		Point3d xyz;
		double *A = new double[6 * ptsCount];
		double *B = new double[2 * ptsCount];
		double *tPs = new double[12 * ptsCount];
		bool *passed = new bool[ptsCount];
		double *Ps = new double[12 * ptsCount];
		Point2d *match2Dpts = new Point2d[ptsCount];
		Point2d *reprojectedPt = new Point2d[ptsCount];

		int nviewsi = (int)viewIDAll3D.size();
		for (int ii = 0; ii < nviewsi; ii++)
		{
			viewID = viewIDAll3D.at(ii);

			match2Dpts[ii] = uvAll3D.at(ii);
			LensCorrectionPoint(&match2Dpts[ii], corpusData.camera[viewID].K, corpusData.camera[viewID].distortion, 1);

			if (corpusData.camera[viewID].ShutterModel == 0)
				for (int kk = 0; kk < 12; kk++)
					Ps[12 * ii + kk] = corpusData.camera[viewID].P[kk];
		}

		NviewTriangulation(match2Dpts, Ps, &xyz, nviewsi, 1, NULL, NULL);
		ProjectandDistort(xyz, reprojectedPt, Ps, NULL, NULL, nviewsi);

		double error = 0.0;
		for (int ii = 0; ii < nviewsi; ii++)
			error += pow(reprojectedPt[ii].x - match2Dpts[ii].x, 2) + pow(reprojectedPt[ii].y - match2Dpts[ii].y, 2);
		error = sqrt(error / nviewsi);
		if (passed[0])
		{
			printf("3D: %f %f %f Error: %f\n", xyz.x, xyz.y, xyz.z, error);
			ofs << xyz.x << " " << xyz.y << " " << xyz.z << endl;
			if (maxPts == 2)
				TwoPoints.push_back(xyz);
		}
	}
	ofs.close();

	if (maxPts == 2)
	{
		Point3d dif = TwoPoints[0] - TwoPoints[1];
		return sqrt(dif.dot(dif));
	}
	else
		return 0;
}
double TriangulatePointsFromNonCorpusCameras(char *Path, VideoData &VideoI, vector<int> &SelectedFrames, int distortionCorrected, int maxPts, double threshold)
{
	char Fname[200];

	int nviews = (int)SelectedFrames.size();
	for (int ii = 0; ii < nviews; ii++)
	{
		int fid = SelectedFrames[ii];
		if (distortionCorrected == 1)
			for (int jj = 0; jj < 7; jj++)
				VideoI.VideoInfo[fid].distortion[jj] = 0.0;
	}

	int n3D = 1, viewID, ptsCount = 0;
	vector<Point3d> t3D;
	vector<Point2d> uv;
	vector<int> viewIDAll3D;
	vector<Point2d>uvAll3D;
	vector<Point3d>TwoPoints;

	sprintf(Fname, "%s/Points.txt", Path);
	ofstream ofs(Fname);
	if (ofs.fail())
		cerr << "Cannot write " << Fname << endl;
	for (int npts = 0; npts < maxPts; npts++)
	{
		t3D.clear(), uv.clear(), viewIDAll3D.clear(), uvAll3D.clear();
		for (int ii = 0; ii < (int)SelectedFrames.size(); ii++)
		{
			cvNamedWindow("Image", CV_WINDOW_NORMAL); setMouseCallback("Image", onMouse);
			sprintf(Fname, "%s/0/%d.jpg", Path, SelectedFrames[ii]);
			if (IsFileExist(Fname) == 0)
				sprintf(Fname, "%s/0/%d.png", Path, SelectedFrames[ii]);
			Mat Img = imread(Fname);
			if (Img.empty())
			{
				printf("Cannot load %s\n", Fname);
				return 1;
			}

			CvPoint text_origin = { Img.cols / 30, Img.cols / 30 };
			sprintf(Fname, "Point %d/%d of Image %d/%d", npts + 1, maxPts, ii + 1, SelectedFrames.size());
			if (npts % 2 == 0)
				putText(Img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 1.0 * Img.cols / 640, CV_RGB(255, 0, 0), 2);
			else
				putText(Img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 1.0 * Img.cols / 640, CV_RGB(0, 255, 0), 2);

			imshow("Image", Img), waitKey(0); cout << "\a";
			viewIDAll3D.push_back(SelectedFrames[ii]);
			uvAll3D.push_back(Point2d(MousePosX, MousePosY));
			ofs << MousePosX << " " << MousePosY << endl;
		}


		//Test if 3D is correct
		ptsCount = (int)uvAll3D.size();
		Point3d xyz;
		double *A = new double[6 * ptsCount];
		double *B = new double[2 * ptsCount];
		double *tPs = new double[12 * ptsCount];
		bool *passed = new bool[ptsCount];
		double *Ps = new double[12 * ptsCount];
		Point2d *match2Dpts = new Point2d[ptsCount];
		Point2d *reprojectedPt = new Point2d[ptsCount];

		int nviewsi = (int)viewIDAll3D.size();
		for (int ii = 0; ii < nviewsi; ii++)
		{
			viewID = viewIDAll3D.at(ii);

			match2Dpts[ii] = uvAll3D.at(ii);
			if (distortionCorrected == 1)
				LensCorrectionPoint(&match2Dpts[ii], VideoI.VideoInfo[viewID].K, VideoI.VideoInfo[viewID].distortion, 1);

			if (VideoI.VideoInfo[viewID].ShutterModel == 0)
				for (int kk = 0; kk < 12; kk++)
					Ps[12 * ii + kk] = VideoI.VideoInfo[viewID].P[kk];
		}

		NviewTriangulation(match2Dpts, Ps, &xyz, nviewsi, 1, NULL, NULL);
		ProjectandDistort(xyz, reprojectedPt, Ps, NULL, NULL, nviewsi);

		double error = 0.0;
		for (int ii = 0; ii < nviewsi; ii++)
			error += pow(reprojectedPt[ii].x - match2Dpts[ii].x, 2) + pow(reprojectedPt[ii].y - match2Dpts[ii].y, 2);
		error = sqrt(error / nviewsi);
		if (passed[0])
		{
			printf("3D: %f %f %f Error: %f\n", xyz.x, xyz.y, xyz.z, error);
			ofs << xyz.x << " " << xyz.y << " " << xyz.z << endl;
			if (maxPts == 2)
				TwoPoints.push_back(xyz);

			int id = SelectedFrames[0];
			GetRTFromrt(VideoI.VideoInfo[id].rt, VideoI.VideoInfo[id].R, VideoI.VideoInfo[id].T);
			GetCfromT(VideoI.VideoInfo[id].R, VideoI.VideoInfo[id].T, VideoI.VideoInfo[id].C);

			double rayDir[3], ij[3] = { VideoI.VideoInfo[id].K[2], VideoI.VideoInfo[id].K[5], 1 };
			getRayDir(rayDir, VideoI.VideoInfo[id].invK, VideoI.VideoInfo[id].R, ij);

			Point3d CamCenter(VideoI.VideoInfo[id].C[0], VideoI.VideoInfo[id].C[1], VideoI.VideoInfo[id].C[2]), opticalAxis(rayDir[0], rayDir[1], rayDir[2]);

			Point3d lineofSightDist = xyz - CamCenter;
			if (lineofSightDist.dot(opticalAxis) > 0)
				printf("depth: %.3f\n", sqrt(lineofSightDist.dot(lineofSightDist)));
		}
	}
	ofs.close();

	if (maxPts == 2)
	{
		Point3d dif = TwoPoints[0] - TwoPoints[1];
		return sqrt(dif.dot(dif));
	}
	else
		return 0;

	return 0;
}
double TriangulatePointsFromNonCorpusCameras(char *Path, vector<CameraData> &VideoI, vector<int> &SelectedFrames, int distortionCorrected, int maxPts, double threshold)
{
	char Fname[200];

	int nviews = (int)SelectedFrames.size();
	for (int ii = 0; ii < nviews; ii++)
		if (distortionCorrected == 1)
			for (int jj = 0; jj < 7; jj++)
				VideoI[ii].distortion[jj] = 0.0;

	int n3D = 1, viewID, ptsCount = 0;
	vector<Point3d> t3D;
	vector<Point2d> uv;
	vector<int> viewIDAll3D;
	vector<Point2d>uvAll3D;
	vector<Point3d>TwoPoints;

	for (int npts = 0; npts < maxPts; npts++)
	{
		t3D.clear(), uv.clear(), viewIDAll3D.clear(), uvAll3D.clear();
		for (int ii = 0; ii < (int)SelectedFrames.size(); ii++)
		{
			cvNamedWindow("Image", CV_WINDOW_NORMAL); setMouseCallback("Image", onMouse);
			sprintf(Fname, "%s/0/%d.jpg", Path, VideoI[SelectedFrames[ii]].frameID);
			if (IsFileExist(Fname) == 0)
				sprintf(Fname, "%s/0/%d.png", Path, VideoI[SelectedFrames[ii]].frameID);
			Mat Img = imread(Fname);
			if (Img.empty())
			{
				printf("Cannot load %s\n", Fname);
				return 1;
			}

			CvPoint text_origin = { Img.cols / 30, Img.cols / 30 };
			sprintf(Fname, "Point %d/%d of Image %d/%d", npts + 1, maxPts, ii + 1, SelectedFrames.size());
			if (npts % 2 == 0)
				putText(Img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 1.0 * Img.cols / 640, CV_RGB(255, 0, 0), 2);
			else
				putText(Img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 1.0 * Img.cols / 640, CV_RGB(0, 255, 0), 2);

			imshow("Image", Img), waitKey(0); cout << "\a";
			viewIDAll3D.push_back(SelectedFrames[ii]);
			uvAll3D.push_back(Point2d(MousePosX, MousePosY));
		}


		//Test if 3D is correct
		ptsCount = (int)uvAll3D.size();
		Point3d xyz;
		double *A = new double[6 * ptsCount];
		double *B = new double[2 * ptsCount];
		double *tPs = new double[12 * ptsCount];
		bool *passed = new bool[ptsCount];
		double *Ps = new double[12 * ptsCount];
		Point2d *match2Dpts = new Point2d[ptsCount];
		Point2d *reprojectedPt = new Point2d[ptsCount];

		int nviewsi = (int)viewIDAll3D.size();
		for (int ii = 0; ii < nviewsi; ii++)
		{
			viewID = viewIDAll3D.at(ii);

			match2Dpts[ii] = uvAll3D.at(ii);
			if (distortionCorrected == 1)
				LensCorrectionPoint(&match2Dpts[ii], VideoI[viewID].K, VideoI[viewID].distortion, 1);

			if (VideoI[viewID].ShutterModel == 0)
				for (int kk = 0; kk < 12; kk++)
					Ps[12 * ii + kk] = VideoI[viewID].P[kk];
		}

		NviewTriangulation(match2Dpts, Ps, &xyz, nviewsi, 1, NULL, NULL);
		ProjectandDistort(xyz, reprojectedPt, Ps, NULL, NULL, nviewsi);

		double error = 0.0;
		for (int ii = 0; ii < nviewsi; ii++)
			error += pow(reprojectedPt[ii].x - match2Dpts[ii].x, 2) + pow(reprojectedPt[ii].y - match2Dpts[ii].y, 2);
		error = sqrt(error / nviewsi);
		if (passed[0])
		{
			printf("3D: %f %f %f Error: %f\n", xyz.x, xyz.y, xyz.z, error);
			if (maxPts == 2)
				TwoPoints.push_back(xyz);

			int id = SelectedFrames[0];
			GetRTFromrt(VideoI[id].rt, VideoI[id].R, VideoI[id].T);
			GetCfromT(VideoI[id].R, VideoI[id].T, VideoI[id].C);

			double rayDir[3], ij[3] = { VideoI[id].K[2], VideoI[id].K[5], 1 };
			getRayDir(rayDir, VideoI[id].invK, VideoI[id].R, ij);

			Point3d CamCenter(VideoI[id].C[0], VideoI[id].C[1], VideoI[id].C[2]), opticalAxis(rayDir[0], rayDir[1], rayDir[2]);

			Point3d lineofSightDist = xyz - CamCenter;
			if (lineofSightDist.dot(opticalAxis) > 0)
				printf("depth: %.3f\n", sqrt(lineofSightDist.dot(lineofSightDist)));
		}
	}

	if (maxPts == 2)
	{
		Point3d dif = TwoPoints[0] - TwoPoints[1];
		return sqrt(dif.dot(dif));
	}
	else
		return 0;

	return 0;
}

void ComputeFlowForMicroBA(vector<ImgData> &allImgs, SparseFlowData &sfd)
{
	int maxFeatures = 20000, pyrLevel = 5;
	cv::Size winSize(23, 23);

	int nimages = (int)allImgs.size();
	Mat refImg = allImgs[0].color;
	Mat refGray;
	if (allImgs[0].nchannels == 1)
		refGray = refImg;
	else
		cvtColor(refImg, refGray, CV_RGB2GRAY);

	vector<Point2f> refFeatures;
	goodFeaturesToTrack(refGray, refFeatures, maxFeatures, 1e-10, 5, noArray(), 10);

	int nFeatures = refFeatures.size();
	Mat img = refImg.clone();
	for (int i = 0; i < nFeatures; i++)
		circle(img, refFeatures[i], 3, Scalar(0, 0, 255), 2);

	namedWindow("Features", CV_WINDOW_NORMAL);
	imshow("Features", img); waitKey(1);

	int *inlier_mask = new int[nFeatures];
	Point2f *t_trackedFeatures = new Point2f[nFeatures*nimages];
	for (int i = 0; i < nFeatures; i++)
	{
		inlier_mask[i] = 1;
		t_trackedFeatures[i*nimages] = refFeatures[i];
	}

	// feature tracking (reference <-> non-reference)
	printf("Tracking: ");
	for (int jj = 1; jj < (int)allImgs.size(); jj++)
	{
		printf("%d ..", jj);
		vector<Point2f> corners_fwd, corners_bwd;
		vector<unsigned char> status_fwd, status_bwd;
		vector<float> err_fwd, err_bwd;

		Mat gray;
		if (allImgs[0].nchannels == 1)
			gray = allImgs[jj].color;
		else
			cvtColor(allImgs[jj].color, gray, CV_RGB2GRAY);

		calcOpticalFlowPyrLK(refGray, gray, refFeatures, corners_fwd, status_fwd, err_fwd, winSize, pyrLevel);
		calcOpticalFlowPyrLK(gray, refGray, corners_fwd, corners_bwd, status_bwd, err_bwd, winSize, pyrLevel);

		Mat features_i = Mat(2, nFeatures, CV_32FC1);
		for (int ii = 0; ii<nFeatures; ii++)
		{
			t_trackedFeatures[ii*nimages + jj] = corners_fwd[ii];

			float bidirectional_error = norm(refFeatures[ii] - corners_bwd[ii]);
			if (!(status_fwd[ii] == 0 || status_bwd[ii] == 0 || bidirectional_error>0.1))
				inlier_mask[ii]++;
		}
	}

	int num_inlier = 0;
	for (int i = 0; i < nFeatures; i++)
	{
		if (inlier_mask[i] >= nimages * 10 / 10)
		{
			num_inlier++;
			circle(img, refFeatures[i], 3, Scalar(0, 255, 0), 2);
		}
	}


	imshow("Features", img); waitKey(1);
	printf("\n#Inliers/#Detections: %d / %d\n", num_inlier, nFeatures);

	sfd.nfeatures = num_inlier, sfd.nimages = nimages;
	sfd.flow = new Point2f[nimages*num_inlier];
	int idx = 0;
	for (int ii = 0; ii < nFeatures; ii++)
	{
		if (inlier_mask[ii] >= nimages * 10 / 10)
		{
			for (int jj = 0; jj < nimages; jj++)
				sfd.flow[idx*nimages + jj] = t_trackedFeatures[ii*nimages + jj];
			idx++;
		}
	}
	return;
}
struct FlowBASmall {
	FlowBASmall(double *RefRayDir_, double *RefCC, Point2f &observation, double *intrinsic, double isigma) : RefCC(RefCC), observation(observation), intrinsic(intrinsic), isigma(isigma)
	{
		for (int ii = 0; ii < 3; ii++)
			RefRayDir[ii] = RefRayDir_[ii];//xycn changes for every pixel. Pass by ref does not work
	}

	template <typename T>	bool operator()(const T* const rt, const T* const idepth, T* residuals) 	const
	{
		T XYZ[3], d = idepth[0];
		for (int ii = 0; ii < 3; ii++)
			XYZ[ii] = RefRayDir[ii] / d + RefCC[ii]; //X = r*d+c

		//project to other views
		T tp[3] = { XYZ[0] - rt[2] * XYZ[1] + rt[1] * XYZ[2] + rt[3],
			rt[2] * XYZ[0] + XYZ[1] - rt[0] * XYZ[2] + rt[4],
			-rt[1] * XYZ[0] + rt[0] * XYZ[1] + XYZ[2] + rt[5] };
		T xcn = tp[0] / tp[2], ycn = tp[1] / tp[2];

		T tu = (T)intrinsic[0] * xcn + (T)intrinsic[2] * ycn + (T)intrinsic[3];
		T tv = (T)intrinsic[1] * ycn + (T)intrinsic[4];

		residuals[0] = (T)isigma*((T)observation.x - tu);
		residuals[1] = (T)isigma*((T)observation.y - tv);

		return true;
	}
	static ceres::CostFunction* Create(double *RefRayDir, double *RefCC, Point2f &observation, double *intrinsic, double isigma)
	{
		return (new ceres::AutoDiffCostFunction<FlowBASmall, 2, 6, 1>(new FlowBASmall(RefRayDir, RefCC, observation, intrinsic, isigma)));
	}

	Point2f observation;
	double RefRayDir[3], *RefCC, *intrinsic, isigma;
};
struct FlowBASmall2 {
	FlowBASmall2(double *xycn_, double *R_ref, double *C_Ref, Point2f &observation, double *intrinsic, double isigma) : R_ref(R_ref), C_Ref(C_Ref), observation(observation), intrinsic(intrinsic), isigma(isigma)
	{
		for (int ii = 0; ii < 3; ii++)
			xycn[ii] = xycn_[ii];//xycn changes for every pixel. Pass by ref does not work
	}

	template <typename T>	bool operator()(const T* const rt, const T* const idepth, T* residuals) 	const
	{
		T invd = idepth[0];

		T R_nonref_t[9];
		ceres::AngleAxisToRotationMatrix(rt, R_nonref_t);//this gives R' due to its column major format

		//R_nonref*R_ref
		T R_compose[9] = { T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0) };
		for (int ii = 0; ii < 3; ii++)
			for (int jj = 0; jj < 3; jj++)
				for (int kk = 0; kk < 3; kk++)
					R_compose[ii * 3 + jj] += R_nonref_t[kk * 3 + ii] * R_ref[kk * 3 + jj];

		T RC[3] = { T(0), T(0), T(0) }; //R_nonref*C_ref
		for (int ii = 0; ii < 3; ii++)
			for (int jj = 0; jj < 3; jj++)
				RC[ii] += R_nonref_t[jj * 3 + ii] * C_Ref[jj];

		//project to other views
		T tp[3] = { (T)xycn[0] - rt[2] * (T)xycn[1] + rt[1] + (RC[0] + rt[3]) * invd,
			rt[2] * (T)xycn[0] + (T)xycn[1] - rt[0] + (RC[1] + rt[4]) * invd,
			-rt[1] * (T)xycn[0] + rt[0] * (T)xycn[1] + (T)1 + (RC[2] + rt[5]) * invd };  //(R_nonRef*R_ref*xycn+(R_nonRef*C_Ref+T)*idepth)/idepth ~(R_nonRef*R_ref*xycn+(R_nonRef*C_Ref+T)*idepth)

		//T tp[3] = { (T)xycn[0] - rt[2] * (T)xycn[1] + rt[1] + rt[3] * invd,
		//	rt[2] * (T)xycn[0] + (T)xycn[1] - rt[0] + rt[4] * invd,
		//	-rt[1] * (T)xycn[0] + rt[0] * (T)xycn[1] + (T)1 + rt[5] * invd };  //(R*xycn+T*idepth)/idepth ~ R*xycn+T*idepth
		T xcn = tp[0] / tp[2], ycn = tp[1] / tp[2];

		T tu = (T)intrinsic[0] * xcn + (T)intrinsic[2] * ycn + (T)intrinsic[3];
		T tv = (T)intrinsic[1] * ycn + (T)intrinsic[4];

		residuals[0] = (T)isigma*((T)observation.x - tu);
		residuals[1] = (T)isigma*((T)observation.y - tv);

		return true;
	}
	static ceres::CostFunction* Create(double *xycn, double *R_ref, double *C_Ref, Point2f &observation, double *intrinsic, double isigma)
	{
		return (new ceres::AutoDiffCostFunction<FlowBASmall2, 2, 6, 1>(new FlowBASmall2(xycn, R_ref, C_Ref, observation, intrinsic, isigma)));
	}

	Point2f observation;
	double xycn[3], *R_ref, *C_Ref, *intrinsic, isigma;
};
double FlowBasedBundleAdjustment(char *Path, vector<ImgData> &allImgs, SparseFlowData &sfd, vector<CameraData> &allCalibInfo, double reProjectionSigma, int fixPose, int fixDepth, int SmallAngle = 0)
{
	char Fname[512];
	int nimages = (int)allImgs.size();

	ceres::Problem problem;

	double *invD = new double[sfd.nfeatures];
	double w_min = 0.01, w_max = 1.0;
	for (int ii = 0; ii < sfd.nfeatures; ii++)
		invD[ii] = w_min;// +(w_max - w_min)*double(rand()) / RAND_MAX;

	GetKFromIntrinsic(allCalibInfo[0].K, allCalibInfo[0].intrinsic);
	GetiK(allCalibInfo[0].invK, allCalibInfo[0].K);
	getRfromr(allCalibInfo[0].rt, allCalibInfo[0].R);
	GetCfromT(allCalibInfo[0].R, allCalibInfo[0].rt + 3, allCalibInfo[0].C);

	ceres::LossFunction *RobustLoss = new ceres::HuberLoss(1.0);
	for (int ii = 0; ii < sfd.nfeatures; ii++)
	{
		for (int cid = 1; cid < (int)allImgs.size(); cid++)
		{
			int i = sfd.flow[ii*nimages].x, j = sfd.flow[ii*nimages].y;
			double  ij[3] = { i, j, 1 };

			double rayDirRef[3] = { 0, 0, 0 };
			getRayDir(rayDirRef, allCalibInfo[0].invK, allCalibInfo[0].R, ij);
			ceres::CostFunction* cost_function = FlowBASmall::Create(rayDirRef, allCalibInfo[0].C, sfd.flow[ii*nimages + cid], allCalibInfo[cid].intrinsic, 1.0 / reProjectionSigma);

			//double xycnRef[3] = { 0, 0, 0 };
			//mat_mul(allCalibInfo[0].invK, ij, xycnRef, 3, 3, 1);
			//ceres::CostFunction* cost_function = FlowBASmall2::Create(xycnRef, allCalibInfo[0].R, allCalibInfo[0].C, sfd.flow[ii*nimages + cid], allCalibInfo[cid].intrinsic, 1.0 / reProjectionSigma);

			problem.AddResidualBlock(cost_function, RobustLoss, allCalibInfo[cid].rt, &invD[ii]);
		}
	}

	ceres::Solver::Options options;
	options.num_threads = omp_get_max_threads(); //jacobian eval
	options.num_linear_solver_threads = omp_get_max_threads(); //linear solver
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.linear_solver_type = ceres::ITERATIVE_SCHUR;
	options.preconditioner_type = ceres::JACOBI;
	options.use_inner_iterations = true;
	options.use_nonmonotonic_steps = false;
	options.max_num_iterations = 100;
	options.parameter_tolerance = 1.0e-9;
	options.minimizer_progress_to_stdout = true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << "\n";

	vector<double> ResX, ResY;
	ResX.reserve(sfd.nfeatures*(int)allImgs.size()), ResY.reserve(sfd.nfeatures*(int)allImgs.size());
	for (int ii = 0; ii < sfd.nfeatures; ii++)
	{
		for (int cid = 1; cid < (int)allImgs.size(); cid++)
		{
			int i = sfd.flow[ii*nimages].x, j = sfd.flow[ii*nimages].y;
			double  ij[3] = { i, j, 1 }, xycnRef[3] = { 0, 0, 0 };

			double rayDirRef[3] = { 0, 0, 0 };
			getRayDir(rayDirRef, allCalibInfo[0].invK, allCalibInfo[0].R, ij);
			ceres::CostFunction* cost_function = FlowBASmall::Create(rayDirRef, allCalibInfo[0].C, sfd.flow[ii*nimages + cid], allCalibInfo[cid].intrinsic, 1.0 / reProjectionSigma);

			//mat_mul(allCalibInfo[0].invK, ij, xycnRef, 3, 3, 1);
			//ceres::CostFunction* cost_function = FlowBASmall2::Create(xycnRef, allCalibInfo[0].R, allCalibInfo[0].C, sfd.flow[ii*nimages + cid], allCalibInfo[cid].intrinsic, 1.0 / reProjectionSigma);

			vector<double*> parameter_blocks;
			parameter_blocks.push_back(allCalibInfo[cid].rt), parameter_blocks.push_back(&invD[ii]);

			double resi[2];
			cost_function->Evaluate(&parameter_blocks[0], resi, NULL);
			ResX.push_back(resi[0]), ResY.push_back(resi[1]);
		}
	}
	double meanResX = MeanArray(ResX), meanResY = MeanArray(ResY);
	double varResX = VarianceArray(ResX, meanResX), varResY = VarianceArray(ResY, meanResY);
	printf("Reprojection error after BA:\n Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n", meanResX, meanResY, sqrt(varResX), sqrt(varResY));

	int count = 0;
	for (int i = 0; i < sfd.nfeatures; i++)
		if (invD[i] < 0)
			count++;
	if (count > sfd.nfeatures - count)
		printf("\nFlip detected\n");

	//Saving data
	sprintf(Fname, "%s/SMC.txt", Path); FILE *fp = fopen(Fname, "w+");
	for (int cid = 0; cid < (int)allImgs.size(); cid++)
		fprintf(fp, "%d %.16f %.16f %.16f %.16f %.16f %.16f %\n", allImgs[cid].frameID, allCalibInfo[cid].rt[0], allCalibInfo[cid].rt[1], allCalibInfo[cid].rt[2], allCalibInfo[cid].rt[3], allCalibInfo[cid].rt[4], allCalibInfo[cid].rt[5]);
	fclose(fp);

	if (allImgs[0].nchannels == 1)
		sprintf(Fname, "%s/3d.txt", Path);
	else
		sprintf(Fname, "%s/n3dGL.txt", Path);
	fp = fopen(Fname, "w+");
	for (int ii = 0; ii < sfd.nfeatures; ii++)
	{
		int i = sfd.flow[ii*nimages].x, j = sfd.flow[ii*nimages].y, width = allImgs[0].width;
		double XYZ[3], rayDirRef[3] = { 0, 0, 0 }, ij[3] = { i, j, 1 };
		getRayDir(rayDirRef, allCalibInfo[0].invK, allCalibInfo[0].R, ij);

		for (int kk = 0; kk < 3; kk++)
			XYZ[kk] = rayDirRef[kk] / invD[ii] + allCalibInfo[0].C[kk];
		if (allImgs[0].nchannels == 1)
			fprintf(fp, "%.8e %.8e %.8e\n", XYZ[0], XYZ[1], XYZ[2]);
		else
			fprintf(fp, "%.8e %.8e %.8e %d %d %d\n", XYZ[0], XYZ[1], XYZ[2],
			(int)allImgs[0].color.data[(i + j*width) * 3 + 2], (int)allImgs[0].color.data[(i + j*width) * 3 + 1], (int)allImgs[0].color.data[(i + j*width) * 3]);
	}
	fclose(fp);

	float *depthMap = new float[allImgs[0].width*allImgs[0].height];
	for (int ii = 0; ii < allImgs[0].width*allImgs[0].height; ii++)
		depthMap[ii] = 0.0;
	for (int ii = 0; ii < sfd.nfeatures; ii++)
	{
		int i = sfd.flow[ii*nimages].x, j = sfd.flow[ii*nimages].y, width = allImgs[0].width;
		depthMap[i + j * width] = 1.0 / invD[ii];
	}
	//sprintf(Fname, "%s/D_%d.dat", Path, allImgs[0].frameID), WriteGridBinary(Fname, depthMap, allImgs[0].width, allImgs[0].height, 1);

	sprintf(Fname, "%s/depth.txt", Path);  fp = fopen(Fname, "w+");
	for (int ii = 0; ii < sfd.nfeatures; ii++)
	{
		int i = sfd.flow[ii*nimages].x, j = sfd.flow[ii*nimages].y;
		fprintf(fp, "%d %d %.8f\n", i, j, invD[ii]);
	}
	fclose(fp);

	printf("Done\n");

	delete[]invD, delete[]depthMap;

	return 0;
}

class  BilinearInterpolator
{
public:
	explicit BilinearInterpolator(const uchar *Data, int width, int height, int nchannels, int stride) : Data(Data), width(width), height(height), nchannels(nchannels), stride(stride){}

	// Evaluate the interpolated function value and/or its derivative. Returns false if r or c is out of bounds.
	void Evaluate(double x, double y, double* f, double* dfdr, double* dfdc) const
	{
		int xiD = (int)(x), yiD = (int)(y);
		int xiU = xiD + 1, yiU = yiD + 1;

		int xiDs = xiD *stride, xiUs = xiU *stride, yiDws = yiD*width*stride, yiUws = yiU*width*stride;
		double xxid = x - xiD, yyiD = y - yiD;
		for (int kk = 0; kk < nchannels; kk++)
		{
			double f00 = (double)(int)Data[xiDs + yiDws + kk];
			double f01 = (double)(int)Data[xiUs + yiDws + kk];
			double f10 = (double)(int)Data[xiDs + yiUws + kk];
			double f11 = (double)(int)Data[xiUs + yiUws + kk];
			f[kk] = (f01 - f00)*(x - xiD) + (f10 - f00)*(y - yiD) + (f11 - f01 - f10 + f00)*(x - xiD)*(y - yiD) + f00;

			if (dfdr != NULL)
			{
				dfdr[kk] = (f01 - f00) + (f11 - f01 - f10 + f00)*(y - yiD);
				dfdc[kk] = (f10 - f00) + (f11 - f01 - f10 + f00)*(x - xiD);
			}
		}
	}

	// The following two Evaluate overloads are needed for interfacing with automatic differentiation. The first is for when a scalar evaluation is done, and the second one is for when Jets are used.
	void Evaluate(const double& r, const double& c, double* f) const {
		Evaluate(r, c, f, NULL, NULL);
	}

	template<typename JetT> void Evaluate(const JetT& x, const JetT& y, JetT* f) const
	{
		double frc[3], dfdr[3], dfdc[3];
		Evaluate(x.a, y.a, frc, dfdr, dfdc);
		for (int kk = 0; kk < nchannels; kk++)
		{
			f[kk].a = frc[kk];
			f[kk].v = dfdr[kk] * x.v + dfdc[kk] * y.v;
		}
	}

private:
	int width, height, nchannels, stride;
	const uchar *Data;
};
struct DepthImgWarping_SSD_BiCubic {
	DepthImgWarping_SSD_BiCubic(Point2i uv, double *xycnRef_, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb, int imgID) :
		uv(uv), refImg(refImg), nonRefImgs(nonRefImgs), intrinsic(intrinsic), imgSize(imgSize), nchannels(nchannels), isigma(isigma), hb(hb), imgID(imgID)
	{
		boundary = max(imgSize.x, imgSize.y) / 50;
		for (int ii = 0; ii < 3; ii++)
			xycnRef[ii] = xycnRef_[ii];//xycn changes for every pixel. Pass by ref does not work
	}
	template <typename T>  bool operator()(T const* const* Parameters, T* residuals) const
	{
		//Parametes[0][0]: inverse depth for ref, Parameters[1][0..5]: poses for ref, Parameters[2][0..5]: poses for non-ref, Parameters[3][0..1]: photometric compenstation
		T Rt[9];
		ceres::AngleAxisToRotationMatrix(Parameters[1], Rt);//this gives R' due to its column major format

		T RefRayDir[3] = { (T)0.0, (T) 0.0, (T) 0.0 };
		for (int ii = 0; ii < 3; ii++)
			for (int jj = 0; jj < 3; jj++)
				RefRayDir[ii] += Rt[ii * 3 + jj] * (T)xycnRef[jj]; //ray direction: r = R'*iK*i;

		T C[3] = { (T)0.0, (T) 0.0, (T) 0.0 };
		for (int ii = 0; ii < 3; ii++)
			for (int jj = 0; jj < 3; jj++)
				C[ii] += Rt[ii * 3 + jj] * Parameters[1][jj + 3]; ////-C = R't;

		T XYZ[3], d = Parameters[0][0];
		for (int ii = 0; ii < 3; ii++)
			XYZ[ii] = RefRayDir[ii] / d - C[ii]; //X = r*d+c

		//project to other views
		T tp[3], xcn, ycn, tu, tv, color[3];
		ceres::AngleAxisRotatePoint(Parameters[2], XYZ, tp);
		tp[0] += Parameters[2][3], tp[1] += Parameters[2][4], tp[2] += Parameters[2][5];
		xcn = tp[0] / tp[2], ycn = tp[1] / tp[2];

		tu = (T)intrinsic[0] * xcn + (T)intrinsic[2] * ycn + (T)intrinsic[3];
		tv = (T)intrinsic[1] * ycn + (T)intrinsic[4];

		if (nchannels == 1)
		{
			Grid2D<uchar, 1>  img(nonRefImgs, 0, imgSize.y, 0, imgSize.x);
			BiCubicInterpolator<Grid2D < uchar, 1 > > Imgs(img);

			if (tu<(T)boundary || tu>(T)(imgSize.x - boundary) || tv<(T)boundary || tv>(T)(imgSize.y - boundary))
				for (int ii = 0; ii < (2 * hb + 1)*(2 * hb + 1); ii++)
					residuals[ii] = (T)1000;
			else
			{
				int id = 0;
				for (int jj = -hb; jj <= hb; jj++)
				{
					for (int ii = -hb; ii <= hb; ii++)
					{
						Imgs.Evaluate(tv + (T)jj, tu + (T)ii, color);//ceres takes row, column
						uchar refColor = refImg[uv.x + ii + (uv.y + jj)*imgSize.x];
						//residuals= Parameters[3][0] * color[0] + Parameters[3][1] - (T)(double)(int)refColor;
						residuals[id] = (color[0] - (T)(double)(int)refColor)* (T)isigma;
						id++;
					}
				}
			}
		}
		else
		{
			Grid2D<uchar, 3>  img(nonRefImgs, 0, imgSize.y, 0, imgSize.x);
			BiCubicInterpolator<Grid2D < uchar, 3 > > Imgs(img);

			if (tu<(T)boundary || tu>(T)(imgSize.x - boundary) || tv<(T)boundary || tv>(T)(imgSize.y - boundary))
				for (int ii = 0; ii < nchannels; ii++)
					residuals[ii] = (T)1000;
			else
			{
				int id = 0;
				residuals[0] = (T)0, residuals[1] = (T)0, residuals[2] = (T)0;
				for (int jj = -hb; jj <= hb; jj++)
				{
					for (int ii = -hb; ii <= hb; ii++)
					{
						Imgs.Evaluate(tv + (T)jj, tu + (T)ii, color);//ceres takes row, column
						for (int kk = 0; kk < nchannels; kk++)
						{
							uchar refColor = refImg[(uv.x + ii + (uv.y + jj)*imgSize.x)*nchannels + kk];
							//residuals[kk] += Parameters[3][0] * color[kk] + Parameters[3][1] - (T)(double)(int)refColor;
							residuals[kk] += pow((color[kk] - (T)(double)(int)refColor)* (T)isigma, 2);
						}
					}
				}
				for (int kk = 0; kk < nchannels; kk++)
				{
					if (residuals[kk] < (T)1e-9)
						residuals[kk] = (T)1e-9;
					else
						residuals[kk] = sqrt(residuals[kk]);
				}
			}
		}

		return true;
	}
private:
	uchar *refImg, *nonRefImgs;
	Point2i uv, imgSize;
	double xycnRef[3], *intrinsic, isigma;
	int  nchannels, hb, imgID, boundary;
};
struct DepthImgWarping_SSD_BiLinear {
	DepthImgWarping_SSD_BiLinear(Point2i uv, double *xycnRef_, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb, int imgID) :
		uv(uv), refImg(refImg), nonRefImgs(nonRefImgs), intrinsic(intrinsic), imgSize(imgSize), nchannels(nchannels), isigma(isigma), hb(hb), imgID(imgID)
	{
		boundary = max(imgSize.x, imgSize.y) / 50;
		for (int ii = 0; ii < 3; ii++)
			xycnRef[ii] = xycnRef_[ii];//xycn changes for every pixel. Pass by ref does not work
	}
	template <typename T>  bool operator()(T const* const* Parameters, T* residuals) const
	{
		//Parametes[0][0]: inverse depth for ref, Parameters[1][0..5]: poses for ref, Parameters[2][0..5]: poses for non-ref, Parameters[3][0..1]: photometric compenstation
		T Rt[9];
		ceres::AngleAxisToRotationMatrix(Parameters[1], Rt);//this gives R' due to its column major format

		T RefRayDir[3] = { (T)0.0, (T) 0.0, (T) 0.0 };
		for (int ii = 0; ii < 3; ii++)
			for (int jj = 0; jj < 3; jj++)
				RefRayDir[ii] += Rt[ii * 3 + jj] * (T)xycnRef[jj]; //ray direction: r = R'*iK*i;

		T C[3] = { (T)0.0, (T) 0.0, (T) 0.0 };
		for (int ii = 0; ii < 3; ii++)
			for (int jj = 0; jj < 3; jj++)
				C[ii] += Rt[ii * 3 + jj] * Parameters[1][jj + 3]; ////-C = R't;

		T XYZ[3], d = Parameters[0][0];
		for (int ii = 0; ii < 3; ii++)
			XYZ[ii] = RefRayDir[ii] / d - C[ii]; //X = r*d+c

		//project to other views
		T tp[3], xcn, ycn, tu, tv, color[3];
		ceres::AngleAxisRotatePoint(Parameters[2], XYZ, tp);
		tp[0] += Parameters[2][3], tp[1] += Parameters[2][4], tp[2] += Parameters[2][5];
		xcn = tp[0] / tp[2], ycn = tp[1] / tp[2];

		tu = (T)intrinsic[0] * xcn + (T)intrinsic[2] * ycn + (T)intrinsic[3];
		tv = (T)intrinsic[1] * ycn + (T)intrinsic[4];

		BilinearInterpolator interp(nonRefImgs, imgSize.x, imgSize.y, nchannels, 3);
		if (tu<(T)boundary || tu>(T)(imgSize.x - boundary) || tv<(T)boundary || tv>(T)(imgSize.y - boundary))
			for (int ii = 0; ii < nchannels; ii++)
				residuals[ii] = (T)1000;
		else
		{
			residuals[0] = (T)0, residuals[1] = (T)0, residuals[2] = (T)0;
			for (int jj = -hb; jj <= hb; jj++)
			{
				for (int ii = -hb; ii <= hb; ii++)
				{
					interp.Evaluate(tu + (T)ii, tv + (T)jj, color);
					for (int kk = 0; kk < nchannels; kk++)
					{
						uchar refColor = refImg[(uv.x + ii + (uv.y + jj)*imgSize.x)*nchannels + kk];
						residuals[kk] += pow((color[kk] - (T)(double)(int)refColor)* (T)isigma, 2);
					}
				}
			}
			for (int kk = 0; kk < nchannels; kk++)
			{
				if (residuals[kk] < (T)1e-9)
					residuals[kk] = (T)1e-9;
				else
					residuals[kk] = sqrt(residuals[kk]);
			}
		}

		return true;
	}
private:
	uchar *refImg, *nonRefImgs;
	Point2i uv, imgSize;
	double xycnRef[3], *intrinsic, isigma;
	int  nchannels, hb, imgID, boundary;
};
struct DepthImgWarping_SSD_BiLinear2 {
	DepthImgWarping_SSD_BiLinear2(Point2i uv, double *xycnRef_, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb, int imgID) :
		uv(uv), refImg(refImg), nonRefImgs(nonRefImgs), intrinsic(intrinsic), imgSize(imgSize), nchannels(nchannels), isigma(isigma), hb(hb), imgID(imgID)
	{
		boundary = max(imgSize.x, imgSize.y) / 50;
		for (int ii = 0; ii < 3; ii++)
			xycnRef[ii] = xycnRef_[ii];//xycn changes for every pixel. Pass by ref does not work
	}
	template <typename T>	bool operator()(const T* const idepth, const T* const rt0, const T* const rt1, T* residuals) const
	{
		//Parametes[0][0]: inverse depth for ref, Parameters[1][0..5]: poses for ref, Parameters[2][0..5]: poses for non-ref, Parameters[3][0..1]: photometric compenstation
		T Rt[9];
		ceres::AngleAxisToRotationMatrix(rt0, Rt);//this gives R' due to its column major format

		T RefRayDir[3] = { (T)0.0, (T) 0.0, (T) 0.0 };
		for (int ii = 0; ii < 3; ii++)
			for (int jj = 0; jj < 3; jj++)
				RefRayDir[ii] += Rt[ii * 3 + jj] * (T)xycnRef[jj]; //ray direction: r = R'*iK*i;

		T C[3] = { (T)0.0, (T) 0.0, (T) 0.0 };
		for (int ii = 0; ii < 3; ii++)
			for (int jj = 0; jj < 3; jj++)
				C[ii] += Rt[ii * 3 + jj] * rt0[jj + 3]; ////-C = R't;

		T XYZ[3];
		for (int ii = 0; ii < 3; ii++)
			XYZ[ii] = RefRayDir[ii] / idepth[0] - C[ii]; //X = r*d+c

		//project to other views
		T tp[3], xcn, ycn, tu, tv, color[3];
		ceres::AngleAxisRotatePoint(rt1, XYZ, tp);
		tp[0] += rt1[3], tp[1] += rt1[4], tp[2] += rt1[5];
		xcn = tp[0] / tp[2], ycn = tp[1] / tp[2];

		tu = (T)intrinsic[0] * xcn + (T)intrinsic[2] * ycn + (T)intrinsic[3];
		tv = (T)intrinsic[1] * ycn + (T)intrinsic[4];

		BilinearInterpolator interp(nonRefImgs, imgSize.x, imgSize.y, nchannels, 3);
		if (tu<(T)boundary || tu>(T)(imgSize.x - boundary) || tv<(T)boundary || tv>(T)(imgSize.y - boundary))
			for (int ii = 0; ii < nchannels; ii++)
				residuals[ii] = (T)1000;
		else
		{
			residuals[0] = (T)0, residuals[1] = (T)0, residuals[2] = (T)0;
			for (int jj = -hb; jj <= hb; jj++)
			{
				for (int ii = -hb; ii <= hb; ii++)
				{
					interp.Evaluate(tu + (T)ii, tv + (T)jj, color);
					for (int kk = 0; kk < nchannels; kk++)
					{
						uchar refColor = refImg[(uv.x + ii + (uv.y + jj)*imgSize.x)*nchannels + kk];
						residuals[kk] += pow((color[kk] - (T)(double)(int)refColor)* (T)isigma, 2);
					}
				}
			}
			for (int kk = 0; kk < nchannels; kk++)
			{
				if (residuals[kk] < (T)1e-9)
					residuals[kk] = (T)1e-9;
				else
					residuals[kk] = sqrt(residuals[kk]);
			}
		}

		return true;
	}
	static ceres::CostFunction* Create(Point2i uv, double *xycnRef_, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb, int imgID)
	{
		return (new ceres::AutoDiffCostFunction<DepthImgWarping_SSD_BiLinear2, 3, 1, 6, 6>(new DepthImgWarping_SSD_BiLinear2(uv, xycnRef_, refImg, nonRefImgs, intrinsic, imgSize, nchannels, isigma, hb, imgID)));
	}
private:
	uchar *refImg, *nonRefImgs;
	Point2i uv, imgSize;
	double xycnRef[3], *intrinsic, isigma;
	int  nchannels, hb, imgID, boundary;
};
struct DepthImgWarping_SSD_BiLinear3{
	DepthImgWarping_SSD_BiLinear3(Point2i uv, double *RefRayDir_, double *C, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb, int imgID) :
		uv(uv), refImg(refImg), C(C), nonRefImgs(nonRefImgs), intrinsic(intrinsic), imgSize(imgSize), nchannels(nchannels), isigma(isigma), hb(hb), imgID(imgID)
	{
		boundary = max(imgSize.x, imgSize.y) / 50;
		for (int ii = 0; ii < 3; ii++)
			RefRayDir[ii] = RefRayDir_[ii];//xycn changes for every pixel. Pass by ref does not work
	}
	template <typename T>	bool operator()(const T* const idepth, const T* const rt1, T* residuals) const
	{
		T XYZ[3];
		for (int ii = 0; ii < 3; ii++)
			XYZ[ii] = RefRayDir[ii] / idepth[0] - C[ii]; //X = r*d+c

		//project to other views
		T tp[3], xcn, ycn, tu, tv, color[3];
		ceres::AngleAxisRotatePoint(rt1, XYZ, tp);
		tp[0] += rt1[3], tp[1] += rt1[4], tp[2] += rt1[5];
		xcn = tp[0] / tp[2], ycn = tp[1] / tp[2];

		tu = (T)intrinsic[0] * xcn + (T)intrinsic[2] * ycn + (T)intrinsic[3];
		tv = (T)intrinsic[1] * ycn + (T)intrinsic[4];

		BilinearInterpolator interp(nonRefImgs, imgSize.x, imgSize.y, nchannels, 3);
		if (tu<(T)boundary || tu>(T)(imgSize.x - boundary) || tv<(T)boundary || tv>(T)(imgSize.y - boundary))
			for (int ii = 0; ii < nchannels; ii++)
				residuals[ii] = (T)1000;
		else
		{
			residuals[0] = (T)0, residuals[1] = (T)0, residuals[2] = (T)0;
			for (int jj = -hb; jj <= hb; jj++)
			{
				for (int ii = -hb; ii <= hb; ii++)
				{
					interp.Evaluate(tu + (T)ii, tv + (T)jj, color);
					for (int kk = 0; kk < nchannels; kk++)
					{
						uchar refColor = refImg[(uv.x + ii + (uv.y + jj)*imgSize.x)*nchannels + kk];
						residuals[kk] += pow((color[kk] - (T)(double)(int)refColor)* (T)isigma, 2);
					}
				}
			}
			for (int kk = 0; kk < nchannels; kk++)
			{
				if (residuals[kk] < (T)1e-9)
					residuals[kk] = (T)1e-9;
				else
					residuals[kk] = sqrt(residuals[kk]);
			}
		}

		return true;
	}
	static ceres::CostFunction* Create(Point2i uv, double *RefRayDir, double *C, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb, int imgID)
	{
		return (new ceres::AutoDiffCostFunction<DepthImgWarping_SSD_BiLinear3, 3, 1, 6>(new DepthImgWarping_SSD_BiLinear3(uv, RefRayDir, C, refImg, nonRefImgs, intrinsic, imgSize, nchannels, isigma, hb, imgID)));
	}
private:
	uchar *refImg, *nonRefImgs;
	Point2i uv, imgSize;
	double RefRayDir[3], *C, *intrinsic, isigma;
	int  nchannels, hb, imgID, boundary;
};
/*struct DepthImgWarping_VAR{
DepthImgWarping_VAR(Point2i uv, double *xycnRef_, uchar *refImg, vector<uchar *> &nonRefImgs, vector<double *> &NonRefIntrinisc, Point2i &imgSize, int nchannels, double isigma, int hb) :
uv(uv), refImg(refImg), nonRefImgs(nonRefImgs), NonRefIntrinisc(NonRefIntrinisc), imgSize(imgSize), nchannels(nchannels), isigma(isigma), hb(hb)
{
nNonRef = (int)nonRefImgs.size();
boundary = max(imgSize.x, imgSize.y) / 50;
for (int ii = 0; ii < 3; ii++)
xycnRef[ii] = xycnRef_[ii];//xycn changes for every pixel. Pass by ref does not work
}
template <typename T>  bool operator()(T const* const* Parameters, T* residuals) const
{
//Parametes[0][0]: inverse depth for ref, Parameters[1][0..5]: poses for ref, Parameters[2..N+2][0..5]: poses for non-ref, Parameters[3+N][0..1]: photometric compenstation
T Rt[9];
ceres::AngleAxisToRotationMatrix(Parameters[1], Rt);//this gives R' due to its column major format

T RefRayDir[3] = { (T)0.0, (T) 0.0, (T) 0.0 };
for (int ii = 0; ii < 3; ii++)
for (int jj = 0; jj < 3; jj++)
RefRayDir[ii] += Rt[ii * 3 + jj] * (T)xycnRef[jj]; //ray direction: r = R'*iK*i;

T C[3] = { (T)0.0, (T) 0.0, (T) 0.0 };
for (int ii = 0; ii < 3; ii++)
for (int jj = 0; jj < 3; jj++)
C[ii] += Rt[ii * 3 + jj] * Parameters[1][jj + 3]; ////-C = R't;

T XYZ[3], d = Parameters[0][0];
for (int ii = 0; ii < 3; ii++)
XYZ[ii] = RefRayDir[ii] / d - C[ii]; //X = r*d+c

//project to other views
int patchWidth = 2 * hb + 1, patchSize = patchWidth*patchWidth, patchSizeChannels = patchSize*nchannels;
T *Patches = new T[nchannels * patchSize * (nNonRef + 1)];//max dim
for (int jj = -hb; jj <= hb; jj++)
{
for (int ii = -hb; ii <= hb; ii++)
for (int kk = 0; kk < nchannels; kk++)
Patches[ii + hb + (jj + hb)*patchWidth + kk*patchSize] = (T)(int)refImg[(uv.x + ii + (uv.y + jj)*imgSize.x)*nchannels + kk];
}

for (int ll = 1; ll < nNonRef + 1; ll++)
{
T tp[3], xcn, ycn, tu, tv, color[3];
ceres::AngleAxisRotatePoint(Parameters[1 + ll], XYZ, tp);
tp[0] += Parameters[1 + ll][3], tp[1] += Parameters[1 + ll][4], tp[2] += Parameters[1 + ll][5];
xcn = tp[0] / tp[2], ycn = tp[1] / tp[2];

double *intrinsic = NonRefIntrinisc[ll - 1];
tu = (T)intrinsic[0] * xcn + (T)intrinsic[2] * ycn + (T)intrinsic[3];
tv = (T)intrinsic[1] * ycn + (T)intrinsic[4];

if (nchannels == 1)
{
Grid2D<uchar, 1>  img(nonRefImgs[ll - 1], 0, imgSize.y, 0, imgSize.x);
BiCubicInterpolator<Grid2D < uchar, 1 > > Imgs(img);

if (tu<(T)boundary || tu>(T)(imgSize.x - boundary) || tv<(T)boundary || tv>(T)(imgSize.y - boundary))
{
for (int jj = -hb; jj <= hb; jj++)
for (int ii = -hb; ii <= hb; ii++)
Patches[ii + hb + (jj + hb)*patchWidth + ll*patchSizeChannels] = (T)(255.0*rand() / RAND_MAX);
}
else
{
for (int jj = -hb; jj <= hb; jj++)
{
for (int ii = -hb; ii <= hb; ii++)
{
Imgs.Evaluate(tv + (T)jj, tu + (T)ii, color);//ceres takes row, column
Patches[ii + hb + (jj + hb)*patchWidth + ll*patchSizeChannels] = color[0];
}
}
}
}
else
{
Grid2D<uchar, 3>  img(nonRefImgs[ll - 1], 0, imgSize.y, 0, imgSize.x);
BiCubicInterpolator<Grid2D < uchar, 3 > > Imgs(img);

if (tu<(T)boundary || tu>(T)(imgSize.x - boundary) || tv<(T)boundary || tv>(T)(imgSize.y - boundary))
for (int jj = -hb; jj <= hb; jj++)
for (int ii = -hb; ii <= hb; ii++)
for (int kk = 0; kk < nchannels; kk++)
Patches[ii + hb + (jj + hb)*patchWidth + kk*patchSize + ll*patchSizeChannels] = (T)(255.0*rand() / RAND_MAX);
else
{
for (int jj = -hb; jj <= hb; jj++)
{
for (int ii = -hb; ii <= hb; ii++)
{
Imgs.Evaluate(tv + (T)jj, tu + (T)ii, color);//ceres takes row, column
for (int kk = 0; kk < nchannels; kk++)
Patches[ii + hb + (jj + hb)*patchWidth + kk*patchSize + ll*patchSizeChannels] = color[kk];
}
}
}
}
}

//compute mean
T meanPatch[9 * 9 * 3], varPatch[9*9 * 3];
for (int ii = 0; ii < patchSizeChannels; ii++)
meanPatch[ii] = (T)0, varPatch[ii] = (T)0;

for (int ll = 0; ll < nNonRef + 1; ll++)
{
for (int kk = 0; kk < nchannels; kk++)
for (int jj = -hb; jj <= hb; jj++)
for (int ii = -hb; ii <= hb; ii++)
meanPatch[ii + hb + (jj + hb) *patchWidth + kk*patchSize] += Patches[ii + hb + (jj + hb)*patchWidth + kk*patchSize + ll*patchSizeChannels] / (T)(nNonRef + 1);

}
for (int ll = 0; ll < nNonRef + 1; ll++)
{
for (int kk = 0; kk < nchannels; kk++)
for (int jj = -hb; jj <= hb; jj++)
for (int ii = -hb; ii <= hb; ii++)
varPatch[ii + hb + (jj + hb) *patchWidth + kk*patchSize] += pow(Patches[ii + hb + (jj + hb)*patchWidth + kk*patchSize + ll*patchSizeChannels] - meanPatch[ii + hb + (jj + hb) *patchWidth + kk*patchSize], 2);
}
for (int kk = 0; kk < nchannels; kk++)
{
residuals[kk] = (T)0;
for (int jj = -hb; jj <= hb; jj++)
for (int ii = -hb; ii <= hb; ii++)
residuals[kk] += varPatch[ii + hb + (jj + hb) *patchWidth + kk*patchSize];
residuals[kk] = residuals[kk] / (T)patchSize / (T)nNonRef;
}

delete[]Patches;

return true;
}
private:
uchar *refImg; vector<uchar *> &nonRefImgs;
Point2i uv, imgSize;
double xycnRef[3], isigma; vector<double *> NonRefIntrinisc;
int  nchannels, hb, nNonRef, boundary;
};
double DirectAlignmentPyrVar(char *Path, DirectAlignPara &alignmentParas, vector<ImgData> &allImgs, vector<CameraData> &allCalibInfo, int fixPose, int fixDepth, int pyrID, int SmallAngle = 0, int verbose = 0)
{
char Fname[512];

double dataWeight = alignmentParas.dataWeight, regIntraWeight = alignmentParas.regIntraWeight, regInterWeight = alignmentParas.regInterWeight;
double colorSigma = alignmentParas.colorSigma, depthSigma = alignmentParas.InvDepthSigma; //expected std of variables (grayscale, mm);
double lowDepth = alignmentParas.lowDepth, highDepth = alignmentParas.highDepth;
int nchannels = allImgs[0].nchannels;

//find texture region in the refImg and store in the vector
vector<float *> Grad2All;
//for (int cid = 0; cid < (int)allImgs.size(); cid++)
{
int cid = 0;
int width = allImgs[cid].imgPyr[pyrID].cols, height = allImgs[cid].imgPyr[pyrID].rows, boundary = width / 50;
float *Grad2 = new float[width*height], dx, dy;
for (int jj = boundary; jj < height - boundary; jj++)
{
for (int ii = boundary; ii < width - boundary; ii++)
{
//calculate first order image derivatives: using 1 channel should be enough
dx = (float)(int)allImgs[cid].imgPyr[pyrID].data[(ii + 1)*nchannels + jj*nchannels*width] - (float)(int)allImgs[cid].imgPyr[pyrID].data[(ii - 1)*nchannels + jj*nchannels*width]; //1, 0, -1
dy = (float)(int)allImgs[cid].imgPyr[pyrID].data[ii *nchannels + (jj + 1)*nchannels*width] - (float)(int)allImgs[cid].imgPyr[pyrID].data[ii *nchannels + (jj - 1)*nchannels*width]; //1, 0, -1
Grad2[ii + jj*width] = dx*dx + dy*dy;
}
}
Grad2All.push_back(Grad2);
}

//Getting valid pixels (high texture && has depth &&good conf)
vector<Point2i> segQueue; segQueue.reserve(1e6);
vector<bool> processedQueue; processedQueue.reserve(1e6);
float mag2Thresh = alignmentParas.gradientThresh2; //suprisingly, using more relaxed thresholding better convergence than aggressive one.
vector<bool *> validPixelsAll;
vector<int*> sub2indAll;
vector<int> *indIAll = new vector<int>[allImgs.size()], *indJAll = new vector<int>[allImgs.size()];
vector<float> *GradAll = new vector<float>[allImgs.size()];
vector<double> *invDAll = new vector<double>[allImgs.size()];
//for (int cid = 0; cid < (int)allImgs.size(); cid++)
{
int cid = 0;
int width = allImgs[cid].imgPyr[pyrID].cols, height = allImgs[cid].imgPyr[pyrID].rows, boundary = width / 50;
int *sub2ind = new int[width*height];
bool *validPixels = new bool[width*height];
for (int ii = 0; ii < width*height; ii++)
validPixels[ii] = false, sub2ind[ii] = -1;

invDAll[cid].reserve(width*height), GradAll[cid].reserve(width*height);
indIAll[cid].reserve(width*height), indJAll[cid].reserve(width*height);

for (int jj = boundary; jj < height - boundary; jj++)
{
for (int ii = boundary; ii < width - boundary; ii++)
{
ii = (int)(992 * allImgs[0].scaleFactor[pyrID]), jj = (int)(338 * allImgs[0].scaleFactor[pyrID]);
float mag2 = Grad2All[cid][ii + jj*width], depth = allImgs[cid].InvDepthPyr[pyrID][ii + jj*width];
if (mag2 > mag2Thresh)
indIAll[cid].push_back(ii), indJAll[cid].push_back(jj), invDAll[cid].push_back(1.0 / depth), GradAll[cid].push_back(sqrt(mag2)), sub2ind[ii + jj*width] = (int)indIAll[cid].size() - 1;
break;
}
break;
}

validPixelsAll.push_back(validPixels), sub2indAll.push_back(sub2ind);
}

//detecting nearby pixels for regularization
vector<vector<int> > nNNAll;
vector<vector<int*> > NNAll;
//for (int cid = 0; cid < (int)allImgs.size(); cid++)
{
int cid = 0;
int width = allImgs[cid].imgPyr[pyrID].cols, height = allImgs[cid].imgPyr[pyrID].rows, boundary = width / 50;
vector<int*> NN; NN.reserve(width*height);
vector<int> nNN; nNN.reserve(width*height);
for (int jj = boundary; jj < height - boundary; jj++)
{
for (int ii = boundary; ii < width - boundary; ii++)
{
if (validPixelsAll[cid][ii + jj*width])
{
int count = 0;
int *nn = new int[4];
nn[count] = sub2indAll[cid][ii + jj*width], count++;
if (validPixelsAll[cid][ii - 1 + jj*width])
nn[count] = sub2indAll[cid][ii - 1 + jj*width], count++;
if (validPixelsAll[cid][ii + 1 + jj*width])
nn[count] = sub2indAll[cid][ii + 1 + jj*width], count++;
if (validPixelsAll[cid][ii + (jj - 1)*width])
nn[count] = sub2indAll[cid][ii + (jj - 1)*width], count++;
if (validPixelsAll[cid][ii + (jj + 1)*width])
nn[count] = sub2indAll[cid][ii + (jj + 1)*width], count++;
NN.push_back(nn);
nNN.push_back(count);
}
}
}
nNNAll.push_back(nNN), NNAll.push_back(NN);
}

for (int ii = 0; ii < (int)allImgs.size(); ii++)
{
for (int jj = 0; jj < (int)allImgs.size(); jj++)
allCalibInfo[ii].photometric.push_back(1.0), allCalibInfo[ii].photometric.push_back(0);

GetIntrinsicScaled(allCalibInfo[ii].intrinsic, allCalibInfo[ii].activeIntrinsic, allImgs[ii].scaleFactor[pyrID]);
GetKFromIntrinsic(allCalibInfo[ii].activeK, allCalibInfo[ii].activeIntrinsic);
GetiK(allCalibInfo[ii].activeinvK, allCalibInfo[ii].activeK);
}

//if (pyrID == 5)
{
//Sliding the parameters
int width = allImgs[0].imgPyr[pyrID].cols, height = allImgs[0].imgPyr[pyrID].rows;

Grid2D<uchar, 1>  img1(allImgs[0].imgPyr[pyrID].data, 0, height, 0, width);
BiCubicInterpolator<Grid2D < uchar, 1 > > imgInterp1(img1);

Grid2D<uchar, 3>  img3(allImgs[0].imgPyr[pyrID].data, 0, height, 0, width);
BiCubicInterpolator<Grid2D < uchar, 3 > > imgInterp3(img3);

omp_set_num_threads(omp_get_max_threads());

int hb = (int)min(2.0, 2.0* allImgs[0].scaleFactor[pyrID]), nsteps = 500, count = 0, per = 10;
double step = (1.0 / lowDepth - 1.0 / highDepth) / nsteps;
hb = 4;
double *allCost = new double[nsteps]; allCost[0] = 9e9;
for (int ii = 0; ii < (int)invDAll[0].size(); ii++)
{
#pragma omp critical
{
count++;
if (100.0*count / (int)invDAll[0].size() >= per)
{
printf("%d%%.. ", per);
per += 10;
}
}

vector<uchar *> nonRefImagesPtr;
vector<double *> nonRefIntrinsicPtr;
for (int cid = 1; cid < (int)allImgs.size(); cid++)
nonRefImagesPtr.push_back(allImgs[cid].imgPyr[pyrID].data),
nonRefIntrinsicPtr.push_back(allCalibInfo[cid].activeIntrinsic);

sprintf(Fname, "C:/temp/costD_%d.txt", pyrID);  FILE *fp = fopen(Fname, "w+");
#pragma omp parallel for schedule(dynamic,1)
for (int kk = 0; kk < nsteps; kk++)
{
double residuals = 0;
double invD = step*kk + 1.0 / highDepth;
int i = indIAll[0][ii], j = indJAll[0][ii];
double xycnRef[3] = { 0, 0, 0 }, ij[3] = { i, j, 1 };
mat_mul(allCalibInfo[0].activeinvK, ij, xycnRef, 3, 3, 1);

ceres::DynamicAutoDiffCostFunction<DepthImgWarping_VAR, 4> *cost_function = new ceres::DynamicAutoDiffCostFunction < DepthImgWarping_VAR, 4 >
(new DepthImgWarping_VAR(Point2i(i, j), xycnRef, allImgs[0].imgPyr[pyrID].data, nonRefImagesPtr, nonRefIntrinsicPtr, Point2i(width, height), nchannels, 1.0 / colorSigma, hb));

cost_function->SetNumResiduals(nchannels);

vector<double*> parameter_blocks;
parameter_blocks.push_back(&invD), cost_function->AddParameterBlock(1);
parameter_blocks.push_back(allCalibInfo[0].rt), cost_function->AddParameterBlock(6);
for (int cid = 1; cid < (int)allImgs.size(); cid++)
parameter_blocks.push_back(allCalibInfo[cid].rt), cost_function->AddParameterBlock(6);

double resi[3];
cost_function->Evaluate(&parameter_blocks[0], resi, NULL);
for (int jj = 0; jj < nchannels; jj++)
residuals += resi[jj] * resi[jj];
delete[]cost_function;

allCost[kk] = residuals;
#pragma omp critical
fprintf(fp, "%.4f %.16e\n", invD, residuals);
}
fclose(fp);

double minCost = 9e9, bestInvD = 0;
for (int kk = 0; kk < nsteps; kk++)
{
if (minCost > allCost[kk])
minCost = allCost[kk], bestInvD = step*kk + 1.0 / highDepth;
}
invDAll[0][ii] = bestInvD;
}
printf("100%%\n");
delete[]allCost;

//Saving data
for (int ii = 0; ii < (int)invDAll[0].size(); ii++)
allImgs[0].InvDepthPyr[pyrID][indIAll[0][ii] + indJAll[0][ii] * width] = 1.0 / invDAll[0][ii];
exit(0);
return 0;
}


for (int ii = 0; ii < (int)validPixelsAll.size(); ii++)
delete[] validPixelsAll[ii];
for (int ii = 0; ii < (int)sub2indAll.size(); ii++)
delete[] sub2indAll[ii];
delete[]indIAll, delete[]indJAll, delete[]invDAll, delete[]GradAll;
for (int ii = 0; ii < (int)Grad2All.size(); ii++)
delete[]Grad2All[ii];

return 0.0;
}

struct DepthImgWarping_SSD2 {
DepthImgWarping_SSD2(Point2i uv, double *xycnRef_, uchar *refImg, Point2f *refImgGrad, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigmaC, double isigmaG, int hb, int imgID) :
uv(uv), refImg(refImg), refImgGrad(refImgGrad), nonRefImgs(nonRefImgs), intrinsic(intrinsic), imgSize(imgSize), nchannels(nchannels), isigmaC(isigmaC), isigmaG(isigmaG), hb(hb), imgID(imgID)
{
boundary = max(imgSize.x, imgSize.y) / 50;
for (int ii = 0; ii < 3; ii++)
xycnRef[ii] = xycnRef_[ii];//xycn changes for every pixel. Pass by ref does not work
}
template <typename T>  bool operator()(T const* const* Parameters, T* residuals) const
{
//Parametes[0][0]: inverse depth for ref, Parameters[1][0..5]: poses for ref, Parameters[2][0..5]: poses for non-ref, Parameters[3][0..1]: photometric compenstation
T Rt[9];
ceres::AngleAxisToRotationMatrix(Parameters[1], Rt);//this gives R' due to its column major format

T RefRayDir[3] = { (T)0.0, (T) 0.0, (T) 0.0 };
for (int ii = 0; ii < 3; ii++)
for (int jj = 0; jj < 3; jj++)
RefRayDir[ii] += Rt[ii * 3 + jj] * (T)xycnRef[jj]; //ray direction: r = R'*iK*i;

T C[3] = { (T)0.0, (T) 0.0, (T) 0.0 };
for (int ii = 0; ii < 3; ii++)
for (int jj = 0; jj < 3; jj++)
C[ii] += Rt[ii * 3 + jj] * Parameters[1][jj + 3]; ////-C = R't;

T XYZ[3], d = Parameters[0][0];
for (int ii = 0; ii < 3; ii++)
XYZ[ii] = RefRayDir[ii] / d - C[ii]; //X = r*d+c

//project to other views
T tp[3], xcn, ycn, tu, tv, color[3], dcolorX[3], dcolorY[3];
ceres::AngleAxisRotatePoint(Parameters[2], XYZ, tp);
tp[0] += Parameters[2][3], tp[1] += Parameters[2][4], tp[2] += Parameters[2][5];
xcn = tp[0] / tp[2], ycn = tp[1] / tp[2];

tu = (T)intrinsic[0] * xcn + (T)intrinsic[2] * ycn + (T)intrinsic[3];
tv = (T)intrinsic[1] * ycn + (T)intrinsic[4];

if (nchannels == 1)
{
Grid2D<uchar, 1>  img(nonRefImgs, 0, imgSize.y, 0, imgSize.x);
BiCubicInterpolator<Grid2D < uchar, 1 > > Imgs(img);

if (tu<(T)boundary || tu>(T)(imgSize.x - boundary) || tv<(T)boundary || tv>(T)(imgSize.y - boundary))
residuals[0] = (T)1000, residuals[1] = (T)1000, residuals[2] = (T)1000;
else
{
residuals[0] = (T)0, residuals[1] = (T)0, residuals[2] = (T)0;
for (int jj = -hb; jj <= hb; jj++)
{
for (int ii = -hb; ii <= hb; ii++)
{
Imgs.Evaluate(tv + (T)jj, tu + (T)ii, color, dcolorY, dcolorX);//ceres takes row, column
uchar refColor = refImg[uv.x + ii + (uv.y + jj)*imgSize.x];
residuals[0] += Parameters[3][0] * color[0] + Parameters[3][1] - (T)(double)(int)refColor;
residuals[1] += (T)refImgGrad[uv.x + ii + (uv.y + jj)*imgSize.x].x - dcolorX[0];
residuals[2] += (T)refImgGrad[uv.x + ii + (uv.y + jj)*imgSize.x].y - dcolorY[0];
}
}

residuals[0] = (T)isigmaC*residuals[0], residuals[0] = (T)isigmaG*residuals[0], residuals[0] = (T)isigmaG*residuals[0];
}
}
else
{
Grid2D<uchar, 3>  img(nonRefImgs, 0, imgSize.y, 0, imgSize.x);
BiCubicInterpolator<Grid2D < uchar, 3 > > Imgs(img);

if (tu<(T)boundary || tu>(T)(imgSize.x - boundary) || tv<(T)boundary || tv>(T)(imgSize.y - boundary))
for (int jj = 0; jj < 3 * nchannels; jj++)
residuals[jj] = (T)1000;
else
{
residuals[0] = (T)0, residuals[1] = (T)0, residuals[2] = (T)0;
for (int jj = -hb; jj <= hb; jj++)
{
for (int ii = -hb; ii <= hb; ii++)
{
Imgs.Evaluate(tv + (T)jj, tu + (T)ii, color, dcolorY, dcolorX);//ceres takes row, column
for (int kk = 0; kk < nchannels; kk++)
{
uchar refColor = refImg[(uv.x + ii + (uv.y + jj)*imgSize.x)*nchannels + kk];
residuals[3 * kk] += Parameters[3][0] * color[kk] + Parameters[3][1] - (T)(double)(int)refColor;
residuals[3 * kk + 1] += (T)refImgGrad[(uv.x + ii + (uv.y + jj)*imgSize.x)*nchannels + kk].x - dcolorX[kk];
residuals[3 * kk + 2] += (T)refImgGrad[(uv.x + ii + (uv.y + jj)*imgSize.x)*nchannels + kk].y - dcolorY[kk];
}
}
}
for (int kk = 0; kk < nchannels; kk++)
residuals[3 * kk] = (T)isigmaC*residuals[3 * kk],
residuals[3 * kk + 1] = (T)isigmaG*residuals[3 * kk + 1], residuals[3 * kk + 2] = (T)isigmaG*residuals[3 * kk + 2];
}
}

return true;
}
private:
uchar *refImg, *nonRefImgs;
Point2f *refImgGrad;
Point2i uv, imgSize;
double xycnRef[3], *intrinsic, isigmaC, isigmaG;
int  nchannels, hb, imgID, boundary;
};*/

struct IntraDepthRegularize {
	IntraDepthRegularize(double isigma) : isigma(isigma){}
	template <typename T>	bool operator()(const T* const idepth1, const T* const idepth2, T* residuals) 	const
	{
		T num = idepth2[0] - idepth1[0];
		T denum = idepth1[0] * idepth2[0];
		residuals[0] = (T)isigma*num / denum; //1/id1 - 1/id2 ~ (id2 - id1)/(id1*id2 )
		return true;
	}
	static ceres::CostFunction* Create(const double isigma)
	{
		return (new ceres::AutoDiffCostFunction<IntraDepthRegularize, 1, 1, 1>(new IntraDepthRegularize(isigma)));
	}
	static ceres::CostFunction* CreateNumerDiff(const double isigma)
	{
		return (new ceres::NumericDiffCostFunction<IntraDepthRegularize, ceres::CENTRAL, 1, 1, 1>(new IntraDepthRegularize(isigma)));
	}
	double isigma;
};

/*class NNDepthRegularize
{
public:
NNDepthRegularize(int* sub2indNonRef, double *invdNonRef, int width) : sub2indNonRef(sub2indNonRef), invdNonRef(invdNonRef), width(width){};

// The following two Evaluate overloads are needed for interfacing with automatic differentiation. The first is for when a scalar evaluation is done, and the second one is for when Jets are used.
void Evaluate(const double &x, const double &y, double* f) const
{
int ix = std::floor(x), iy = std::floor(y);
if (x - ix > 0.5)
ix++;
if (y - iy > 0.5)
iy++;

f[0] = invdNonRef[sub2indNonRef[ix + iy*width]];
}
template<typename JetT> void Evaluate(const JetT& x, const JetT &y, JetT* f) const
{
double fx, dfdx;
int ix = std::floor(x), iy = std::floor(y);
if (x - ix > 0.5)
ix++;
if (y - iy > 0.5)
iy++;

f[0].a = invdNonRef[sub2indNonRef[ix + iy*width]];
f[0].v = 0.0;
}

private:
int width;
int*sub2indNonRef;
double *invdNonRef;
};
class NNDepthRegularize : public SizedCostFunction<1, 1, 1>
{
public:
NNDepthRegularize(double x, double y, int* sub2indNonRef, int width) : x(x), y(y), sub2indNonRef(sub2indNonRef), width(width){};

virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
double fx, dfdx;
int ix = std::floor(x), iy = std::floor(y);
if (x - ix > 0.5)
ix++;
if (y - iy > 0.5)
iy++;
int id = sub2indNonRef[ix + iy*width];

double RefInvD = parameters[0][0];
double nonRefInvD = parameters[0][id];

residuals[0] = (nonRefInvD - RefInvD) / (RefInvD*nonRefInvD + 1.0e-16);

if (jacobians != NULL && jacobians[0] != NULL)
{
jacobians[0][0] = -1.0 / RefInvD / RefInvD;
jacobians[0][1] = 1.0 / nonRefInvD / nonRefInvD;
}
return true;
}
int width;
double x, y;
int *sub2indNonRef;
};

struct InterDepthRegularize
{
InterDepthRegularize(const double *rayDir, CameraData *allCalibInfo, int* sub2indNonRef, int RefCid, int nonRefCid, double isigma) : allCalibInfo(allCalibInfo), sub2indNonRef(sub2indNonRef), RefCid(RefCid), nonRefCid(nonRefCid), isigma(isigma)
{
for (int ii = 0; ii < 3; ii++)
rayDirection[ii] = rayDir[ii];
}

template <typename T>	bool operator()(const T* const idepth1, const T* const idepth2, const T* const rt, T* residuals) 	const
{
T XYZ[3], d = idepth1[0];
for (int ii = 0; ii < 3; ii++)
XYZ[ii] = (T)rayDirection[ii] / d + (T)allCalibInfo[RefCid].C[ii];

//project to other views
T tp[3], xcn, ycn, tu, tv, color[3];
ceres::AngleAxisRotatePoint(rt, XYZ, tp);
tp[0] += rt[3], tp[1] += rt[4], tp[2] += rt[5];
xcn = tp[0] / tp[2], ycn = tp[1] / tp[2];

tu = (T)allCalibInfo[nonRefCid].intrinsic[0] * xcn + (T)allCalibInfo[nonRefCid].intrinsic[2] * ycn + (T)allCalibInfo[nonRefCid].intrinsic[3];
tv = (T)allCalibInfo[nonRefCid].intrinsic[1] * ycn + (T)allCalibInfo[nonRefCid].intrinsic[4];

if (tu<(T)50 || tu>(T)(allCalibInfo[nonRefCid].width - 50) || tv<(T)50 || tv>(T)(allCalibInfo[nonRefCid].height - 50))
residuals[0] = (T)1000;
else
{
//compute depth at the nonRefCid
T r[3] = { (T)allCalibInfo[nonRefCid].rt[0], (T)allCalibInfo[nonRefCid].rt[1], (T)allCalibInfo[nonRefCid].rt[2] }, Rt[9];
T xycn1[3] = { xcn, ycn, (T) 0.0 }, nonRefRayDir[3] = { (T)0.0, (T)0.0, (T)0.0 };
ceres::AngleAxisToRotationMatrix(r, Rt);//this gives R' due to its column major format

for (int ii = 0; ii < 3; ii++)
for (int jj = 0; jj < 3; jj++)
nonRefRayDir[ii] += Rt[ii * 3 + jj] * xycn1[jj]; //ray direction: r = R'*iK*i;

T invdRef;
if (nonRefRayDir[0] > nonRefRayDir[1] && nonRefRayDir[0]>nonRefRayDir[2])
invdRef = (XYZ[0] - allCalibInfo[nonRefCid].C[0]) / nonRefRayDir[0]; //depth = (X - cc)/r;
else if (nonRefRayDir[1] > nonRefRayDir[0] && nonRefRayDir[1] > nonRefRayDir[2])
invdRef = (XYZ[1] - allCalibInfo[nonRefCid].C[1]) / nonRefRayDir[1];
else
invdRef = (XYZ[2] - allCalibInfo[nonRefCid].C[2]) / nonRefRayDir[2];

//look up for the depth of the nearest pixel
residuals[0] = (T)isigma*residuals[0];

return true;
}

return true;
}

static ceres::CostFunction* Create(double *rayDir, CameraData *allCalibInfo, int *sub2indNonRef, int RefCid, int nonRefCid, double isigma)
{
return (new ceres::AutoDiffCostFunction<InterDepthRegularize, 1, 1, 1, 6>(new InterDepthRegularize(rayDir, allCalibInfo, sub2indNonRef, RefCid, nonRefCid, isigma)));
}

int*sub2indNonRef;
double rayDirection[3], isigma;
int RefCid, nonRefCid;
CameraData *allCalibInfo;
};*/
struct InterDepthRegularize
{
	InterDepthRegularize(double *xycnRef_, int* sub2indNonRef, double *NonRefintrinsic, int width, int height, double isigma) : CalibInfo(CalibInfo), sub2indNonRef(sub2indNonRef), NonRefintrinsic(NonRefintrinsic), width(width), height(height), isigma(isigma)
	{
		for (int ii = 0; ii < 3; ii++)
			xycnRef[ii] = xycnRef_[ii];
	}
	template <typename T>	bool operator()(const double* const idepth1, const double* const idepth2, const double* const rt0, const double* const rt1, T* residuals) const
	{
		double Rt[9];
		ceres::AngleAxisToRotationMatrix(rt0, Rt);//this gives R' due to its column major format

		double RefRayDir[3] = { 0.0, 0.0, 0.0 };
		for (int ii = 0; ii < 3; ii++)
			for (int jj = 0; jj < 3; jj++)
				RefRayDir[ii] += Rt[ii * 3 + jj] * xycnRef[jj]; //ray direction: r = R'*iK*i;

		double C[3] = { 0, 0, 0 };
		for (int ii = 0; ii < 3; ii++)
			for (int jj = 0; jj < 3; jj++)
				C[ii] += Rt[ii * 3 + jj] * rt0[jj + 3]; ////-C = R't;

		double XYZ[3], d = idepth1[0];
		for (int ii = 0; ii < 3; ii++)
			XYZ[ii] = RefRayDir[ii] / d - C[ii]; //X = r*d+c

		//project to other views
		double tp[3], xcn, ycn, tu, tv;
		ceres::AngleAxisRotatePoint(rt1, XYZ, tp);
		tp[0] += rt1[3], tp[1] += rt1[4], tp[2] += rt1[5];
		xcn = tp[0] / tp[2], ycn = tp[1] / tp[2];

		tu = NonRefintrinsic[0] * xcn + NonRefintrinsic[2] * ycn + NonRefintrinsic[3];
		tv = NonRefintrinsic[1] * ycn + NonRefintrinsic[4];

		if (tu<50 || tu>(width - 50) || tv<50 || tv>(height - 50))
			residuals[0] = 1000;
		else
		{
			//compute depth at the nonRefCid
			double xycnNonRef[3] = { xcn, ycn, 0.0 }, nonRefRayDir[3] = { 0.0, 0.0, 0.0 };
			ceres::AngleAxisToRotationMatrix(rt1, Rt);//this gives R' due to its column major format
			for (int ii = 0; ii < 3; ii++)
				for (int jj = 0; jj < 3; jj++)
					nonRefRayDir[ii] += Rt[ii * 3 + jj] * xycnNonRef[jj]; //ray direction: r = R'*iK*i;

			for (int ii = 0; ii < 3; ii++)
			{
				C[ii] = 0;
				for (int jj = 0; jj < 3; jj++)
					C[ii] += Rt[ii * 3 + jj] * rt1[jj + 3]; ////-C = R't;
			}

			double invdRef;
			if (nonRefRayDir[0] > nonRefRayDir[1] && nonRefRayDir[0] > nonRefRayDir[2])
				invdRef = (XYZ[0] + C[0]) / nonRefRayDir[0]; //depth = (X - cc)/r;
			else if (nonRefRayDir[1] > nonRefRayDir[0] && nonRefRayDir[1] > nonRefRayDir[2])
				invdRef = (XYZ[1] + C[1]) / nonRefRayDir[1];
			else
				invdRef = (XYZ[2] + C[2]) / nonRefRayDir[2];

			//look up for the depth of the nearest pixel
			int rtu = (int)std::floor(tu), rtv = (int)std::floor(tv);
			if (tu - rtu > 0.5)
				rtu++;
			if (tv - rtv > 0.5)
				rtv++;

			double invdNonRef = idepth2[sub2indNonRef[rtu + rtv*width]];
			if (invdNonRef < 0)
				residuals[0] = 1000;
			else
				residuals[0] = isigma*(invdNonRef - invdRef) / (invdRef*invdNonRef + 1.0e-16);
		}

		return true;
	}
	static ceres::CostFunction* Create(double *xycnRef, int *sub2indNonRef, double *NonRefintrinsic, int width, int height, double isigma)
	{
		return (new ceres::NumericDiffCostFunction<InterDepthRegularize, ceres::CENTRAL, 1, 1, 1, 6, 6>(new InterDepthRegularize(xycnRef, sub2indNonRef, NonRefintrinsic, width, height, isigma)));
	}
	int width, height;
	int*sub2indNonRef;
	double *NonRefintrinsic, xycnRef[3], isigma;
	CameraData *CalibInfo;
};

struct DepthImgWarpingSmall {
	DepthImgWarpingSmall(Point2i uv, double *xycnRef_, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb, int imgID) :
		uv(uv), refImg(refImg), nonRefImgs(nonRefImgs), intrinsic(intrinsic), imgSize(imgSize), nchannels(nchannels), isigma(isigma), hb(hb), imgID(imgID)
	{
		boundary = max(imgSize.x, imgSize.y) / 50;
		for (int ii = 0; ii < 3; ii++)
			xycnRef[ii] = xycnRef_[ii];//xycn changes for every pixel. Pass by ref does not work
	}
	template <typename T>  bool operator()(T const* const* Parameters, T* residuals) const
	{
		//Parametes[0][0]: inverse depth for ref, Parameters[1][0..5]: poses for ref, Parameters[2][0..5]: poses for non-ref, Parameters[3][0..1]: photometric compenstation
		T Rt[9];
		ceres::AngleAxisToRotationMatrix(Parameters[1], Rt);//this gives R' due to its column major format

		T RefRayDir[3] = { (T)0.0, (T) 0.0, (T) 0.0 };
		for (int ii = 0; ii < 3; ii++)
			for (int jj = 0; jj < 3; jj++)
				RefRayDir[ii] += Rt[ii * 3 + jj] * (T)xycnRef[jj]; //ray direction: r = R'*iK*i;

		T C[3] = { (T)0.0, (T) 0.0, (T) 0.0 };
		for (int ii = 0; ii < 3; ii++)
			for (int jj = 0; jj < 3; jj++)
				C[ii] += Rt[ii * 3 + jj] * Parameters[1][jj + 3]; ////-C = R't;

		T XYZ[3], d = Parameters[0][0];
		for (int ii = 0; ii < 3; ii++)
			XYZ[ii] = RefRayDir[ii] / d - C[ii]; //X = r*d+c

		//project to other views
		T tp[3], xcn, ycn, tu, tv, color[3];
		//T R_nr[9] = { (T)1, -Parameters[2][2], Parameters[2][1],
		//	Parameters[2][2], (T)1, -Parameters[2][0],
		//	-Parameters[2][1], Parameters[2][0], (T)1 };
		tp[0] = XYZ[0] - Parameters[2][2] * XYZ[1] + Parameters[2][1] * XYZ[2] + Parameters[2][3]; //tp[0] = R_nr[0] * XYZ[0] + R_nr[1] * XYZ[1] + R_nr[2] * XYZ[2] + Parameters[2][3];
		tp[1] = Parameters[2][2] * XYZ[0] + XYZ[1] - Parameters[2][0] * XYZ[2] + Parameters[2][4]; //tp[1] = R_nr[3] * XYZ[0] + R_nr[4] * XYZ[1] + R_nr[5] * XYZ[2] + Parameters[2][4];
		tp[2] = -Parameters[2][1] * XYZ[0] + Parameters[2][0] * XYZ[1] + XYZ[2] + Parameters[2][5]; //tp[2] = R_nr[6] * XYZ[0] + R_nr[7] * XYZ[1] + R_nr[8] * XYZ[2] + Parameters[2][5];
		xcn = tp[0] / tp[2], ycn = tp[1] / tp[2];

		tu = (T)intrinsic[0] * xcn + (T)intrinsic[2] * ycn + (T)intrinsic[3];
		tv = (T)intrinsic[1] * ycn + (T)intrinsic[4];

		BilinearInterpolator interp(nonRefImgs, imgSize.x, imgSize.y, nchannels, 3);
		//Grid2D<uchar, 3>  img(nonRefImgs, 0, imgSize.y, 0, imgSize.x);
		//BiCubicInterpolator<Grid2D < uchar, 3 > > Imgs(img);

		if (tu<(T)boundary || tu>(T)(imgSize.x - boundary) || tv<(T)boundary || tv>(T)(imgSize.y - boundary))
			for (int ii = 0; ii < nchannels; ii++)
				residuals[ii] = (T)1000;
		else
		{
			int id = 0;
			residuals[0] = (T)0, residuals[1] = (T)0, residuals[2] = (T)0;
			for (int jj = -hb; jj <= hb; jj++)
			{
				for (int ii = -hb; ii <= hb; ii++)
				{
					interp.Evaluate(tu + (T)ii, tv + (T)jj, color);
					//Imgs.Evaluate(tv + (T)jj, tu + (T)ii, color);//ceres takes row, column //bicubic
					for (int kk = 0; kk < nchannels; kk++)
					{
						uchar refColor = refImg[(uv.x + ii + (uv.y + jj)*imgSize.x)*nchannels + kk];
						//residuals[kk] += Parameters[3][0] * color[kk] + Parameters[3][1] - (T)(double)(int)refColor;
						residuals[kk] += pow((color[kk] - (T)(double)(int)refColor)* (T)isigma, 2);
					}
				}
			}
			for (int kk = 0; kk < nchannels; kk++)
			{
				if (residuals[kk] < (T)1e-9)
					residuals[kk] = (T)1e-9;
				else
					residuals[kk] = sqrt(residuals[kk]);
			}
		}

		return true;
	}
private:
	uchar *refImg, *nonRefImgs;
	Point2i uv, imgSize;
	double xycnRef[3], *intrinsic, isigma;
	int  nchannels, hb, imgID, boundary;
};
struct DepthImgWarpingSmall2
{
	DepthImgWarpingSmall2(Point2i uv, double *xycnRef_, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb, int imgID) :
		uv(uv), refImg(refImg), nonRefImgs(nonRefImgs), intrinsic(intrinsic), imgSize(imgSize), nchannels(nchannels), isigma(isigma), hb(hb), imgID(imgID)
	{
		boundary = max(imgSize.x, imgSize.y) / 50;
		for (int ii = 0; ii < 3; ii++)
			xycnRef[ii] = xycnRef_[ii];//xycn changes for every pixel. Pass by ref does not work
	}

	template <typename T>	bool operator()(const T* const idepth, const T* const rt0, const T* const rt1, T* residuals) const
	{
		//Parametes[0][0]: inverse depth for ref, Parameters[1][0..5]: poses for ref, Parameters[2][0..5]: poses for non-ref, Parameters[3][0..1]: photometric compenstation
		T Rt[9];
		ceres::AngleAxisToRotationMatrix(rt0, Rt);//this gives R' due to its column major format

		T RefRayDir[3] = { (T)0.0, (T) 0.0, (T) 0.0 };
		for (int ii = 0; ii < 3; ii++)
			for (int jj = 0; jj < 3; jj++)
				RefRayDir[ii] += Rt[ii * 3 + jj] * (T)xycnRef[jj]; //ray direction: r = R'*iK*i;

		T C[3] = { (T)0.0, (T) 0.0, (T) 0.0 };
		for (int ii = 0; ii < 3; ii++)
			for (int jj = 0; jj < 3; jj++)
				C[ii] += Rt[ii * 3 + jj] * rt0[jj + 3]; ////-C = R't;

		T XYZ[3];
		for (int ii = 0; ii < 3; ii++)
			XYZ[ii] = RefRayDir[ii] / idepth[0] - C[ii]; //X = r*d+c

		//project to other views
		T tp[3], xcn, ycn, tu, tv, color[3];
		//T R_nr[9] = { (T)1, -rt1[2], rt1[1],
		//	rt1[2], (T)1, -rt1[0],
		//	-rt1[1], rt1[0], (T)1 };
		tp[0] = XYZ[0] - rt1[2] * XYZ[1] + rt1[1] * XYZ[2] + rt1[3]; //tp[0] = R_nr[0] * XYZ[0] + R_nr[1] * XYZ[1] + R_nr[2] * XYZ[2] + rt1[3];
		tp[1] = rt1[2] * XYZ[0] + XYZ[1] - rt1[0] * XYZ[2] + rt1[4]; //tp[1] = R_nr[3] * XYZ[0] + R_nr[4] * XYZ[1] + R_nr[5] * XYZ[2] + rt1[4];
		tp[2] = -rt1[1] * XYZ[0] + rt1[0] * XYZ[1] + XYZ[2] + rt1[5]; //tp[2] = R_nr[6] * XYZ[0] + R_nr[7] * XYZ[1] + R_nr[8] * XYZ[2] + rt1[5];
		xcn = tp[0] / tp[2], ycn = tp[1] / tp[2];

		tu = (T)intrinsic[0] * xcn + (T)intrinsic[2] * ycn + (T)intrinsic[3];
		tv = (T)intrinsic[1] * ycn + (T)intrinsic[4];

		Grid2D<uchar, 3>  img(nonRefImgs, 0, imgSize.y, 0, imgSize.x);
		BiCubicInterpolator<Grid2D < uchar, 3 > > Imgs(img);

		if (tu<(T)boundary || tu>(T)(imgSize.x - boundary) || tv<(T)boundary || tv>(T)(imgSize.y - boundary))
			for (int jj = 0; jj < nchannels; jj++)
				residuals[jj] = (T)1000;
		else
		{
			residuals[0] = (T)0, residuals[1] = (T)0, residuals[2] = (T)0;
			for (int jj = -hb; jj <= hb; jj++)
			{
				for (int ii = -hb; ii <= hb; ii++)
				{
					Imgs.Evaluate(tv + (T)jj, tu + (T)ii, color);//ceres takes row, column
					for (int kk = 0; kk < nchannels; kk++)
					{
						uchar refColor = refImg[(uv.x + ii + (uv.y + jj)*imgSize.x)*nchannels + kk];
						residuals[kk] += pow((color[kk] - (T)(double)(int)refColor)* (T)isigma, 2);
					}
				}
			}
			for (int kk = 0; kk < nchannels; kk++)
			{
				if (residuals[kk] < (T)1e-9)
					residuals[kk] = (T)1e-9;
				else
					residuals[kk] = sqrt(residuals[kk]);
			}
		}

		return true;
	}
	static ceres::CostFunction* Create(Point2i uv, double *xycnRef, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb, int imgID)
	{
		return (new ceres::AutoDiffCostFunction<DepthImgWarpingSmall2, 3, 1, 6, 6>(new DepthImgWarpingSmall2(uv, xycnRef, refImg, nonRefImgs, intrinsic, imgSize, nchannels, isigma, hb, imgID)));
	}
	uchar *refImg, *nonRefImgs;
	Point2i uv, imgSize;
	double xycnRef[3], *intrinsic, isigma;
	int  nchannels, hb, imgID, boundary;
};
struct DepthImgWarpingSmall3
{
	DepthImgWarpingSmall3(Point2i uv, double *RefRayDir_, double *C, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb, int imgID) :
		uv(uv), C(C), refImg(refImg), nonRefImgs(nonRefImgs), intrinsic(intrinsic), imgSize(imgSize), nchannels(nchannels), isigma(isigma), hb(hb), imgID(imgID)
	{
		boundary = max(imgSize.x, imgSize.y) / 50;
		for (int ii = 0; ii < 3; ii++)
			RefRayDir[ii] = RefRayDir_[ii];//xycn changes for every pixel. Pass by ref does not work
	}

	template <typename T>	bool operator()(const T* const idepth, const T* const rt1, T* residuals) const
	{
		T XYZ[3];
		for (int ii = 0; ii < 3; ii++)
			XYZ[ii] = RefRayDir[ii] / idepth[0] - C[ii]; //X = r*d+c

		//project to other views
		T tp[3], xcn, ycn, tu, tv, color[3];
		//T R_nr[9] = { (T)1, -rt1[2], rt1[1],
		//	rt1[2], (T)1, -rt1[0],
		//	-rt1[1], rt1[0], (T)1 };
		tp[0] = XYZ[0] - rt1[2] * XYZ[1] + rt1[1] * XYZ[2] + rt1[3]; //tp[0] = R_nr[0] * XYZ[0] + R_nr[1] * XYZ[1] + R_nr[2] * XYZ[2] + rt1[3];
		tp[1] = rt1[2] * XYZ[0] + XYZ[1] - rt1[0] * XYZ[2] + rt1[4]; //tp[1] = R_nr[3] * XYZ[0] + R_nr[4] * XYZ[1] + R_nr[5] * XYZ[2] + rt1[4];
		tp[2] = -rt1[1] * XYZ[0] + rt1[0] * XYZ[1] + XYZ[2] + rt1[5]; //tp[2] = R_nr[6] * XYZ[0] + R_nr[7] * XYZ[1] + R_nr[8] * XYZ[2] + rt1[5];
		xcn = tp[0] / tp[2], ycn = tp[1] / tp[2];

		tu = (T)intrinsic[0] * xcn + (T)intrinsic[2] * ycn + (T)intrinsic[3];
		tv = (T)intrinsic[1] * ycn + (T)intrinsic[4];

		BilinearInterpolator interp(nonRefImgs, imgSize.x, imgSize.y, nchannels, 3);
		//Grid2D<uchar, 3>  img(nonRefImgs, 0, imgSize.y, 0, imgSize.x);
		//BiCubicInterpolator<Grid2D < uchar, 3 > > Imgs(img);

		if (tu<(T)boundary || tu>(T)(imgSize.x - boundary) || tv<(T)boundary || tv>(T)(imgSize.y - boundary))
			for (int ii = 0; ii < nchannels; ii++)
				residuals[ii] = (T)1000;
		else
		{
			int id = 0;
			residuals[0] = (T)0, residuals[1] = (T)0, residuals[2] = (T)0;
			for (int jj = -hb; jj <= hb; jj++)
			{
				for (int ii = -hb; ii <= hb; ii++)
				{
					interp.Evaluate(tu + (T)ii, tv + (T)jj, color);
					//Imgs.Evaluate(tv + (T)jj, tu + (T)ii, color);//ceres takes row, column //bicubic
					for (int kk = 0; kk < nchannels; kk++)
					{
						uchar refColor = refImg[(uv.x + ii + (uv.y + jj)*imgSize.x)*nchannels + kk];
						//residuals[kk] += Parameters[3][0] * color[kk] + Parameters[3][1] - (T)(double)(int)refColor;
						residuals[kk] += pow((color[kk] - (T)(double)(int)refColor)* (T)isigma, 2);
					}
				}
			}
			for (int kk = 0; kk < nchannels; kk++)
			{
				if (residuals[kk] < (T)1e-9)
					residuals[kk] = (T)1e-9;
				else
					residuals[kk] = sqrt(residuals[kk]);
			}
		}

		return true;
	}
	static ceres::CostFunction* Create(Point2i uv, double *RefRayDir, double *C, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb, int imgID)
	{
		return (new ceres::AutoDiffCostFunction<DepthImgWarpingSmall3, 3, 1, 6>(new DepthImgWarpingSmall3(uv, RefRayDir, C, refImg, nonRefImgs, intrinsic, imgSize, nchannels, isigma, hb, imgID)));
	}
	uchar *refImg, *nonRefImgs;
	Point2i uv, imgSize;
	double RefRayDir[3], *C, *intrinsic, isigma;
	int  nchannels, hb, imgID, boundary;
};
double DirectAlignmentPyr(char *Path, DirectAlignPara &alignmentParas, vector<ImgData> &allImgs, vector<CameraData> &allCalibInfo, int fixPose, int fixDepth, int pyrID, int SmallAngle = 0, int WriteOutput = 0, int verbose = 0)
{
	char Fname[512];

	double dataWeight = alignmentParas.dataWeight, regIntraWeight = alignmentParas.regIntraWeight, regInterWeight = alignmentParas.regInterWeight;
	double colorSigma = alignmentParas.colorSigma, depthSigma = alignmentParas.depthSigma; //expected std of variables (grayscale, mm);
	double lowDepth = alignmentParas.lowDepth, highDepth = alignmentParas.highDepth;
	int nchannels = allImgs[0].nchannels;

	for (int ii = 0; ii < (int)allImgs.size(); ii++)
	{
		for (int jj = 0; jj < (int)allImgs.size(); jj++)
			allCalibInfo[ii].photometric.push_back(1.0), allCalibInfo[ii].photometric.push_back(0);

		GetIntrinsicScaled(allCalibInfo[ii].intrinsic, allCalibInfo[ii].activeIntrinsic, allImgs[ii].scaleFactor[pyrID]);
		GetKFromIntrinsic(allCalibInfo[ii].activeK, allCalibInfo[ii].activeIntrinsic);
		GetiK(allCalibInfo[ii].activeinvK, allCalibInfo[ii].activeK);
	}

	ceres::Problem problem;

	//Data term
	for (int cidJ = 0; cidJ < 1; cidJ++) //for all refImages
	{
		int hb = (int)(0.003*allImgs[cidJ].imgPyr[pyrID].cols);
		printf("Using window of %dx%d\n", 2 * hb + 1, 2 * hb + 1);
		for (int cidI = 0; cidI < (int)allImgs.size(); cidI++)
		{
			if (cidI == cidJ)
				continue;
			int width = allImgs[cidI].imgPyr[pyrID].cols, height = allImgs[cidI].imgPyr[pyrID].rows, boundary = width / 50;
			for (int jj = boundary; jj < height - boundary; jj++)
			{
				for (int ii = boundary; ii < width - boundary; ii++)
				{
					if (allImgs[cidJ].validPixelsPyr[pyrID][ii + jj*width] == 0)
						continue;

					float gradMag = allImgs[cidJ].gradPyr[pyrID][ii + jj*width];
					double xycnRef[3] = { 0, 0, 0 }, ij[3] = { ii, jj, 1 }, rayDirRef[3];
					mat_mul(allCalibInfo[cidJ].activeinvK, ij, xycnRef, 3, 3, 1);
					getRayDir(rayDirRef, allCalibInfo[cidJ].activeinvK, allCalibInfo[cidJ].R, ij);

					ceres::LossFunction *ColorLoss = new ceres::HuberLoss(alignmentParas.HuberSizeColor);
					ceres::LossFunction *ScaleColorLoss = new ceres::ScaledLoss(ColorLoss, dataWeight*gradMag, ceres::TAKE_OWNERSHIP);

					double v = allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*width];
					if (SmallAngle == 1)
					{
						/*ceres::DynamicAutoDiffCostFunction<DepthImgWarpingSmall, 4> *cost_function = new ceres::DynamicAutoDiffCostFunction < DepthImgWarpingSmall, 4 >
							(new DepthImgWarpingSmall(Point2i(ii, jj), xycnRef, allImgs[cidJ].imgPyr[pyrID].data, allImgs[cidI].imgPyr[pyrID].data, allCalibInfo[cidI].activeIntrinsic, Point2i(width, height), allImgs[cidI].nchannels, 1.0 / colorSigma, hb, cidI));

							cost_function->SetNumResiduals(nchannels);

							vector<double*> parameter_blocks;
							parameter_blocks.push_back(&allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*width]), cost_function->AddParameterBlock(1);
							parameter_blocks.push_back(allCalibInfo[cidJ].rt), cost_function->AddParameterBlock(6);
							parameter_blocks.push_back(allCalibInfo[cidI].rt), cost_function->AddParameterBlock(6);
							//parameter_blocks.push_back(&allCalibInfo[cidI].photometric[2 * cidJ]), cost_function->AddParameterBlock(2);

							problem.AddResidualBlock(cost_function, ScaleColorLoss, parameter_blocks);*/
						//ceres::CostFunction* cost_function = DepthImgWarpingSmall2::Create(Point2i(ii, jj), xycnRef, allImgs[cidJ].imgPyr[pyrID].data, allImgs[cidI].imgPyr[pyrID].data, allCalibInfo[cidI].activeIntrinsic, Point2i(width, height), allImgs[cidI].nchannels, 1.0 / colorSigma, hb, cidI);
						ceres::CostFunction* cost_function = DepthImgWarpingSmall3::Create(Point2i(ii, jj), rayDirRef, allCalibInfo[cidJ].C, allImgs[cidJ].imgPyr[pyrID].data, allImgs[cidI].imgPyr[pyrID].data, allCalibInfo[cidI].activeIntrinsic, Point2i(width, height), allImgs[cidI].nchannels, 1.0 / colorSigma, hb, cidI);
						problem.AddResidualBlock(cost_function, ScaleColorLoss, &allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*width], &allCalibInfo[cidI].rt[0]);
					}
					else
					{
						/*ceres::DynamicAutoDiffCostFunction<DepthImgWarping_SSD_BiLinear, 4> *cost_function = new ceres::DynamicAutoDiffCostFunction < DepthImgWarping_SSD2, 4 >
							(new DepthImgWarping_SSD_BiLinear(Point2i(ii, jj), xycnRef, allImgs[cidJ].imgPyr[pyrID].data, allImgs[cidI].imgPyr[pyrID].data, allCalibInfo[cidI].activeIntrinsic, Point2i(width, height), allImgs[cidI].nchannels, 1.0 / colorSigma, hb, cidI));

							cost_function->SetNumResiduals(nchannels);

							vector<double*> parameter_blocks;
							parameter_blocks.push_back(&allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*width]), cost_function->AddParameterBlock(1);
							parameter_blocks.push_back(allCalibInfo[cidJ].rt), cost_function->AddParameterBlock(6);
							parameter_blocks.push_back(allCalibInfo[cidI].rt), cost_function->AddParameterBlock(6);
							//parameter_blocks.push_back(&allCalibInfo[cidI].photometric[2 * cidJ]), cost_function->AddParameterBlock(2);
							problem.AddResidualBlock(cost_function, ScaleColorLoss, parameter_blocks);*/

						//ceres::CostFunction* cost_function = DepthImgWarping_SSD_BiLinear2::Create(Point2i(ii, jj), xycnRef, allImgs[cidJ].imgPyr[pyrID].data, allImgs[cidI].imgPyr[pyrID].data, allCalibInfo[cidI].activeIntrinsic, Point2i(width, height), allImgs[cidI].nchannels, 1.0 / colorSigma, hb, cidI);
						//problem.AddResidualBlock(cost_function, ScaleColorLoss, &allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*width], &allCalibInfo[cidJ].rt[0], &allCalibInfo[cidI].rt[0]);
						ceres::CostFunction* cost_function = DepthImgWarping_SSD_BiLinear3::Create(Point2i(ii, jj), rayDirRef, allCalibInfo[cidJ].C, allImgs[cidJ].imgPyr[pyrID].data, allImgs[cidI].imgPyr[pyrID].data, allCalibInfo[cidI].activeIntrinsic, Point2i(width, height), allImgs[cidI].nchannels, 1.0 / colorSigma, hb, cidI);
						problem.AddResidualBlock(cost_function, ScaleColorLoss, &allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*width], &allCalibInfo[cidI].rt[0]);
					}

					if (fixDepth == 1)
						problem.SetParameterBlockConstant(&allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*width]); //depth
					//if (cidJ == 0)
					//	problem.SetParameterBlockConstant(allCalibInfo[cidJ].rt); //pose Ref
					if (fixPose == 1)
						problem.SetParameterBlockConstant(allCalibInfo[cidI].rt); //pose nonRef
					//problem.SetParameterBlockConstant(&allCalibInfo[cidI].photometric[2 * cidJ]); //photometric

					problem.SetParameterLowerBound(&allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*width], 0, 1e-16); //positive depth
					problem.SetParameterUpperBound(&allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*width], 0, 3); //positive depth
				}
			}
		}
	}

	//Intra depth regularization term
	/*if (fixDepth == 0)
	{
	for (int cid = 0; cid < 1; cid++)
	{
	double idepthSigma = 1.0 / depthSigma;
	int width = allImgs[cid].imgPyr[pyrID].cols, height = allImgs[cid].imgPyr[pyrID].rows, boundary = width / 50;

	int GeoWeightBW = min(3, boundary * 2 + 1);
	float *localWeights = new float[GeoWeightBW*GeoWeightBW];
	vector<Point2i> indexList;
	for (int range = 1; range <= (GeoWeightBW - 1) / 2; range++)
	{
	for (int r = -range; r <= range; r++)
	for (int c = -range; c <= range; c++)
	if (abs(r) >= range || abs(c) >= range)
	indexList.push_back(Point2i(r, c));
	}


	for (int jj = boundary; jj < height - boundary; jj++)
	{
	for (int ii = boundary; ii < width - boundary; ii++)
	{
	if (allImgs[cid].validPixelsPyr[pyrID][ii + jj*width] == 0)
	continue;
	//int hbw = (GeoWeightBW - 1) / 2;
	//ComputeGeodesicWeight(allImgs[cid].imgPyr[pyrID].data, allImgs[cid].validPixelsPyr[pyrID], width, nchannels, localWeights, ii, jj, hbw, 3.0 * 6, 3);

	for (auto ij : indexList)
	{
	if (ii + ij.x<boundary || ii + ij.x> width - boundary || jj + ij.y < boundary || jj + ij.y > height - boundary)
	continue;
	if (allImgs[cid].validPixelsPyr[pyrID][ii + ij.x + (jj + ij.y)*width])
	{
	double colorDif = 0;
	for (int kk = 0; kk < nchannels; kk++)
	colorDif += (double)((int)allImgs[cid].imgPyr[pyrID].data[kk + (ii + jj* width)*nchannels] - (int)allImgs[cid].imgPyr[pyrID].data[kk + (ii - 1 + jj* width)*nchannels]);
	colorDif /= nchannels;
	double edgePreservingWeight = std::exp(-pow(colorDif / alignmentParas.colorSigma, 2));
	//double edgePreservingWeight = localWeights[ij.x + hbw + (ij.y + hbw)*GeoWeightBW];

	ceres::LossFunction *RegIntraLoss = new ceres::HuberLoss(3);
	ceres::LossFunction *ScaleRegIntraLoss = new ceres::ScaledLoss(RegIntraLoss, regIntraWeight*edgePreservingWeight, ceres::TAKE_OWNERSHIP);

	ceres::CostFunction* cost_function = IntraDepthRegularize::Create(idepthSigma);
	problem.AddResidualBlock(cost_function, ScaleRegIntraLoss, &allImgs[cid].InvDepthPyr[pyrID][ii + jj*width], &allImgs[cid].InvDepthPyr[pyrID][ii + ij.x + (jj + ij.y)*width]);
	}
	}
	}
	}
	delete[]localWeights;
	}
	}*/

	/*//Inter depth regularization term
	ceres::LossFunction *RegInterLoss = NULL;
	ceres::LossFunction* ScaleRegInterLoss = new ceres::ScaledLoss(RegInterLoss, regInterWeight, ceres::TAKE_OWNERSHIP);
	for (int cidJ = 0; cidJ < (int)allImgs.size(); cidJ++)
	{
	for (int cidI = 0; cidI < (int)allImgs.size(); cidI++)
	{
	if (cidI == cidJ)
	continue;

	for (int ii = 0; ii < (int)invDAll[cidJ].size(); ii++)
	{
	double xycnRef[3] = { 0, 0, 0 }, ij[3] = { indIAll[cidJ][ii], indJAll[cidJ][ii], 1 };
	mat_mul(allCalibInfo[cidJ].invK, ij, xycnRef, 3, 3, 1);

	ceres::CostFunction* cost_function = InterDepthRegularize::Create(xycnRef, sub2indAll[cidI], allCalibInfo[cidI].intrinsic, allCalibInfo[cidI].width, allCalibInfo[cidI].height, 0.5 / depthSigma);
	problem.AddResidualBlock(cost_function, ScaleRegInterLoss, &invDAll[cidJ][0], &invDAll[cidI][0], allCalibInfo[cidJ].rt, allCalibInfo[cidI].rt);
	}
	}
	}*/


	//Set up callback to update residual images
	int iter = 0;
	class MyCallBack : public ceres::IterationCallback{
	public:
		MyCallBack(std::function<void()> callback, int &iter, int &myVerbose) : callback_(callback), iter(iter), myVerbose(myVerbose){}
		virtual ~MyCallBack() {}

		ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary)
		{
			iter = summary.iteration;
			if (myVerbose == 1)
			{
				if (iter == 0 || summary.step_is_successful)
					callback_();
			}
			else
				printf("(%d: %.3e).. ", iter, summary.cost);

			return ceres::SOLVER_CONTINUE;
		}
		int &iter, &myVerbose;
		std::function<void()> callback_;
	};
	auto update_Result = [&]()
	{
		char Fname[512];
		sprintf(Fname, "%s/Level_%d", Path, pyrID); makeDir(Fname);

		for (int cidJ = 0; cidJ < 1; cidJ++)
		{
			int widthJ = allImgs[cidJ].imgPyr[pyrID].cols, heightJ = allImgs[cidJ].imgPyr[pyrID].rows, lengthJ = widthJ*heightJ, npts = 0, boundaryJ = widthJ / 50;
			double residuals = 0.0;

			uchar *resImg = new uchar[widthJ*heightJ* nchannels];
			uchar *synthImg = new uchar[widthJ*heightJ* nchannels];
			for (int cidI = 0; cidI < (int)allImgs.size(); cidI++)
			{
				if (cidI == cidJ)
					continue;

				for (int ii = 0; ii < widthJ*heightJ* nchannels; ii++)
					resImg[ii] = (uchar)128, synthImg[ii] = (uchar)128;

				int widthI = allImgs[cidI].imgPyr[pyrID].cols, heightI = allImgs[cidI].imgPyr[pyrID].rows;
				for (int jj = boundaryJ; jj < heightJ - boundaryJ; jj++)
				{
					for (int ii = boundaryJ; ii < widthJ - boundaryJ; ii++)
					{
						if (allImgs[cidJ].validPixelsPyr[pyrID][ii + jj*widthJ] == 0)
							continue;

						//back-project ref depth to 3D
						double XYZ[3], rayDirRef[3], ij[3] = { ii, jj, 1 };
						double d = allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*widthJ];

						getRfromr(allCalibInfo[cidJ].rt, allCalibInfo[cidJ].R);
						getRayDir(rayDirRef, allCalibInfo[cidJ].activeinvK, allCalibInfo[cidJ].R, ij);
						GetCfromT(allCalibInfo[cidJ].R, allCalibInfo[cidJ].rt + 3, allCalibInfo[cidJ].C);
						for (int kk = 0; kk < 3; kk++)
							XYZ[kk] = rayDirRef[kk] / d + allCalibInfo[cidJ].C[kk];

						//project to other views
						double tp[3], xcn, ycn, tu, tv, color[3];
						if (SmallAngle == 1)
						{
							//T R_nr[9] = { (T)1, -Parameters[2][2], Parameters[2][1],
							//	Parameters[2][2], (T)1, -Parameters[2][0],
							//	-Parameters[2][1], Parameters[2][0], (T)1 };
							tp[0] = XYZ[0] - allCalibInfo[cidI].rt[2] * XYZ[1] + allCalibInfo[cidI].rt[1] * XYZ[2];
							tp[1] = allCalibInfo[cidI].rt[2] * XYZ[0] + XYZ[1] - allCalibInfo[cidI].rt[0] * XYZ[2];
							tp[2] = -allCalibInfo[cidI].rt[1] * XYZ[0] + allCalibInfo[cidI].rt[0] * XYZ[1] + XYZ[2];
						}
						else
							ceres::AngleAxisRotatePoint(allCalibInfo[cidI].rt, XYZ, tp);
						tp[0] += allCalibInfo[cidI].rt[3], tp[1] += allCalibInfo[cidI].rt[4], tp[2] += allCalibInfo[cidI].rt[5];
						xcn = tp[0] / tp[2], ycn = tp[1] / tp[2];

						tu = allCalibInfo[cidI].activeIntrinsic[0] * xcn + allCalibInfo[cidI].activeIntrinsic[2] * ycn + allCalibInfo[cidI].activeIntrinsic[3];
						tv = allCalibInfo[cidI].activeIntrinsic[1] * ycn + allCalibInfo[cidI].activeIntrinsic[4];

						if (tu<1 || tu> widthI - 2 || tv<1 || tv>heightI - 2)
							continue;

						BilinearInterpolator interp(allImgs[cidI].imgPyr[pyrID].data, widthI, heightI, nchannels, nchannels);
						if (nchannels == 1)
						{
							//Grid2D<uchar, 1>  img(allImgs[cidI].imgPyr[pyrID].data, 0, heightI, 0, widthI);
							//BiCubicInterpolator<Grid2D < uchar, 1 > > imgInterp(img);
							//imgInterp.Evaluate(tv, tu, color);//ceres takes row, column

							interp.Evaluate(tu, tv, color);
							double fcolor = allCalibInfo[cidI].photometric[2 * cidJ] * color[0] + allCalibInfo[cidI].photometric[2 * cidJ + 1];
							double dif = fcolor - (double)(int)allImgs[cidJ].imgPyr[pyrID].data[ii + jj * widthJ];

							resImg[ii + jj * widthJ] = (uchar)(int)(max(min(128.0 + 5.0*dif, 255.0), 0.0)); //magnify 5x
							resImg[ii + jj * widthJ] = (uchar)(int)(fcolor + 0.5);

							residuals += dif*dif;
							npts++;
						}
						else
						{
							//Grid2D<uchar, 3>  img(allImgs[cidI].imgPyr[pyrID].data, 0, heightI, 0, widthI);
							//BiCubicInterpolator<Grid2D < uchar, 3 > > imgInterp(img);
							//imgInterp.Evaluate(tv, tu, color);//ceres takes row, column

							interp.Evaluate(tu, tv, color);

							for (int kk = 0; kk < 3; kk++)
							{
								double fcolor = allCalibInfo[cidI].photometric[2 * cidJ] * color[kk] + allCalibInfo[cidI].photometric[2 * cidJ + 1];
								uchar refColor = allImgs[cidJ].imgPyr[pyrID].data[kk + (ii + jj* widthJ)*nchannels];
								double dif = fcolor - (double)(int)refColor;

								resImg[ii + jj * widthJ + kk*lengthJ] = (uchar)(int)(max(min(128.0 + 5.0*dif, 255.0), 0.0)); //magnify 5x
								synthImg[ii + jj * widthJ + kk*lengthJ] = (uchar)(int)(fcolor + 0.5);
								residuals += dif*dif;
							}
							npts += 3;
						}
					}
				}
				if (fixDepth == 1)
				{
					sprintf(Fname, "%s/Level_%d/D_R_%03d_%03d_%03d.png", Path, pyrID, cidJ, cidI, iter), WriteGridToImage(Fname, resImg, widthJ, heightJ, nchannels);
					sprintf(Fname, "%s/Level_%d/D_S_%03d_%03d_%03d.png", Path, pyrID, cidJ, cidI, iter), WriteGridToImage(Fname, synthImg, widthJ, heightJ, nchannels);
					sprintf(Fname, "%s/pose_%d.txt", Path, cidJ); FILE *fp = fopen(Fname, "a+");
					fprintf(fp, "Iter %d PyrID D D: %d Cid: %d %.16e %.16e %.16e %.16e %.16e %.16e %.4f %.4f \n", iter, pyrID, cidI, allCalibInfo[cidI].rt[0], allCalibInfo[cidI].rt[1], allCalibInfo[cidI].rt[2], allCalibInfo[cidI].rt[3], allCalibInfo[cidI].rt[4], allCalibInfo[cidI].rt[5], allCalibInfo[cidI].photometric[2 * cidJ], allCalibInfo[cidI].photometric[2 * cidJ + 1]);
					fclose(fp);
				}
				if (fixPose == 1)
				{
					sprintf(Fname, "%s/Level_%d/P_R_%03d_%03d_%03d.png", Path, pyrID, cidJ, cidI, iter), WriteGridToImage(Fname, resImg, widthJ, heightJ, nchannels);
					sprintf(Fname, "%s/Level_%d/P_S_%03d_%03d_%03d.png", Path, pyrID, cidJ, cidI, iter), WriteGridToImage(Fname, synthImg, widthJ, heightJ, nchannels);
					sprintf(Fname, "%s/pose_%d.txt", Path, cidJ); FILE *fp = fopen(Fname, "a+");
					fprintf(fp, "Iter %d PyrID P: %d Cid: %d %.16e %.16e %.16e %.16e %.16e %.16e %.4f %.4f \n", iter, pyrID, cidI, allCalibInfo[cidI].rt[0], allCalibInfo[cidI].rt[1], allCalibInfo[cidI].rt[2], allCalibInfo[cidI].rt[3], allCalibInfo[cidI].rt[4], allCalibInfo[cidI].rt[5], allCalibInfo[cidI].photometric[2 * cidJ], allCalibInfo[cidI].photometric[2 * cidJ + 1]);
					fclose(fp);
				}
				/*if (fixPose == 0 && fixDepth == 0)
				{
				sprintf(Fname, "%s/Level_%d/R_%03d_%03d_%03d.png", Path, pyrID, cidJ, cidI, iter), WriteGridToImage(Fname, resImg, widthJ, heightJ, nchannels);
				sprintf(Fname, "%s/Level_%d/S_%03d_%03d_%03d.png", Path, pyrID, cidJ, cidI, iter), WriteGridToImage(Fname, synthImg, widthJ, heightJ, nchannels);
				sprintf(Fname, "%s/pose_%d.txt", Path, cidJ); FILE *fp = fopen(Fname, "a+");
				fprintf(fp, "Iter %d PyrID: %d Cid: %d %.16e %.16e %.16e %.16e %.16e %.16e %.4f %.4f \n", iter, pyrID, cidI, allCalibInfo[cidI].rt[0], allCalibInfo[cidI].rt[1], allCalibInfo[cidI].rt[2], allCalibInfo[cidI].rt[3], allCalibInfo[cidI].rt[4], allCalibInfo[cidI].rt[5], allCalibInfo[cidI].photometric[2 * cidJ], allCalibInfo[cidI].photometric[2 * cidJ + 1]);
				fclose(fp);
				}*/
			}
			sprintf(Fname, "%s/iter_%d_%d.txt", Path, cidJ, pyrID);  FILE *fp = fopen(Fname, "a"); fprintf(fp, "%d %.16e\n", iter, sqrt(residuals / npts)); fclose(fp);

			delete[]resImg, delete[]synthImg;
		}
	};
	if (verbose == 1)
	{
		for (int cid = 0; cid < 1; cid++)
		{
			sprintf(Fname, "%s/iter_%d_%d.txt", Path, cid, pyrID);
			FILE *fp = fopen(Fname, "w+"); fclose(fp);
		}
	}

	//MyCallBack *myCallback = new MyCallBack(update_Result, iter, verbose);
	//myCallback->callback_();

	ceres::Solver::Options options;
	options.update_state_every_iteration = true;
	options.callbacks.push_back(new MyCallBack(update_Result, iter, verbose));

	options.num_threads = omp_get_max_threads(); //jacobian eval
	options.num_linear_solver_threads = omp_get_max_threads(); //linear solver
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.linear_solver_type = ceres::ITERATIVE_SCHUR;
	options.preconditioner_type = ceres::JACOBI;
	options.use_inner_iterations = true;
	options.use_nonmonotonic_steps = false;
	options.max_num_iterations = pyrID > 1 ? 50 : 200;
	options.parameter_tolerance = 1.0e-12;
	options.minimizer_progress_to_stdout = verbose == 1 ? true : false;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	if (verbose == 1)
		std::cout << "\n" << summary.BriefReport() << "\n";

	/*if (pyrID >= 4)
	{
	printf("Computing conf score...");
	double startTime = omp_get_wtime();
	ceres::Covariance::Options covar_options;
	covar_options.num_threads = omp_get_max_threads();
	covar_options.algorithm_type == ceres::SUITE_SPARSE_QR;
	ceres::Covariance covariance(covar_options);

	vector<pair<const double*, const double*> > covariance_blocks;
	for (int cidJ = 0; cidJ < 1; cidJ++) //for all refImages
	{
	for (int cidI = 0; cidI < (int)allImgs.size(); cidI++)
	{
	if (cidI == cidJ)
	continue;
	int width = allImgs[cidI].imgPyr[pyrID].cols, height = allImgs[cidI].imgPyr[pyrID].rows, boundary = width / 50;
	for (int jj = boundary; jj < height - boundary; jj++)
	{
	for (int ii = boundary; ii < width - boundary; ii++)
	{
	if (allImgs[cidJ].validPixelsPyr[pyrID][ii + jj*width] ==0)
	continue;

	double *x = &allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*width];
	covariance_blocks.push_back(make_pair(x, x));
	}
	}
	}
	}

	CHECK(covariance.Compute(covariance_blocks, &problem));

	for (int cid = 0; cid < 1; cid++)
	{
	int width = allImgs[cid].imgPyr[pyrID].cols, height = allImgs[cid].imgPyr[pyrID].rows, boundary = width / 50;
	for (int ii = 0; ii < width*height; ii++)
	allImgs[cid].DepthConf[ii] = 9e9;

	for (int jj = boundary; jj < height - boundary; jj++)
	{
	for (int ii = boundary; ii < width - boundary; ii++)
	{
	if (allImgs[cid].validPixelsPyr[pyrID][ii + jj*width] == 0)
	continue;

	covariance.GetCovarianceBlock(&allImgs[cid].InvDepthPyr[pyrID][ii + jj*width], &allImgs[cid].InvDepthPyr[pyrID][ii + jj*width], &allImgs[cid].DepthConf[ii + jj*width]);
	}
	}
	sprintf(Fname, "%s/conf_%d_%d.dat", Path, cid, pyrID), WriteGridBinary(Fname, allImgs[cid].DepthConf, width, height, 1);
	}
	printf("%.2fs\n", omp_get_wtime() - startTime);
	}*/

	/*double maxInvD = 0, minInvD = 9e9;
	for (int cid = 0; cid < 1; cid++)
	{
	int width = allImgs[cid].imgPyr[pyrID].cols, height = allImgs[cid].imgPyr[pyrID].rows, lengthJ = width*height, boundary = width / 50;
	vector<Point3d> aXYZ; aXYZ.reserve(width*height);
	for (int jj = boundary; jj < height - boundary; jj++)
	{
	for (int ii = boundary; ii < width - boundary; ii++)
	{
	if (allImgs[cid].validPixelsPyr[pyrID][ii + jj*width])
	{
	double XYZ[3], rayDir[3], ij[3] = { ii, jj, 1 };
	getRayDir(rayDir, allCalibInfo[cid].activeinvK, allCalibInfo[cid].R, ij);
	double invd = allImgs[cid].InvDepthPyr[pyrID][ii + jj * width];

	for (int kk = 0; kk < 3; kk++)
	XYZ[kk] = rayDir[kk] / invd + allCalibInfo[cid].C[kk];
	aXYZ.push_back(Point3d(XYZ[0], XYZ[1], XYZ[2]));
	}
	}
	}

	double meanX = 0, meanY = 0, meanZ = 0;
	for (auto XYZi : aXYZ)
	meanX += XYZi.x, meanY += XYZi.y, meanZ += XYZi.z;
	meanX /= (int)aXYZ.size(), meanY /= (int)aXYZ.size(), meanZ /= (int)aXYZ.size();

	double stdX = 0, stdY = 0, stdZ = 0;
	for (auto XYZi : aXYZ)
	stdX += pow(XYZi.x - meanX, 2), meanY += pow(XYZi.y - meanY, 2), meanZ += pow(XYZi.z - meanZ, 2);
	stdX = sqrt(stdX / (int)aXYZ.size()), stdY = sqrt(stdY / (int)aXYZ.size()), stdZ = sqrt(stdZ / (int)aXYZ.size());
	double scale = max(max(stdX, stdY), stdZ);

	//simple scale the depth
	for (int jj = boundary; jj < height - boundary; jj++)
	for (int ii = boundary; ii < width - boundary; ii++)
	if (allImgs[cid].validPixelsPyr[pyrID][ii + jj*width])
	allImgs[cid].InvDepthPyr[pyrID][ii + jj * width] /= scale;

	//have to move the camera before scale it down.
	for (int cid = 0; cid < (int)allImgs.size(); cid++)
	{
	double R[9], T[3], C[3];
	GetRTFromrt(allCalibInfo[cid].rt, R, T);
	GetCfromT(R, T, C);
	C[0] = (C[0] - meanX) / scale, C[1] = (C[1] - meanY) / scale, C[2] = (C[2] - meanZ) / scale;
	GetTfromC(R, C, T);
	GetrtFromRT(allCalibInfo[cid].rt, R, T);
	}
	break;
	}*/


	//let's normalized everything
	double maxInvD = 0, minInvD = 9e9;
	for (int cid = 0; cid < 1; cid++)
	{
		int width = allImgs[cid].imgPyr[pyrID].cols, height = allImgs[cid].imgPyr[pyrID].rows, lengthJ = width*height, boundary = width / 50;
		for (int jj = boundary; jj < height - boundary; jj++)
		{
			for (int ii = boundary; ii < width - boundary; ii++)
			{
				if (allImgs[cid].validPixelsPyr[pyrID][ii + jj*width]>0)
				{
					double invd = allImgs[cid].InvDepthPyr[pyrID][ii + jj * width];
					maxInvD = max(invd, maxInvD);
					minInvD = min(invd, minInvD);
				}
			}
		}
		for (int jj = boundary; jj < height - boundary; jj++)
		{
			for (int ii = boundary; ii < width - boundary; ii++)
			{
				if (allImgs[cid].validPixelsPyr[pyrID][ii + jj*width]>0)
				{
					double XYZ[3], rayDir[3], ij[3] = { ii, jj, 1 };
					allImgs[cid].InvDepthPyr[pyrID][ii + jj * width] /= maxInvD; //divide by maxInvD instead of min because the depth tends to be pushed to maxInvD range
				}
			}
		}
	}
	for (int cid = 0; cid < (int)allImgs.size(); cid++)
		for (int ii = 3; ii < 6; ii++)
			allCalibInfo[cid].rt[ii] *= maxInvD;

	if (WriteOutput == 1)
	{
		printf("Saving data...");
		sprintf(Fname, "%s/pose.txt", Path); FILE *fp = fopen(Fname, "a+");
		for (int cid = 0; cid < (int)allImgs.size(); cid++)
			fprintf(fp, "Cid: %d  PyrID: %d %.16e %.16e %.16e %.16e %.16e %.16e %\n", allImgs[cid].frameID, pyrID,
			allCalibInfo[cid].rt[0], allCalibInfo[cid].rt[1], allCalibInfo[cid].rt[2], allCalibInfo[cid].rt[3], allCalibInfo[cid].rt[4], allCalibInfo[cid].rt[5]);
		fclose(fp);

		for (int cid = 0; cid < 1; cid++)
		{
			int width = allImgs[cid].imgPyr[pyrID].cols, height = allImgs[cid].imgPyr[pyrID].rows, lengthJ = width*height, boundary = width / 50;
			sprintf(Fname, "%s/invD_%d_%d.dat", Path, cid, pyrID), WriteGridBinary(Fname, allImgs[cid].InvDepthPyr[pyrID], width, height, 1);

			double rayDir[3], opticalAxis[3], ij[3] = { allCalibInfo[cid].activeK[2], allCalibInfo[cid].activeK[5], 1 };
			getRayDir(opticalAxis, allCalibInfo[cid].activeinvK, allCalibInfo[cid].R, ij);
			double normOptical = sqrt(pow(opticalAxis[0], 2) + pow(opticalAxis[1], 2) + pow(opticalAxis[2], 2));

			/*double *depthMap = new double[width*height];
			for (int jj = boundary; jj < height - boundary; jj++)
			{
			for (int ii = boundary; ii < width - boundary; ii++)
			{
			if (allImgs[cid].validPixelsPyr[pyrID][ii + jj*width] == 0)
			depthMap[ii + jj*width] = 0;
			else
			{
			double ij[3] = { ii, jj, 1 };
			getRayDir(rayDir, allCalibInfo[cid].activeinvK, allCalibInfo[cid].R, ij);
			double cos = (rayDir[0] * opticalAxis[0] + rayDir[1] * opticalAxis[1] + rayDir[2] * opticalAxis[2]) /
			sqrt(pow(rayDir[0], 2) + pow(rayDir[1], 2) + pow(rayDir[2], 2)) / normOptical;

			double invd = allImgs[cid].InvDepthPyr[pyrID][ii + jj * width];
			if (abs(invd) > DBL_MAX)
			depthMap[ii + jj * width] = DBL_MAX;
			else
			depthMap[ii + jj* width] = 1.0 / (cos / invd + DBL_MIN);
			}
			}
			}
			sprintf(Fname, "%s/invfrontoD_%d_%d.dat", Path, cid, pyrID), WriteGridBinary(Fname, depthMap, width, height);
			delete[]depthMap;*/

			if (nchannels == 1)
				sprintf(Fname, "%s/3d_%d_%d.txt", Path, cid, pyrID);
			else
				sprintf(Fname, "%s/3d_%d_%d.txt", Path, cid, pyrID);
			fp = fopen(Fname, "w+");
			for (int jj = boundary; jj < height - boundary; jj++)
			{
				for (int ii = boundary; ii < width - boundary; ii++)
				{
					if (allImgs[cid].validPixelsPyr[pyrID][ii + jj*width]>0)
					{
						double XYZ[3], rayDir[3], ij[3] = { ii, jj, 1 };
						getRayDir(rayDir, allCalibInfo[cid].activeinvK, allCalibInfo[cid].R, ij);
						double invd = allImgs[cid].InvDepthPyr[pyrID][ii + jj * width];

						for (int kk = 0; kk < 3; kk++)
							XYZ[kk] = rayDir[kk] / invd + allCalibInfo[cid].C[kk];
						if (nchannels == 1)
							fprintf(fp, "%.8e %.8e %.8e\n", XYZ[0], XYZ[1], XYZ[2]);
						else
							fprintf(fp, "%.8e %.8e %.8e %d %d %d %d\n", XYZ[0], XYZ[1], XYZ[2],
							(int)allImgs[cid].imgPyr[pyrID].data[(ii + jj*width)*nchannels + 2], (int)allImgs[cid].imgPyr[pyrID].data[(ii + jj*width)*nchannels + 1], (int)allImgs[cid].imgPyr[pyrID].data[(ii + jj*width)*nchannels], allImgs[cid].validPixelsPyr[pyrID][ii + jj*width]);
					}
				}
			}
			fclose(fp);
		}
	}

	return summary.final_cost;
}
int DirectAlignment(char *Path, DirectAlignPara &alignmentParas, vector<ImgData> &allImgs, vector<CameraData> &allCalibInfo, int smallAngle = 0, double scale = 1.0)
{
	int nscales = 4, //actually 6 scales = 5 down sampled images + org image
		innerIter = 20;

	//find texture region in the refImg and store in the vector
	for (int cid = 0; cid < (int)allImgs.size(); cid++)
	{
		int width = allImgs[cid].width, height = allImgs[cid].height, nchannels = allImgs[cid].nchannels, boundary = width / 50;
		for (int pyrID = 0; pyrID < nscales + 1; pyrID++)
			allImgs[cid].scaleFactor.push_back(1.0 / pow(2, pyrID));

		GaussianBlur(allImgs[cid].color, allImgs[cid].color, Size(5, 5), 0.707);
		buildPyramid(allImgs[cid].color, allImgs[cid].imgPyr, nscales);

		if (cid > 0)
			continue;

		for (int jj = boundary; jj < height - boundary; jj++)
		{
			for (int ii = boundary; ii < width - boundary; ii++)
			{
				//calculate first order image derivatives: using 1 channel should be enough. This is actually gaussian derivative filter.
				float dx = (float)(int)allImgs[cid].color.data[(ii + 1)*nchannels + jj*nchannels*width] - (float)(int)allImgs[cid].color.data[(ii - 1)*nchannels + jj*nchannels*width], //1, 0, --1
					dy = (float)(int)allImgs[cid].color.data[ii *nchannels + (jj + 1)*nchannels*width] - (float)(int)allImgs[cid].color.data[ii *nchannels + (jj - 1)*nchannels*width]; //1, 0, -1
				allImgs[cid].Grad[ii + jj*width] = sqrt(dx*dx + dy*dy);
			}
		}

		vector<Point2i> segQueue; segQueue.reserve(1e6);
		vector<bool> processedQueue; processedQueue.reserve(1e6);
		if (alignmentParas.removeSmallRegions)
		{
			bool *processedPixels = new bool[width*height];
			for (int ii = 0; ii < width*height; ii++)
				processedPixels[ii] = false;
			for (int jj = boundary; jj < height - boundary; jj++)
			{
				for (int ii = boundary; ii < width - boundary; ii++)
				{
					float mag = allImgs[cid].Grad[ii + jj*width];
					if (!processedPixels[ii + jj*width] && mag > alignmentParas.gradientThresh)
					{
						processedQueue.clear(), segQueue.clear();
						processedQueue.push_back(false), segQueue.push_back(Point2i(ii, jj));
						processedPixels[ii + jj*width] = true;
						while (true)
						{
							int npts_before = (int)segQueue.size();
							for (int kk = 0; kk < (int)segQueue.size(); kk++)
							{
								if (processedQueue[kk])
									continue;
								processedQueue[kk] = true;
								for (int i = -1; i <= 1; i++)
								{
									for (int j = -1; j <= 1; j++)
									{
										int x = segQueue[kk].x + i, y = segQueue[kk].y + j;
										if (x<boundary || x>width - boundary || y<boundary || y>height - boundary)
											continue;
										mag = allImgs[cid].Grad[x + y*width];
										bool boolean = processedPixels[x + y*width];
										if (!boolean && mag > alignmentParas.gradientThresh)
											segQueue.push_back(Point2i(x, y)), processedQueue.push_back(false), processedPixels[x + y*width] = true;
									}
								}
							}
							if ((int)segQueue.size() - npts_before == 0)
								break;
						}

						if (segQueue.size() > (int)(100.0))//10x10 @ 1920x1080
						{
							for (int kk = 0; kk < (int)segQueue.size(); kk++)
							{
								int x = segQueue[kk].x, y = segQueue[kk].y;
								allImgs[cid].validPixels[x + y*width] = 1;
							}
						}
					}
				}
			}
			delete[]processedPixels;
		}
		else
		{
			/*for (int jj = boundary; jj < height - boundary; jj++)
			for (int ii = boundary; ii < width - boundary; ii++)
			if (allImgs[cid].Grad[ii + jj*width] > alignmentParas.gradientThresh)
			allImgs[cid].validPixels[ii + jj*width] = 1;*/

			int nPartions = 4, HarrisminDistance = 5;
			vector<Point2f> uv; uv.reserve(100000);
			Mat img; cvtColor(allImgs[cid].color, img, CV_BGR2GRAY);
			BucketGoodFeaturesToTrack(img, uv, nPartions, 100000, 0.001, HarrisminDistance*max(1, width / 1920), 7 * max(1, width / 1920), 0, 0.04);

			int count = 0;
			for (auto pt : uv)
			{
				if (pt.x < boundary || pt.y < boundary || pt.x > width - boundary || pt.y > height - boundary)
					continue;
				allImgs[cid].validPixels[(int)pt.x + (int)(pt.y)*width] = 1;
				count++;
			}

			int orgnUV = uv.size();
			char Fname[512];  sprintf(Fname, "%s/K.txt", Path); FILE *fp = fopen(Fname, "r");
			float u, v, s;
			while (fscanf(fp, "%f %f %f %f %f %f ", &u, &v, &s, &s, &s, &s) != EOF)
				uv.push_back(Point2f(u, v));
			fclose(fp);

			for (int ii = orgnUV; ii < uv.size(); ii++)
			{
				Point2f pt = uv[ii];
				allImgs[cid].validPixels[(int)pt.x + (int)(pt.y)*width] = ii + 2 - orgnUV; //let have it start from 1 to avoid confusion with other features
			}


			/*int i, j; double invd;
			char Fname[512]; sprintf(Fname, "%s/depth.txt", Path);
			FILE *fp = fopen(Fname, "r");
			while (fscanf(fp, "%d %d %lf ", &i, &j, &invd) != EOF)
			{
			if (i < boundary || j < boundary || i> width - boundary || j > height - boundary)
			continue;
			allImgs[cid].validPixels[i + j*width] = true;
			//allImgs[cid].InvDepth[i + j*width] = invd;
			}
			fclose(fp);*/
		}

		BuildDataPyramid(allImgs[cid].validPixels, allImgs[cid].validPixelsPyr, allImgs[cid].width, allImgs[cid].height, nscales, true);
		BuildDataPyramid(allImgs[cid].Grad, allImgs[cid].gradPyr, allImgs[cid].width, allImgs[cid].height, nscales);
		BuildDataPyramid(allImgs[cid].InvDepth, allImgs[cid].InvDepthPyr, allImgs[cid].width, allImgs[cid].height, nscales);
	}

	/*int pyrID = 1;
	for (int cid = 0; cid < 1; cid++)
	{
	char Fname[512];  sprintf(Fname, "%s/invD_%d_%d.dat", Path, cid, pyrID);
	ReadGridBinary(Fname, allImgs[cid].InvDepthPyr[pyrID], allImgs[cid].imgPyr[pyrID].cols, allImgs[cid].imgPyr[pyrID].rows);

	UpsamleDepthNN(allImgs[cid].InvDepthPyr[pyrID], allImgs[cid].InvDepthPyr[pyrID - 1], allImgs[cid].validPixelsPyr[pyrID], allImgs[cid].validPixelsPyr[pyrID - 1],
	allImgs[cid].imgPyr[pyrID].cols, allImgs[cid].imgPyr[pyrID].rows, allImgs[cid].imgPyr[pyrID - 1].cols, allImgs[cid].imgPyr[pyrID - 1].rows);
	}
	char Fname[512];  sprintf(Fname, "%s/pose_.txt", Path); FILE *fp = fopen(Fname, "r");
	for (int cid = 0; cid < (int)allImgs.size(); cid++)
	{
	int dummy; fscanf(fp, "%d ", &dummy);
	if (dummy != allCalibInfo[cid].frameID)
	int a = 0;
	for (int ii = 0; ii < 6; ii++)
	fscanf(fp, "%lf ", &allCalibInfo[cid].rt[ii]);
	}
	fclose(fp);*/

	double startTime = omp_get_wtime();
	for (int pyrID = nscales; pyrID >= 0; pyrID--)
	{
		printf("\n\n@ level: %d\n", pyrID);
		for (int cid = 0; cid < (int)allImgs.size(); cid++)
		{
			if (cid > 0)
				continue;

			char Fname[512];
			sprintf(Fname, "%s/Level_%d", Path, pyrID); makeDir(Fname);
			sprintf(Fname, "%s/Level_%d/%03d.png", Path, pyrID, cid);
			imwrite(Fname, allImgs[cid].imgPyr[pyrID]);

			sprintf(Fname, "%s/Level_%d/m_%03d.png", Path, pyrID, cid);
			WriteGridToImage(Fname, allImgs[cid].validPixelsPyr[pyrID], allImgs[cid].imgPyr[pyrID].cols, allImgs[cid].imgPyr[pyrID].rows, 1);
		}

		/*double startTimeI = omp_get_wtime();
		double percentage, innerCost1, innerCost2, pinnerCost1 = 1e16, pinnerCost2 = 1e16;
		for (int innerID = 0; innerID < innerIter; innerID++) //Pose is mostly well estimated even with nosiy depth. Depth is improved with good pose. -->Alternating optim, which is extremely helpful in practice
		{
		printf("\n@(Pose) Iter %d: ", innerID);
		innerCost2 = DirectAlignmentPyr(Path, alignmentParas, allImgs, allCalibInfo, 1, 0, pyrID, smallAngle, 0, 0);

		printf("\n@(Depth) Iter %d: ", innerID); //at low res, calibration error is not servere, let's optim depth first
		innerCost1 = DirectAlignmentPyr(Path, alignmentParas, allImgs, allCalibInfo, 0, 1, pyrID, smallAngle, 0, 0);

		percentage = abs(innerCost1 + innerCost2 - pinnerCost1 - pinnerCost2) / (pinnerCost1 + pinnerCost2);
		printf("\nChange: %.3e\n", percentage);
		if (percentage < 1.0e-3)
		break;
		pinnerCost1 = innerCost1, pinnerCost2 = innerCost2;
		}
		printf("\nTotal time: %.2f\n\n", omp_get_wtime() - startTimeI);

		//if (pyrID == 0)
		//	DirectAlignmentPyr(Path, alignmentParas, allImgs, allCalibInfo, pyrID < 5 ? 0 : 1, 0, pyrID, smallAngle, 1, 0);*/

		DirectAlignmentPyr(Path, alignmentParas, allImgs, allCalibInfo, 0, 0, pyrID, smallAngle, 1, 1);

		if (pyrID != 0)
			for (int cid = 0; cid < 1; cid++)
			{
				UpsamleDepthNN(allImgs[cid].InvDepthPyr[pyrID], allImgs[cid].InvDepthPyr[pyrID - 1], allImgs[cid].validPixelsPyr[pyrID], allImgs[cid].validPixelsPyr[pyrID - 1],
					allImgs[cid].imgPyr[pyrID].cols, allImgs[cid].imgPyr[pyrID].rows, allImgs[cid].imgPyr[pyrID - 1].cols, allImgs[cid].imgPyr[pyrID - 1].rows);

				//char Fname[512];  sprintf(Fname, "%s/invD_%d_%d.dat", Path, cid, pyrID - 1);
				//WriteGridBinary(Fname, allImgs[cid].InvDepthPyr[pyrID - 1], allImgs[cid].imgPyr[pyrID - 1].cols, allImgs[cid].imgPyr[pyrID - 1].rows, 1);
			}
	}
	printf("\n\nTotal time: %.2f\n", omp_get_wtime() - startTime);

	for (int cid = 0; cid < 1; cid++)
		for (int pyrID = 0; pyrID < nscales + 1; pyrID++)
			delete[]allImgs[cid].InvDepthPyr[pyrID], delete[]allImgs[cid].gradPyr[pyrID], delete[]allImgs[cid].validPixelsPyr[pyrID];

	return 1;
}

int RescaleMicro3D(char *Path, int refF1 = 350, int refF2 = 365)
{
	double scale1 = 1.542329, scale2 = 0.8168816;
	double rt1[] = { 0.4257606806235341, -0.0054366615673998, 0.0761549609425020, -5.4898060490921612, -0.1419847087298890, -0.3444073481333543 };
	double rt2[] = { 0.4549580531995368, 0.0112249635446353, 0.0707017186351800, -5.3832044614201253, -0.2122191938992374, -0.3152637961078051 };

	char Fname[512];

	Point3d p3d; Point3i pc;
	vector<Point3d> p3d1, p3d2;
	vector<Point3i> pc1, pc2;

	sprintf(Path, "%s/%d/3d_0_1.txt", Path, refF1);  FILE *fp = fopen(Fname, "r");
	while (fscanf(fp, "%lf %lf %lf %d %d %d ", &p3d.x, &p3d.y, &p3d.z, &pc.x, &pc.y, &pc.y) != EOF)
		p3d1.push_back(p3d); pc1.push_back(pc);
	fclose(fp);

	sprintf(Path, "%s/%d/3d_0_1.txt", Path, refF2);  fp = fopen(Fname, "r");
	while (fscanf(fp, "%lf %lf %lf %d %d %d ", &p3d.x, &p3d.y, &p3d.z, &pc.x, &pc.y, &pc.y) != EOF)
		p3d2.push_back(p3d); pc2.push_back(pc);
	fclose(fp);

	double R1[9], R2[9], T1[3], T2[3];
	GetRTFromrt(rt1, R1, T1);
	GetRTFromrt(rt2, R2, T2);
	for (int ii = 0; ii < (int)p3d1.size(); ii++)
	{
		double X = p3d1[ii].x / scale1, Y = p3d1[ii].x / scale1, Z = p3d1[ii].z / scale1;
		double tX = X*R1[0] + Y *R1[1] + Z*R1[2] + T1[0];
		double tY = X*R1[3] + Y *R1[4] + Z*R1[5] + T1[1];
		double tZ = X*R1[6] + Y *R1[7] + Z*R1[8] + T1[2];
		p3d1[ii] = Point3d(tX, tY, tZ);
	}

	for (int ii = 0; ii < (int)p3d2.size(); ii++)
	{
		double X = p3d2[ii].x, Y = p3d2[ii].x, Z = p3d2[ii].z;
		double tX = X*R1[0] + Y *R1[1] + Z*R1[2] + T1[0];
		double tY = X*R1[3] + Y *R1[4] + Z*R1[5] + T1[1];
		double tZ = X*R1[6] + Y *R1[7] + Z*R1[8] + T1[2];
		p3d2[ii] = Point3d(tX, tY, tZ);
	}
	return 0;
}
int main(int argc, char** argv)
{
	//srand(time(NULL));
	srand(1);
	char Fname[512], Path[] = "C:/Data/MSR/Micro/Rocks";

	int nchannels = 3, refF = 100, rangeF = 15, stepF = 1;
	vector<int> vrefF; vrefF.push_back(50), vrefF.push_back(150), vrefF.push_back(200), vrefF.push_back(250),
		vrefF.push_back(300), vrefF.push_back(450), vrefF.push_back(500), vrefF.push_back(550), vrefF.push_back(590), vrefF.push_back(600);
	for (auto refF : vrefF)
	{
		vector<ImgData> allImgs;
		vector<CameraData> allCalibInfo;

		int mode = 0;
		if (mode == 0)
		{
			double scale = 1.0, DepthLow = -1, DepthHigh = -1;
			double dataWeight = 1.0, regIntraWeight = 0.0, //atof(argv[1]),
				regInterWeight = 1.0 - dataWeight - regIntraWeight;
			double reprojectionSigma = 0.3, colorSigma = 3.0, depthSigma = 1.0;  //expected std of variables
			double ImGradientThesh = 5;
			DirectAlignPara alignmentParas(dataWeight, regIntraWeight, regInterWeight, colorSigma, depthSigma, ImGradientThesh, DepthLow, DepthHigh, reprojectionSigma);
			alignmentParas.HuberSizeColor = 30, alignmentParas.removeSmallRegions = false;

			/*vector<int> frames; frames.push_back(0), frames.push_back(14), frames.push_back(15), frames.push_back(27), frames.push_back(28);
		double SfMdistance = TriangulatePointsFromNonCorpusCameras(Path, allCalibInfo, frames, 1, 2, 2.0);
		printf("SfM measured distance: %.6f\n ", SfMdistance);
		return 0;*/

			vector<int> allframes;
			allframes.push_back(refF);
			for (int ii = stepF; ii <= rangeF; ii += stepF)
				allframes.push_back(refF + ii), allframes.push_back(refF - ii);
			for (auto fid : allframes)
			{
				CameraData camI;
				camI.frameID = fid;
				//camI.intrinsic[0] = 851.687977, camI.intrinsic[1] = 780.464167, camI.intrinsic[3] = 604.448653, camI.intrinsic[4] = 442.588207,
				//camI.distortion[0] = 0.176961, camI.distortion[1] = -0.376240, camI.distortion[2] = 0.242703, camI.distortion[3] = -0.014976, camI.distortion[4] = -0.004914; //Bench--> does not work
				//camI.intrinsic[0] = 1735.130306, camI.intrinsic[1] = camI.intrinsic[0], camI.intrinsic[3] = 960, camI.intrinsic[4] = 540,
				//camI.distortion[0] = 0, camI.distortion[1] = 0, camI.distortion[2] = 0, camI.distortion[3] = 0, camI.distortion[4] = 0; //Bikes--> work
				//camI.intrinsic[0] = 700.0, camI.intrinsic[1] = 700.0, camI.intrinsic[3] = 960, camI.intrinsic[4] = 540, camI.distortion[0] = 0, camI.distortion[1] = 0.0143828495182; //Synth
				camI.intrinsic[0] = 1943.98413086, camI.intrinsic[1] = camI.intrinsic[0], camI.intrinsic[3] = 960, camI.intrinsic[4] = 540,
					camI.distortion[0] = 0, camI.distortion[1] = 0, camI.distortion[2] = 0, camI.distortion[3] = 0, camI.distortion[4] = 0; //Rocks
				//camI.intrinsic[0] = 1161.852242, camI.intrinsic[1] = 1162.484980, camI.intrinsic[3] = 960.104777, camI.intrinsic[4] = 540.489932;
				//	camI.distortion[0] = -0.134064, camI.distortion[1] = 0.119985, camI.distortion[2] = -0.023575, camI.distortion[3] = 0.000378, camI.distortion[4] = 0.000467; //Drone --Does not work
				//camI.intrinsic[0] = 1233.617524, camI.intrinsic[1] = camI.intrinsic[0], camI.intrinsic[3] = 640, camI.intrinsic[4] = 360, camI.distortion[0] = 0, camI.distortion[1] = 0; //Staris-->work
				//camI.intrinsic[0] = 1345.546348, camI.intrinsic[1] = , camI.intrinsic[0], camI.intrinsic[3] = 640, camI.intrinsic[4] = 360, camI.distortion[0] = 0, camI.distortion[1] = 0; //Stone
				//camI.intrinsic[0] = 1560.119972, camI.intrinsic[1] = , camI.intrinsic[0], camI.intrinsic[3] = 960, camI.intrinsic[4] = 540, camI.distortion[0] = 0, camI.distortion[1] = 0; //Desk
				//camI.intrinsic[0] = 1991.1764, camI.intrinsic[1] = , camI.intrinsic[0], camI.intrinsic[3] = 960, camI.intrinsic[4] = 540, camI.distortion[0] = 0, camI.distortion[1] = 0; //kitchen Bear Creek-->work
				//camI.intrinsic[0] = 1833.892646, camI.intrinsic[1] = , camI.intrinsic[0], camI.intrinsic[3] = 1920 / 2, camI.intrinsic[4] = 1080 / 2, camI.distortion[0] = 0.0676, camI.distortion[1] = 0.0054; //MSR2 (living room Bear Creek -->work)
				//camI.intrinsic[0] = 1953.567326, camI.intrinsic[1] = , camI.intrinsic[0], camI.intrinsic[3] = 1920 / 2, camI.intrinsic[4] = 1080 / 2, camI.distortion[0] = 0, camI.distortion[1] = 0; //MSR1
				//camI.intrinsic[0] = 1169.175198, camI.intrinsic[1] = , camI.intrinsic[0], camI.intrinsic[3] = 720, camI.intrinsic[4] = 540, camI.distortion[0] = 0, camI.distortion[1] = 0; //MSR6

				camI.ShutterModel = 0;
				camI.intrinsic[2] = 0;
				GetKFromIntrinsic(camI.K, camI.intrinsic);
				GetiK(camI.invK, camI.K);

				for (int ii = 0; ii < 6; ii++)
					camI.rt[ii] = 0.0;
				GetRTFromrt(camI.rt, camI.R, camI.T);
				GetCfromT(camI.R, camI.T, camI.C);
				AssembleP(camI.K, camI.R, camI.T, camI.P);
				allCalibInfo.push_back(camI);
			}
			for (int ii = 0; ii < (int)allCalibInfo.size(); ii++)
			{
				ImgData imgI;
				sprintf(Fname, "%s/Images/%.4d.png", Path, allCalibInfo[ii].frameID);
				imgI.color = imread(Fname, nchannels == 1 ? 0 : 1);
				if (imgI.color.data == NULL)
				{
					printf("Cannot load %s\n", Fname);
					continue;
				}
				imgI.width = imgI.color.cols, imgI.height = imgI.color.rows, imgI.frameID = allCalibInfo[ii].frameID, imgI.nchannels = nchannels;
				allImgs.push_back(imgI);
			}
			//LensCorrection(allImgs, allCalibInfo[0].intrinsic, allCalibInfo[0].distortion);

			InitializeRT(allImgs, allCalibInfo);
			/*sprintf(Fname, "%s/Corpus/n3dGL.xyz", Path);
			Corpus CorpusData; ReadCorpusInfo(Fname, CorpusData, scale);
			float depth[2];
			ExtractDepthFromCorpus(CorpusData, allCalibInfo[0], 0, depth);*/

			for (int cid = 0; cid < 1; cid++)
			{
				allImgs[cid].validPixels = new int[allImgs[cid].width*allImgs[cid].height];
				allImgs[cid].Grad = new float[allImgs[cid].width*allImgs[cid].height];
				allImgs[cid].InvDepth = new double[allImgs[cid].width*allImgs[cid].height];
				allImgs[cid].DepthConf = new double[allImgs[cid].width*allImgs[cid].height];

				for (int jj = 0; jj < allImgs[cid].height; jj++)
					for (int ii = 0; ii < allImgs[cid].width; ii++)
						allImgs[cid].validPixels[ii + jj*allImgs[cid].width] = 0, allImgs[cid].Grad[ii + jj*allImgs[cid].width] = 0.0, allImgs[cid].InvDepth[ii + jj*allImgs[cid].width] = 1.0 / min(2.5, max(0.001, gaussian_noise(1.0, 0.5)));
			}

			/*sprintf(Fname, "%s/SMC.txt", Path);
			FILE *fp = fopen(Fname, "r");
			if (fp == NULL)
			return 1;
			else
			{
			for (int jj = 0; jj < (int)allframes.size(); jj++)
			{
			int fid;  fscanf(fp, "%d ", &fid);
			if (fid != allCalibInfo[jj].frameID)
			printf("Problem reading pose para for %d\n", fid);

			for (int ii = 0; ii < 6; ii++)
			fscanf(fp, "%lf ", &allCalibInfo[jj].rt[ii]);
			GetRTFromrt(allCalibInfo[jj].rt, allCalibInfo[jj].R, allCalibInfo[jj].T);
			GetCfromT(allCalibInfo[jj].R, allCalibInfo[jj].T, allCalibInfo[jj].C);
			AssembleP(allCalibInfo[jj].K, allCalibInfo[jj].R, allCalibInfo[jj].T, allCalibInfo[jj].P);
			}
			fclose(fp);
			}*/


			printf("\n\n\nWorking on (%d, %d, %d)\n", refF - rangeF, refF, refF + rangeF);
			sprintf(Fname, "%s/%d", Path, refF); makeDir(Fname);
			DirectAlignment(Fname, alignmentParas, allImgs, allCalibInfo, 1, scale);
		}
		else if (mode == 1)
		{
			vector<int> allframes; allframes.push_back(refF);
			for (int ii = stepF; ii < rangeF; ii += stepF)
				allframes.push_back(refF + ii), allframes.push_back(refF - ii);
			for (int ii = 0; ii < allframes.size(); ii++)
			{
				ImgData imgI;
				sprintf(Fname, "%s/images/%.4d.png", Path, allframes[ii]);
				imgI.color = imread(Fname, nchannels == 1 ? 0 : 1);
				if (imgI.color.data == NULL)
					continue;
				imgI.frameID = allframes[ii], imgI.width = imgI.color.cols, imgI.height = imgI.color.rows, imgI.nchannels = nchannels;
				allImgs.push_back(imgI);
			}

			SparseFlowData sfd;
			ComputeFlowForMicroBA(allImgs, sfd);

			FILE *fp = fopen("C:/temp/x.txt", "w+");
			fprintf(fp, "%d %d\n", sfd.nfeatures, sfd.nimages);
			for (int jj = 0; jj < sfd.nfeatures; jj++)
			{
				for (int ii = 0; ii < sfd.nimages; ii++)
					fprintf(fp, "%.4f %.4f ", sfd.flow[jj*sfd.nimages + ii].x, sfd.flow[jj*sfd.nimages + ii].y);
				fprintf(fp, "\n");
			}
			fclose(fp);

			/*int nimages, nfeatures;  float du, dv;
			FILE *fp = fopen("C:/temp/x.txt", "r");
			while (fscanf(fp, "%d %d ", &nfeatures, &nimages) != EOF)
			{
			sfd.nfeatures = nfeatures, sfd.nimages = nimages;
			sfd.flow = new Point2f[nfeatures*nimages];
			for (int jj = 0; jj < sfd.nfeatures; jj++)
			{
			for (int ii = 0; ii < sfd.nimages; ii++)
			{
			fscanf(fp, "%f %f ", &du, &dv);
			sfd.flow[jj*sfd.nimages + ii].x = du, sfd.flow[jj*sfd.nimages + ii].y = dv;
			}
			}
			}
			fclose(fp);*/

			for (auto fid : allframes)
			{
				CameraData camI;
				camI.frameID = fid;

				camI.intrinsic[0] = 1943.98413086, camI.intrinsic[3] = 960, camI.intrinsic[4] = 540, camI.distortion[0] = 0, camI.distortion[1] = 0.0143828495182; //Rocks
				//camI.intrinsic[0] = 0.7 * 3840, camI.intrinsic[3] = 3840 / 2, camI.intrinsic[4] = 2160 / 2, camI.distortion[0] = 0, camI.distortion[1] = 0; //Drone
				//camI.intrinsic[0] = 1233.617524, camI.intrinsic[3] = 640, camI.intrinsic[4] = 360, camI.distortion[0] = 0, camI.distortion[1] = 0; //Staris
				//camI.intrinsic[0] = 1345.546348, camI.intrinsic[3] = 640, camI.intrinsic[4] = 360, camI.distortion[0] = 0, camI.distortion[1] = 0; //Stone
				//camI.intrinsic[0] = 1560.119972, camI.intrinsic[3] = 960, camI.intrinsic[4] = 540, camI.distortion[0] = 0, camI.distortion[1] = 0; //Desk
				//camI.intrinsic[0] = 1991.1764, camI.intrinsic[3] = 960, camI.intrinsic[4] = 540, camI.distortion[0] = 0, camI.distortion[1] = 0; //kitchen Bear Creek
				//camI.intrinsic[0] = 1833.892646, camI.intrinsic[3] = 1920 / 2, camI.intrinsic[4] = 1080 / 2, camI.distortion[0] = 0.0676, camI.distortion[1] = 0.0054; //MSR2
				//camI.intrinsic[0] = 1953.567326, camI.intrinsic[3] = 1920 / 2, camI.intrinsic[4] = 1080 / 2, camI.distortion[0] = 0, camI.distortion[1] = 0; //MSR1
				//camI.intrinsic[0] = 1169.175198, camI.intrinsic[3] = 720, camI.intrinsic[4] = 540, camI.distortion[0] = 0, camI.distortion[1] = 0; //MSR6

				camI.ShutterModel = 0;
				camI.intrinsic[1] = camI.intrinsic[0], camI.intrinsic[2] = 0;
				GetKFromIntrinsic(camI.K, camI.intrinsic);
				GetiK(camI.invK, camI.K);

				for (int ii = 0; ii < 6; ii++)
					camI.rt[ii] = 0.0;
				GetRTFromrt(camI.rt, camI.R, camI.T);
				GetCfromT(camI.R, camI.T, camI.C);
				AssembleP(camI.K, camI.R, camI.T, camI.P);
				allCalibInfo.push_back(camI);
			}

			for (int ii = 0; ii < (int)allframes.size(); ii++)
				for (int jj = 0; jj < 3; jj++)
					allCalibInfo[ii].rt[jj] = 0; //rotation is not precise, translation is pretty good via bspline of keyframes

			double reProjectionSigma = 1.0;
			FlowBasedBundleAdjustment(Path, allImgs, sfd, allCalibInfo, reProjectionSigma, 0, 0, 1);
			waitKey(0);
		}
	}

	return 0;
}






