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

int GetKeyFrameFeature3D(char *Path, int keyF)
{
	int dummy;
	Point3d p3d; Point3i pc;
	vector<Point3d> vp3d, vp3di;
	vector<Point3i> vpc, vpci;

	char Fname[512];  sprintf(Fname, "%s/Corpus/Corpus_3D.txt", Path);  FILE *fp = fopen(Fname, "r");
	fscanf(fp, "%d %d %d ", &dummy, &dummy, &dummy);
	while (fscanf(fp, "%lf %lf %lf %d %d %d ", &p3d.x, &p3d.y, &p3d.z, &pc.x, &pc.y, &pc.y) != EOF)
		vp3d.push_back(p3d); vpc.push_back(pc);
	fclose(fp);


	vector<int> threeDId;
	int totalPts = 0, n3D, id3D, cid = 0;
	sprintf(Fname, "%s/Corpus/Corpus_threeDIdAllViews.txt", Path); fp = fopen(Fname, "r");
	while (fscanf(fp, "%d ", &n3D) != EOF)
	{
		if (cid == keyF)
			threeDId.reserve(n3D);
		for (int ii = 0; ii < n3D; ii++)
		{
			fscanf(fp, "%d ", &id3D);
			if (cid == keyF)
				threeDId.push_back(id3D);
		}
		cid++;
	}
	fclose(fp);

	sprintf(Fname, "%s/Corpus/CorpusK_%.4d.txt", Path, keyF); fp = fopen(Fname, "r");
	float u, v, s;
	vector<Point3f> uvs;
	while (fscanf(fp, "%f %f %f ", &u, &v, &s) != EOF)
		uvs.push_back(Point3f(u, v, s));
	fclose(fp);

	sprintf(Fname, "%s/Corpus/CorpusK3D_%.4d.txt", Path, keyF); fp = fopen(Fname, "w");
	for (int ii = 0; ii < uvs.size(); ii++)
	{
		int pid = threeDId[ii];
		fprintf(fp, "%d %.3f %.3f %.2f %.6e %.6e %.6e\n", pid, uvs[ii].x, uvs[ii].y, uvs[ii].z, vp3d[pid].x, vp3d[pid].y, vp3d[pid].z);
	}
	fclose(fp);

	return 0;
}
int RescaleMicro3D(char *Path, int refF1 = 350, int refF2 = 365)
{
	/*double rt1[] = { 0.4257606806235341, -0.0054366615673998, 0.0761549609425020, -5.4898060490921612, -0.1419847087298890, -0.3444073481333543 };
	double rt2[] = { 0.4549580531995368, 0.0112249635446353, 0.0707017186351800, -5.3832044614201253, -0.2122191938992374, -0.3152637961078051 };

	char Fname[512];

	int id;  Point3d p3d; Point3i pc;
	vector<Point3d> p3d1, p3d2;
	vector<Point3i> pc1, pc2;
	vector<int> id1, id2;

	sprintf(Path, "%s/%d/3d_0_0.txt", Path, refF1);  FILE *fp = fopen(Fname, "r");
	while (fscanf(fp, "%lf %lf %lf %d %d %d %d %d ", &p3d.x, &p3d.y, &p3d.z, &pc.x, &pc.y, &pc.y, &id) != EOF)
	p3d1.push_back(p3d); pc1.push_back(pc), id1.push_back(id);
	fclose(fp);

	sprintf(Fname, "%s/%d/K.txt", Path, refF1); fp = fopen(Fname, "r");
	float u, v, s, x, y, z;
	vector<Point3d> p3d1_;
	while (fscanf(fp, "%f %f %f %f %f %f ", &u, &v, &s, &x, &y, &z) != EOF)
	p3d1_.push_back(Point3d(x, y, z));
	fclose(fp);

	//find scale


	sprintf(Path, "%s/%d/3d_0_0.txt", Path, refF2);  fp = fopen(Fname, "r");
	while (fscanf(fp, "%lf %lf %lf %d %d %d %d", &p3d.x, &p3d.y, &p3d.z, &pc.x, &pc.y, &pc.y, &id) != EOF)
	p3d2.push_back(p3d); pc2.push_back(pc), id2.push_back(id);
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
	}*/
	return 0;
}

int CayleyProjection(double *intrinsic, double* rt, double *wt, Point2d &predicted, Point3d Point, int width, int height)
{
	//Solving Eq. (5) of the p6p rolling shutter paper for the row location given all other parameters

	//transformed_X = R(v)*X
	double p[3] = { Point.x, Point.y, Point.z };
	double R_global[9];	getRfromr(rt, R_global);
	double Tx = rt[3], Ty = rt[4], Tz = rt[5];
	double tx = wt[3], ty = wt[4], tz = wt[5];
	double K[9] = { intrinsic[0], intrinsic[2], intrinsic[3], 0.0, intrinsic[1], intrinsic[4], 0.0, 0.0, 1.0 };

	/*if (0)//abs(wt[0]) + abs(wt[1]) + abs(wt[2]) > 0.5 && (abs(tx) + abs(ty) + abs(tz) > 30))
	{
	double tp[3]; mat_mul(R_global, p, tp, 3, 3, 1);
	double X = tp[0], Y = tp[1], Z = tp[2];
	double wx = wt[0], wy = wt[1], wz = wt[2], wx2 = wx*wx, wy2 = wy*wy, wz2 = wz*wz, wxz = wx*wz, wxy = wx*wy, wyz = wy*wz;

	//Set up polynomial coefficients (obtained from matlab symbolic)
	double c[5];
	Mat coeffs(1, 5, CV_64F, c);
	c[4] = tz*wz2 + tz*wy2 + tz*wx2;
	c[3] = 2.0 * Y*wyz + 2.0 * X*wxz - ty*wz2 - ty*wy2 - ty*wx2 - Z*wy2 - Z*wx2 + Z*wz2 + Tz*wz2 + Tz*wy2 + Tz*wx2;
	c[2] = -2.0 * Z*wyz - 2.0 * X*wxy - Y*wy2 - Ty*wz2 - Ty*wy2 - Ty*wx2 + 2.0 * Y*wx - 2.0 * X*wy + Y*wz2 + Y*wx2 + tz;
	c[1] = 2.0 * Z*wx - 2.0 * X*wz - ty + Z + Tz;
	c[0] = -Y - Ty;

	std::vector<std::complex<double> > roots;
	solvePoly(coeffs, roots);

	int count = 0;
	for (int ii = 0; ii < roots.size(); ii++)
	{
	if (fabs(roots[ii].imag()) > 1e-10)
	continue;

	double j = roots[ii].real(), j2 = j*j, j3 = j2*j;
	double lamda = (Tz + Z + j*tz + Tz*j2 * wx2 + Tz*j2 * wy2 + Tz*j2 * wz2 - Z*j2 * wx2 - Z*j2 * wy2 + Z*j2 * wz2 + j3 * tz*wx2 + j3 * tz*wy2 + j3 * tz*wz2 - 2.0 * X*j*wy + 2.0 * Y*j*wx + 2.0 * X*j2 * wxz + 2.0 * Y*j2 * wyz) / (j2 * wx2 + j2 * wy2 + j2 * wz2 + 1.0);
	double naiveDepth = Z + Tz;
	if (abs((lamda - naiveDepth) / naiveDepth) > 0.1) //very different from the orginal depth
	continue;
	double i = (Tx + X + j*tx + Tx*j2 * wx2 + Tx*j2 * wy2 + Tx*j2 * wz2 + X*j2 * wx2 - X*j2 * wy2 - X*j2 * wz2 + j3 * tx*wx2 + j3 * tx*wy2 + j3 * tx*wz2 - 2.0 * Y*j*wz + 2.0 * Z*j*wy + 2.0 * Y*j2 * wxy + 2.0 * Z*j2 * wxz) / (Tz + Z + j*tz + Tz*j2 * wx2 + Tz*j2 * wy2 + Tz*j2 * wz2 - Z*j2 * wx2 - Z*j2 * wy2 + Z*j2 * wz2 + j3 * tz*wx2 + j3 * tz*wy2 + j3 * tz*wz2 - 2.0 * X*j*wy + 2.0 * Y*j*wx + 2.0 * X*j2 * wxz + 2.0 * Y*j2 * wyz);

	Point2d uv(intrinsic[0] * i + intrinsic[2] * j + intrinsic[3], intrinsic[1] * j + intrinsic[4]);
	if (uv.x < 0 || uv.x > width - 1 || uv.y < 0 || uv.y > height - 1)
	continue;
	else
	{
	predicted = uv;
	count++;
	}
	}
	return count;
	}
	else*/
	{
		double wx, wy, wz, wx2, wy2, wz2, wxy, wxz, wyz, denum, Rw[9], R[9], tp[3];

		mat_mul(R_global, p, tp, 3, 3, 1);
		tp[1] += Ty, tp[2] += Tz;
		double j = tp[1] / tp[2], j_ = j;

		for (int iter = 0; iter < 10; iter++)
		{
			wx = j*wt[0], wy = j*wt[1], wz = j*wt[2];
			wx2 = wx*wx, wy2 = wy*wy, wz2 = wz*wz, wxz = wx*wz, wxy = wx*wy, wyz = wy*wz;

			denum = 1.0 + wx2 + wy2 + wz2;

			Rw[0] = 1.0 + wx2 - wy2 - wz2, Rw[1] = 2.0 * wxy - 2.0 * wz, Rw[2] = 2.0 * wy + 2.0 * wxz,
				Rw[3] = 2.0 * wz + 2.0 * wxy, Rw[4] = 1.0 - wx2 + wy2 - wz2, Rw[5] = 2.0 * wyz - 2.0 * wx,
				Rw[6] = 2.0 * wxz - 2.0 * wy, Rw[7] = 2.0 * wx + 2.0 * wyz, Rw[8] = 1.0 - wx2 - wy2 + wz2;

			for (int ii = 0; ii < 9; ii++)
				Rw[ii] = Rw[ii] / denum;

			mat_mul(Rw, R_global, R, 3, 3, 3);
			mat_mul(R, p, tp, 3, 3, 1);
			tp[0] += Tx, tp[1] += Ty, tp[2] += Tz;

			j = (tp[1] + j*ty) / (tp[2] + j*tz);
			double change = abs((j - j_) / j_);
			if (change < 1.0e-6)
				break;
			j_ = j;
		}
		double i = (tp[0] + j*tx) / (tp[2] + j*tz);

		Point2d uv(intrinsic[0] * i + intrinsic[2] * j + intrinsic[3], intrinsic[1] * j + intrinsic[4]);
		if (uv.x < 0 || uv.x > width - 1 || uv.y < 0 || uv.y > height - 1)
			return 0;
		else
		{
			predicted = uv;
			return 1;
		}
	}
}
int CayleyReprojectionDebug(double *intrinsic, double* rt, double *wt, Point2d &observed, Point3d Point, int width, int height, double *residuals)
{
	Point2d predicted;
	int count = CayleyProjection(intrinsic, rt, wt, predicted, Point, width, height);
	residuals[0] = predicted.x - observed.x, residuals[1] = predicted.y - observed.y;

	return count;
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
struct DepthImgWarping_SSD_BiLinear_PoseOnly{
	DepthImgWarping_SSD_BiLinear_PoseOnly(Point2i uv, double *RefRayDir_, double *C, double idepth, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb, int imgID) :
		uv(uv), refImg(refImg), C(C), idepth(idepth), nonRefImgs(nonRefImgs), intrinsic(intrinsic), imgSize(imgSize), nchannels(nchannels), isigma(isigma), hb(hb), imgID(imgID)
	{
		boundary = max(imgSize.x, imgSize.y) / 50;
		for (int ii = 0; ii < 3; ii++)
			RefRayDir[ii] = RefRayDir_[ii];//xycn changes for every pixel. Pass by ref does not work
	}
	template <typename T>	bool operator()(const T* const rt1, T* residuals) const
	{
		T XYZ[3];
		for (int ii = 0; ii < 3; ii++)
			XYZ[ii] = (T)(RefRayDir[ii] / idepth + C[ii]); //X = r*d+c

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
	static ceres::CostFunction* Create(Point2i uv, double *RefRayDir, double *C, double idepth, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb, int imgID)
	{
		return (new ceres::AutoDiffCostFunction<DepthImgWarping_SSD_BiLinear_PoseOnly, 3, 6>(new DepthImgWarping_SSD_BiLinear_PoseOnly(uv, RefRayDir, C, idepth, refImg, nonRefImgs, intrinsic, imgSize, nchannels, isigma, hb, imgID)));
	}
private:
	uchar *refImg, *nonRefImgs;
	Point2i uv, imgSize;
	double RefRayDir[3], *C, *intrinsic, isigma, idepth;
	int  nchannels, hb, imgID, boundary;
};
struct DepthImgWarping_SSD_BiLinear_PoseOnly2{
	DepthImgWarping_SSD_BiLinear_PoseOnly2(Point3d &xyz, double *rtRef, Point2d &refuv, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb) :
		xyz(xyz), rtRef(rtRef), refuv(refuv), refImg(refImg), nonRefImgs(nonRefImgs), intrinsic(intrinsic), imgSize(imgSize), nchannels(nchannels), isigma(isigma), hb(hb)
	{
		boundary = max(imgSize.x, imgSize.y) / 50;
	}
	template <typename T>	bool operator()(const T* const rt1, T* residuals) const
	{
		//project to other views
		T tp[3], xcn, ycn, tu, tv, color1[3], color2[3];
		T XYZ[3] = { (T)xyz.x, (T)xyz.y, (T)xyz.z };

		T rtRef_[6];
		for (int ii = 0; ii < 6; ii++) rtRef_[ii] = (T)rtRef[ii];
		ceres::AngleAxisRotatePoint(rtRef_, XYZ, tp);
		tp[0] += rtRef_[3], tp[1] += rtRef_[4], tp[2] += rtRef_[5];
		xcn = tp[0] / tp[2], ycn = tp[1] / tp[2];

		T tu1 = (T)intrinsic[0] * xcn + (T)intrinsic[2] * ycn + (T)intrinsic[3];
		T tv1 = (T)intrinsic[1] * ycn + (T)intrinsic[4];

		ceres::AngleAxisRotatePoint(rt1, XYZ, tp);
		tp[0] += rt1[3], tp[1] += rt1[4], tp[2] += rt1[5];
		xcn = tp[0] / tp[2], ycn = tp[1] / tp[2];

		T tu2 = (T)intrinsic[0] * xcn + (T)intrinsic[2] * ycn + (T)intrinsic[3];
		T tv2 = (T)intrinsic[1] * ycn + (T)intrinsic[4];

		if (tu2<(T)boundary || tu2>(T)(imgSize.x - boundary) || tv2<(T)boundary || tv2>(T)(imgSize.y - boundary))
			for (int ii = 0; ii < nchannels; ii++)
				residuals[ii] = (T)1000;
		else
		{
			BilinearInterpolator interp1(refImg, imgSize.x, imgSize.y, nchannels, 3);
			BilinearInterpolator interp2(nonRefImgs, imgSize.x, imgSize.y, nchannels, 3);

			residuals[0] = (T)0, residuals[1] = (T)0, residuals[2] = (T)0;
			for (int jj = -hb; jj <= hb; jj++)
			{
				for (int ii = -hb; ii <= hb; ii++)
				{
					interp1.Evaluate(tu1 + (T)ii, tv1 + (T)jj, color1);
					interp2.Evaluate(tu2 + (T)ii, tv2 + (T)jj, color2);
					for (int kk = 0; kk < nchannels; kk++)
						residuals[kk] += pow((color1[kk] - color2[kk])* (T)isigma, 2);
				}
			}
			for (int kk = 0; kk < nchannels; kk++)
			{
				if (residuals[kk] < (T)1e-9)
					residuals[kk] = (T)1e-9;
				else
					residuals[kk] = sqrt(residuals[kk]) / (T)((2 * hb + 1) * (2 * hb + 1));
			}
		}

		return true;
	}
	static ceres::CostFunction* Create(Point3d &xyz, double *rtRef, Point2d &refuv, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb)
	{
		return (new ceres::AutoDiffCostFunction<DepthImgWarping_SSD_BiLinear_PoseOnly2, 3, 6>(new DepthImgWarping_SSD_BiLinear_PoseOnly2(xyz, rtRef, refuv, refImg, nonRefImgs, intrinsic, imgSize, nchannels, isigma, hb)));
	}
private:
	uchar *refImg, *nonRefImgs;
	Point3d xyz; Point2d refuv;  Point2i  imgSize;
	double *rtRef, *intrinsic, isigma;
	int  nchannels, hb, boundary;
};
struct DepthImgWarping_SSD_BiLinear_PoseOnly3{
	DepthImgWarping_SSD_BiLinear_PoseOnly3(Point3d &xyz, double *rtRef, Point2d &refuv, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb) :
		xyz(xyz), rtRef(rtRef), refuv(refuv), refImg(refImg), nonRefImgs(nonRefImgs), intrinsic(intrinsic), imgSize(imgSize), nchannels(nchannels), isigma(isigma), hb(hb)
	{
		boundary = max(imgSize.x, imgSize.y) / 50;
	}
	template <typename T>	bool operator()(const T* const rt1, T* residuals) const
	{
		//project to other views
		T tp[3], xcn, ycn, tu, tv, color1[3], color2[3];
		T XYZ[3] = { (T)xyz.x, (T)xyz.y, (T)xyz.z };

		T rtRef_[6];
		ceres::AngleAxisRotatePoint(rt1, XYZ, tp);
		tp[0] += rt1[3], tp[1] += rt1[4], tp[2] += rt1[5];
		xcn = tp[0] / tp[2], ycn = tp[1] / tp[2];

		T tu2 = (T)intrinsic[0] * xcn + (T)intrinsic[2] * ycn + (T)intrinsic[3];
		T tv2 = (T)intrinsic[1] * ycn + (T)intrinsic[4];

		if (tu2<(T)0 || tu2>(T)(imgSize.x - 1) || tv2<(T)0 || tv2>(T)(imgSize.y - 1))
			for (int ii = 0; ii < nchannels; ii++)
				residuals[ii] = (T)1000;
		else
		{
			BilinearInterpolator interp1(refImg, imgSize.x, imgSize.y, nchannels, 3);
			BilinearInterpolator interp2(nonRefImgs, imgSize.x, imgSize.y, nchannels, 3);

			int count = 0;
			residuals[0] = (T)0, residuals[1] = (T)0, residuals[2] = (T)0;
			for (int jj = -hb; jj <= hb; jj++)
			{
				for (int ii = -hb; ii <= hb; ii++)
				{
					T tu2_ = tu2 + (T)ii, tv2_ = tv2 + (T)jj;
					if (tu2_<(T)0 || tu2_>(T)(imgSize.x - 1) || tv2_<(T)0 || tv2_>(T)(imgSize.y - 1))
						continue;
					interp1.Evaluate((T)(refuv.x + ii), (T)(refuv.y + jj), color1);
					interp2.Evaluate(tu2_, tv2_, color2);
					for (int kk = 0; kk < nchannels; kk++)
						residuals[kk] += pow((color1[kk] - color2[kk])* (T)isigma, 2);
					count += nchannels;
				}
			}
			for (int kk = 0; kk < nchannels; kk++)
				if (residuals[kk] < (T)1e-9)
					residuals[kk] = (T)1e-9;
				else
					residuals[kk] = sqrt(residuals[kk]) / (T)(count);
		}

		/*if (tu2<(T)hb || tu2>(T)(imgSize.x - hb - 1) || tv2<(T)hb || tv2>(T)(imgSize.y - hb - 1))
			for (int ii = 0; ii < nchannels; ii++)
			residuals[ii] = (T)1000;
			else
			{
			BilinearInterpolator interp1(refImg, imgSize.x, imgSize.y, nchannels, 3);
			BilinearInterpolator interp2(nonRefImgs, imgSize.x, imgSize.y, nchannels, 3);

			residuals[0] = (T)0, residuals[1] = (T)0, residuals[2] = (T)0;
			for (int jj = -hb; jj <= hb; jj++)
			{
			for (int ii = -hb; ii <= hb; ii++)
			{
			interp1.Evaluate((T)(refuv.x + ii), (T)(refuv.y + jj), color1);
			interp2.Evaluate(tu2 + (T)ii, tv2 + (T)jj, color2);
			for (int kk = 0; kk < nchannels; kk++)
			residuals[kk] += pow((color1[kk] - color2[kk])* (T)isigma, 2);
			}
			}
			for (int kk = 0; kk < nchannels; kk++)
			{
			if (residuals[kk] < (T)1e-9)
			residuals[kk] = (T)1e-9;
			else
			residuals[kk] = sqrt(residuals[kk]) / (T)((2 * hb + 1) * (2 * hb + 1));
			}
			}*/

		return true;
	}
	static ceres::CostFunction* Create(Point3d &xyz, double *rtRef, Point2d &refuv, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb)
	{
		return (new ceres::AutoDiffCostFunction<DepthImgWarping_SSD_BiLinear_PoseOnly3, 3, 6>(new DepthImgWarping_SSD_BiLinear_PoseOnly3(xyz, rtRef, refuv, refImg, nonRefImgs, intrinsic, imgSize, nchannels, isigma, hb)));
	}
private:
	uchar *refImg, *nonRefImgs;
	Point3d xyz; Point2d refuv;  Point2i  imgSize;
	double *rtRef, *intrinsic, isigma;
	int  nchannels, hb, boundary;
};
struct DepthImgWarping_SSD_BiLinear_RefPose{
	DepthImgWarping_SSD_BiLinear_RefPose(Point2i uv, double *RefRayDir_, double *C, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb, int imgID) :
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
			XYZ[ii] = RefRayDir[ii] / idepth[0] + C[ii]; //X = r*d+c

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
		return (new ceres::AutoDiffCostFunction<DepthImgWarping_SSD_BiLinear_RefPose, 3, 1, 6>(new DepthImgWarping_SSD_BiLinear_RefPose(uv, RefRayDir, C, refImg, nonRefImgs, intrinsic, imgSize, nchannels, isigma, hb, imgID)));
	}
private:
	uchar *refImg, *nonRefImgs;
	Point2i uv, imgSize;
	double RefRayDir[3], *C, *intrinsic, isigma;
	int  nchannels, hb, imgID, boundary;
};
struct DepthImgWarping_SSD_BiLinear_NoRefPose {
	DepthImgWarping_SSD_BiLinear_NoRefPose(Point2i uv, double *xycnRef_, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb, int imgID) :
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
		return (new ceres::AutoDiffCostFunction<DepthImgWarping_SSD_BiLinear_NoRefPose, 3, 1, 6, 6>(new DepthImgWarping_SSD_BiLinear_NoRefPose(uv, xycnRef_, refImg, nonRefImgs, intrinsic, imgSize, nchannels, isigma, hb, imgID)));
	}
private:
	uchar *refImg, *nonRefImgs;
	Point2i uv, imgSize;
	double xycnRef[3], *intrinsic, isigma;
	int  nchannels, hb, imgID, boundary;
};
struct DepthImgWarping_SSD_BiLinear_RefPose_Dynamic {
	DepthImgWarping_SSD_BiLinear_RefPose_Dynamic(Point2i uv, double *RefRayDir_, double *C, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb, int imgID) :
		uv(uv), refImg(refImg), C(C), nonRefImgs(nonRefImgs), intrinsic(intrinsic), imgSize(imgSize), nchannels(nchannels), isigma(isigma), hb(hb), imgID(imgID)
	{
		boundary = max(imgSize.x, imgSize.y) / 50;
		for (int ii = 0; ii < 3; ii++)
			RefRayDir[ii] = RefRayDir_[ii];//xycn changes for every pixel. Pass by ref does not work
	}
	template <typename T>  bool operator()(T const* const* Parameters, T* residuals) const
	{
		//Parametes[0][0]: inverse depth for ref, Parameters[1][0..5]: poses for ref, Parameters[2][0..5]: poses for non-ref, Parameters[3][0..1]: photometric compenstation
		T XYZ[3], d = Parameters[0][0];
		for (int ii = 0; ii < 3; ii++)
			XYZ[ii] = RefRayDir[ii] / d + C[ii]; //X = r*d+c

		//project to other views
		T tp[3], xcn, ycn, tu, tv, color[3];
		ceres::AngleAxisRotatePoint(Parameters[1], XYZ, tp);
		tp[0] += Parameters[1][3], tp[1] += Parameters[1][4], tp[2] += Parameters[1][5];
		xcn = tp[0] / tp[2], ycn = tp[1] / tp[2];

		tu = (T)intrinsic[0] * xcn + (T)intrinsic[2] * ycn + (T)intrinsic[3];
		tv = (T)intrinsic[1] * ycn + (T)intrinsic[4];

		BilinearInterpolator interp(nonRefImgs, imgSize.x, imgSize.y, nchannels, 3);
		if (tu<(T)boundary || tu>(T)(imgSize.x - boundary) || tv<(T)boundary || tv>(T)(imgSize.y - boundary))
			for (int ii = 0; ii < nchannels*(2 * hb + 1)*(2 * hb + 1); ii++)
				residuals[ii] = (T)1000;
		else
		{
			int count = 0;
			for (int jj = -hb; jj <= hb; jj++)
			{
				for (int ii = -hb; ii <= hb; ii++)
				{
					interp.Evaluate(tu + (T)ii, tv + (T)jj, color);
					for (int kk = 0; kk < nchannels; kk++)
					{
						uchar refColor = refImg[(uv.x + ii + (uv.y + jj)*imgSize.x)*nchannels + kk];
						residuals[count] = (color[kk] - (T)(double)(int)refColor)* (T)isigma;
						count++;
					}
				}
			}
		}

		return true;
	}

private:
	uchar *refImg, *nonRefImgs;
	Point2i uv, imgSize;
	double RefRayDir[3], *C, *intrinsic, isigma;
	int  nchannels, hb, imgID, boundary;
};
struct DepthImgWarping_SSD_BiLinear_NoRefPose_Dynamic {
	DepthImgWarping_SSD_BiLinear_NoRefPose_Dynamic(Point2i uv, double *xycnRef_, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb, int imgID) :
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
struct DepthImgWarping_SSD_BiCubic_NoRefPose_Dynamic {
	DepthImgWarping_SSD_BiCubic_NoRefPose_Dynamic(Point2i uv, double *xycnRef_, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb, int imgID) :
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
struct DepthImgWarping_SSD_BiLinear_RefPoseCayLey{
	DepthImgWarping_SSD_BiLinear_RefPoseCayLey(Point2i uv, double *RefRayDir_, double *C, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb, int imgID, Point2d corres) :
		uv(uv), refImg(refImg), C(C), nonRefImgs(nonRefImgs), intrinsic(intrinsic), imgSize(imgSize), nchannels(nchannels), isigma(isigma), hb(hb), imgID(imgID), corres(corres)
	{
		boundary = max(imgSize.x, imgSize.y) / 50;
		for (int ii = 0; ii < 3; ii++)
			RefRayDir[ii] = RefRayDir_[ii];//xycn changes for every pixel. Pass by ref does not work
	}
	template <typename T>	bool operator()(const double* const idepth, const double* const rt1, const double *wt1, T* residuals) const
	{
		T XYZ[3];
		for (int ii = 0; ii < 3; ii++)
			XYZ[ii] = RefRayDir[ii] / idepth[0] + C[ii]; //X = r*d+c

		//project to other views
		T tp[3], xcn, ycn, tu, tv, color[3];

		Point2d tuv = uv; Point3d xyz(XYZ[0], XYZ[1], XYZ[2]);
		double rt1_[6] = { rt1[0], rt1[1], rt1[2], rt1[3], rt1[4], rt1[5] };
		double wt1_[6] = { wt1[0], wt1[1], wt1[2], wt1[3], wt1[4], wt1[5] };
		CayleyProjection(intrinsic, rt1_, wt1_, tuv, xyz, imgSize.x, imgSize.y);
		tu = tuv.x, tv = tuv.y;

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

	static ceres::CostFunction* Create(Point2i uv, double *RefRayDir, double *C, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb, int imgID, Point2d corres)
	{
		return (new ceres::NumericDiffCostFunction<DepthImgWarping_SSD_BiLinear_RefPoseCayLey, ceres::CENTRAL, 3, 1, 6, 6>(new DepthImgWarping_SSD_BiLinear_RefPoseCayLey(uv, RefRayDir, C, refImg, nonRefImgs, intrinsic, imgSize, nchannels, isigma, hb, imgID, corres)));
	}
private:
	uchar *refImg, *nonRefImgs;
	Point2i uv, imgSize;
	Point2d corres;
	double RefRayDir[3], *C, *intrinsic, isigma;
	int  nchannels, hb, imgID, boundary;
};
struct AnchorDepthRegularize {
	AnchorDepthRegularize(double *RefRayDir_, double *C, Point3d anchor) : anchor(anchor), C(C)
	{
		for (int ii = 0; ii < 3; ii++)
			RefRayDir[ii] = RefRayDir_[ii];//xycn changes for every pixel. Pass by ref does not work
	}
	template <typename T>	bool operator()(const T* const idepth, T* residuals) 	const
	{
		residuals[0] = (T)RefRayDir[0] / idepth[0] + (T)C[0] - (T)anchor.x;
		residuals[1] = (T)RefRayDir[1] / idepth[0] + (T)C[1] - (T)anchor.y;
		residuals[2] = (T)RefRayDir[2] / idepth[0] + (T)C[2] - (T)anchor.z;

		return true;
	}
	static ceres::CostFunction* Create(double *RefRayDir, double *C, Point3d anchor)
	{
		return (new ceres::AutoDiffCostFunction<AnchorDepthRegularize, 3, 1>(new AnchorDepthRegularize(RefRayDir, C, anchor)));
	}
	static ceres::CostFunction* CreateNumerDiff(double *RefRayDir, double *C, Point3d anchor)
	{
		return (new ceres::NumericDiffCostFunction<AnchorDepthRegularize, ceres::CENTRAL, 3, 1>(new AnchorDepthRegularize(RefRayDir, C, anchor)));
	}
	double RefRayDir[3], *C;
	Point3d anchor;
};
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
struct DepthImgWarpingSmall_SSD_BiLinear_NoRefPose
{
	DepthImgWarpingSmall_SSD_BiLinear_NoRefPose(Point2i uv, double *xycnRef_, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb, int imgID) :
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
		return (new ceres::AutoDiffCostFunction<DepthImgWarpingSmall_SSD_BiLinear_NoRefPose, 3, 1, 6, 6>(new DepthImgWarpingSmall_SSD_BiLinear_NoRefPose(uv, xycnRef, refImg, nonRefImgs, intrinsic, imgSize, nchannels, isigma, hb, imgID)));
	}
	uchar *refImg, *nonRefImgs;
	Point2i uv, imgSize;
	double xycnRef[3], *intrinsic, isigma;
	int  nchannels, hb, imgID, boundary;
};
struct DepthImgWarpingSmall_SSD_BiLinear_RefPose
{
	DepthImgWarpingSmall_SSD_BiLinear_RefPose(Point2i uv, double *RefRayDir_, double *C, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb, int imgID) :
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
		return (new ceres::AutoDiffCostFunction<DepthImgWarpingSmall_SSD_BiLinear_RefPose, 3, 1, 6>(new DepthImgWarpingSmall_SSD_BiLinear_RefPose(uv, RefRayDir, C, refImg, nonRefImgs, intrinsic, imgSize, nchannels, isigma, hb, imgID)));
	}
	uchar *refImg, *nonRefImgs;
	Point2i uv, imgSize;
	double RefRayDir[3], *C, *intrinsic, isigma;
	int  nchannels, hb, imgID, boundary;
};

double DirectAlignmentPyr(char *Path, DirectAlignPara &alignmentParas, vector<ImgData> &allImgs, vector<CameraData> &allCalibInfo, int fixPose, int fixDepth, int pyrID, double metricScale, int SmallAngle = 0, int WriteOutput = 0, int verbose = 0)
{
	char Fname[512];

	double dataWeight = alignmentParas.dataWeight, regIntraWeight = alignmentParas.regIntraWeight, regInterWeight = alignmentParas.regInterWeight, anchorWeight = alignmentParas.anchorWeight;
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
		AssembleP(allCalibInfo[ii].activeK, allCalibInfo[ii].R, allCalibInfo[ii].T, allCalibInfo[ii].P);
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
			int width = allImgs[cidI].imgPyr[pyrID].cols, height = allImgs[cidI].imgPyr[pyrID].rows, boundary = width / 50 + 1, orgWidth = allImgs[0].width;
			for (int jj = boundary; jj < height - boundary; jj++)
			{
				for (int ii = boundary; ii < width - boundary; ii++)
				{
					if (allImgs[cidJ].validPixelsPyr[pyrID][ii + jj*width] == 0)
						continue;

					float gradMag = allImgs[cidJ].gradPyr[pyrID][ii + jj*width];
					double xycnRef[3] = { 0, 0, 0 }, ij[3] = { ii, jj, 1 }, rayDirRef[3];
					double v = allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*width];

					ceres::LossFunction *ColorLoss = new ceres::TukeyLoss(alignmentParas.HuberSizeColor);
					ceres::LossFunction *ScaleColorLoss = new ceres::ScaledLoss(ColorLoss, dataWeight*gradMag, ceres::TAKE_OWNERSHIP);

					if (allCalibInfo[cidI].ShutterModel == 0)
					{
						mat_mul(allCalibInfo[cidJ].activeinvK, ij, xycnRef, 3, 3, 1);
						getRayDir(rayDirRef, allCalibInfo[cidJ].activeinvK, allCalibInfo[cidJ].R, ij);

						double XYZ[3];
						for (int kk = 0; kk < 3; kk++)
							XYZ[kk] = rayDirRef[kk] / v + allCalibInfo[cidJ].C[kk];

						Point2d pt;
						Point3d xyz(XYZ[0], XYZ[1], XYZ[2]);
						ProjectandDistort(xyz, &pt, allCalibInfo[cidI].P, NULL, NULL, 1);
						if (pt.x<hb || pt.x > width - hb - 1 || pt.y< hb || pt.y > height - hb - 1)
							continue;

						if (SmallAngle == 1)
						{
							ceres::CostFunction* cost_function = DepthImgWarpingSmall_SSD_BiLinear_RefPose::Create(Point2i(ii, jj), rayDirRef, allCalibInfo[cidJ].C, allImgs[cidJ].imgPyr[pyrID].data, allImgs[cidI].imgPyr[pyrID].data, allCalibInfo[cidI].activeIntrinsic, Point2i(width, height), allImgs[cidI].nchannels, 1.0 / colorSigma, hb, cidI);
							problem.AddResidualBlock(cost_function, ScaleColorLoss, &allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*width], &allCalibInfo[cidI].rt[0]);
						}
						else
						{
							ceres::CostFunction* cost_function = DepthImgWarping_SSD_BiLinear_RefPose::Create(Point2i(ii, jj), rayDirRef, allCalibInfo[cidJ].C, allImgs[cidJ].imgPyr[pyrID].data, allImgs[cidI].imgPyr[pyrID].data, allCalibInfo[cidI].activeIntrinsic, Point2i(width, height), allImgs[cidI].nchannels, 1.0 / colorSigma, hb, cidI);
							problem.AddResidualBlock(cost_function, ScaleColorLoss, &allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*width], &allCalibInfo[cidI].rt[0]);
						}

						/*int orgJJ = 1.0*jj / allImgs[0].scaleFactor[pyrID], orgII = 1.0*ii / allImgs[0].scaleFactor[pyrID], r = 1.0 / allImgs[0].scaleFactor[pyrID];
						double minRadius = 9e9; int minI, minJ;
						for (int j = -r; j <= r; j++)
						{
						for (int i = -r; i < r; i++)
						{
						if (allImgs[0].anchorUV[orgII + i + (orgJJ + j)*orgWidth] == 1)
						{
						double radius2 = i* i + j*j;
						if (radius2 < minRadius)
						minRadius = radius2, minI = i, minJ = j;
						}
						}
						}

						if (minRadius == 0)
						problem.SetParameterBlockConstant(&allImgs[0].InvDepthPyr[pyrID][ii + jj*width]);
						else if (minRadius < 9e9)
						{
						double weight = exp(-minRadius / 4 / r / r);

						double xycnRef[3] = { 0, 0, 0 }, ij[3] = { ii, jj, 1 }, rayDirRef[3];
						mat_mul(allCalibInfo[0].activeinvK, ij, xycnRef, 3, 3, 1);
						getRayDir(rayDirRef, allCalibInfo[cidJ].activeinvK, allCalibInfo[cidJ].R, ij);

						double XYZ[3], v = allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*width];
						for (int kk = 0; kk < 3; kk++)
						XYZ[kk] = rayDirRef[kk] / v + allCalibInfo[cidJ].C[kk];

						ceres::LossFunction *RegAnchorLoss = NULL;
						ceres::LossFunction *ScaleRegAnchorLoss = new ceres::ScaledLoss(RegAnchorLoss, anchorWeight*weight, ceres::TAKE_OWNERSHIP);

						Point3d xyz = allImgs[cidJ].anchorXYZ[orgII + minI + (orgJJ + minJ)*orgWidth];
						ceres::CostFunction* cost_function = AnchorDepthRegularize::Create(rayDirRef, allCalibInfo[cidJ].C, xyz);
						problem.AddResidualBlock(cost_function, ScaleRegAnchorLoss, &allImgs[0].InvDepthPyr[pyrID][ii + jj*width]);
						}*/
					}
					else
					{
						Point2d uv(ii, jj);
						double R[9], T[3], C[3];
						AssembleRT_RS(uv, allCalibInfo[0].K, allCalibInfo[0].R, allCalibInfo[0].T, allCalibInfo[0].wt, R, T);
						getRayDir(rayDirRef, allCalibInfo[0].invK, R, ij);
						GetCfromT(R, T, C);

						double XYZ[3], v = allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*width];
						for (int kk = 0; kk < 3; kk++)
							XYZ[kk] = rayDirRef[kk] / v + C[kk];

						Point2d pt;
						Point3d xyz(XYZ[0], XYZ[1], XYZ[2]);
						ProjectandDistort(xyz, &pt, allCalibInfo[cidI].P, NULL, NULL, 1);
						if (pt.x<0 || pt.x > width || pt.y< 0 || pt.y > height)
							continue;

						ceres::CostFunction* cost_function = DepthImgWarping_SSD_BiLinear_RefPoseCayLey::Create(Point2i(ii, jj), rayDirRef, C, allImgs[cidJ].imgPyr[pyrID].data, allImgs[cidI].imgPyr[pyrID].data, allCalibInfo[cidI].activeIntrinsic,
							Point2i(width, height), allImgs[cidI].nchannels, 1.0 / colorSigma, hb, cidI, allImgs[cidJ].correspondence[ii + jj*width]);
						problem.AddResidualBlock(cost_function, ScaleColorLoss, &allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*width], &allCalibInfo[cidI].rt[0], &allCalibInfo[cidI].wt[0]);
					}

					if (fixDepth == 1)
						problem.SetParameterBlockConstant(&allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*width]); //depth
					//if (cidJ == 0)
					//	problem.SetParameterBlockConstant(allCalibInfo[cidJ].rt); //pose Ref
					if (fixPose == 1)
						problem.SetParameterBlockConstant(allCalibInfo[cidI].rt); //pose nonRef
					//problem.SetParameterBlockConstant(&allCalibInfo[cidI].photometric[2 * cidJ]); //photometric

					if (lowDepth == -1)
					{
						problem.SetParameterLowerBound(&allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*width], 0, 1e-16); //positive depth
						problem.SetParameterUpperBound(&allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*width], 0, 3); //positive depth
					}
					else
					{
						problem.SetParameterLowerBound(&allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*width], 0, 1.0 / highDepth); //positive depth
						problem.SetParameterUpperBound(&allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*width], 0, 1.0 / lowDepth); //positive depth
					}
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
	for (int r = -range; r <= range; r++)
	for (int c = -range; c <= range; c++)
	if (abs(r) >= range || abs(c) >= range)
	indexList.push_back(Point2i(r, c));

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

	int width = allImgs[0].imgPyr[pyrID].cols, height = allImgs[0].imgPyr[pyrID].rows;
	uchar *resImg = new uchar[width*height* nchannels];
	uchar *synthImg = new uchar[width*height* nchannels];
	//Set up callback to update residual images
	int iter = 0, earlyTermination = 0;
	double oldRes = 9e9, newRes = 9e9;
	class MyCallBack : public ceres::IterationCallback{
	public:
		MyCallBack(std::function<void()> callback, int &iter, int &earlyTermination, int &myVerbose) : callback_(callback), iter(iter), earlyTermination(earlyTermination), myVerbose(myVerbose){}
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
			if (earlyTermination == 0)
				return ceres::SOLVER_CONTINUE;
			else
				return ceres::SOLVER_TERMINATE_SUCCESSFULLY;

		}
		int &iter, &earlyTermination, &myVerbose;
		std::function<void()> callback_;
	};
	auto update_Result = [&]()
	{
		char Fname[512];
		sprintf(Fname, "%s/Level_%d", Path, pyrID); makeDir(Fname);

		for (int cidJ = 0; cidJ < 1; cidJ++)
		{
			int hb = (int)(0.003*allImgs[cidJ].imgPyr[pyrID].cols);
			int widthJ = allImgs[cidJ].imgPyr[pyrID].cols, heightJ = allImgs[cidJ].imgPyr[pyrID].rows, lengthJ = widthJ*heightJ, npts = 0, boundaryJ = widthJ / 50;
			double residuals = 0.0;
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
						double tp[3], xcn, ycn, tu, tv, color[3];

						if (allCalibInfo[cidJ].ShutterModel == 0)
						{
							getRfromr(allCalibInfo[cidJ].rt, allCalibInfo[cidJ].R);
							getRayDir(rayDirRef, allCalibInfo[cidJ].activeinvK, allCalibInfo[cidJ].R, ij);
							GetCfromT(allCalibInfo[cidJ].R, allCalibInfo[cidJ].rt + 3, allCalibInfo[cidJ].C);
							for (int kk = 0; kk < 3; kk++)
								XYZ[kk] = rayDirRef[kk] / d + allCalibInfo[cidJ].C[kk];

							//project to other views
							if (SmallAngle == 1)
							{
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
						}
						else
						{
							Point2d uv(ii, jj);
							double R[9], T[3], C[3];
							AssembleRT_RS(uv, allCalibInfo[0].K, allCalibInfo[0].R, allCalibInfo[0].T, allCalibInfo[0].wt, R, T);
							getRayDir(rayDirRef, allCalibInfo[0].invK, R, ij);
							GetCfromT(R, T, C);

							for (int kk = 0; kk < 3; kk++)
								XYZ[kk] = rayDirRef[kk] / d + C[kk];

							double X = allImgs[cidJ].anchorXYZ[ii + jj*widthJ].x, Y = allImgs[cidJ].anchorXYZ[ii + jj*widthJ].y, Z = allImgs[cidJ].anchorXYZ[ii + jj*widthJ].z;

							Point3d p3d(XYZ[0], XYZ[1], XYZ[2]);
							CayleyProjection(allCalibInfo[cidI].activeIntrinsic, allCalibInfo[cidI].rt, allCalibInfo[cidI].wt, uv, p3d, widthI, heightI);
							tu = uv.x, tv = uv.y;
						}

						if (tu<1 || tu> widthI - 2 || tv<1 || tv>heightI - 2)
							continue;

						BilinearInterpolator interp(allImgs[cidI].imgPyr[pyrID].data, widthI, heightI, nchannels, nchannels);

						int count = 0; double resi = 0;
						for (int j = -hb; j <= hb; j++)
						{
							for (int i = -hb; i <= hb; i++)
							{
								if (tu + i<1 || tu + i> widthI - 2 || tv + j<1 || tv + j>heightI - 2)
									continue;
								interp.Evaluate(tu + i, tv + j, color);
								for (int kk = 0; kk < 3; kk++)
								{
									double fcolor = allCalibInfo[cidI].photometric[2 * cidJ] * color[kk] + allCalibInfo[cidI].photometric[2 * cidJ + 1];
									uchar refColor = allImgs[cidJ].imgPyr[pyrID].data[kk + (ii + i + (jj + j)* widthJ)*nchannels];
									double dif = fcolor - (double)(int)refColor;

									resImg[ii + i + (j + jj) * widthJ + kk*lengthJ] = (uchar)(int)(max(min(128.0 + 5.0*dif, 255.0), 0.0)); //magnify 5x
									synthImg[ii + i + (j + jj) * widthJ + kk*lengthJ] = (uchar)(int)(fcolor + 0.5);
									resi += dif*dif;
								}
								count += 3;
							}
						}
						resi = sqrt(resi / count);
						residuals += resi;
						npts += nchannels;
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
				if (fixPose == 0 && fixDepth == 0)
				{
					sprintf(Fname, "%s/Level_%d/R_%03d_%03d_%03d.png", Path, pyrID, cidJ, cidI, iter), WriteGridToImage(Fname, resImg, widthJ, heightJ, nchannels);
					sprintf(Fname, "%s/Level_%d/S_%03d_%03d_%03d.png", Path, pyrID, cidJ, cidI, iter), WriteGridToImage(Fname, synthImg, widthJ, heightJ, nchannels);
					sprintf(Fname, "%s/pose_%d.txt", Path, cidJ); FILE *fp = fopen(Fname, "a+");
					fprintf(fp, "Iter %d PyrID: %d Cid: %d %.16e %.16e %.16e %.16e %.16e %.16e %.4f %.4f \n", iter, pyrID, cidI, allCalibInfo[cidI].rt[0], allCalibInfo[cidI].rt[1], allCalibInfo[cidI].rt[2], allCalibInfo[cidI].rt[3], allCalibInfo[cidI].rt[4], allCalibInfo[cidI].rt[5], allCalibInfo[cidI].photometric[2 * cidJ], allCalibInfo[cidI].photometric[2 * cidJ + 1]);
					fclose(fp);
				}
			}
			oldRes = newRes;
			newRes = residuals / npts;
			double dif = abs(newRes - oldRes) / oldRes;
			if (dif < 0.005 / pow(2, pyrID))
				earlyTermination = 1;
			sprintf(Fname, "%s/iter_%d_%d.txt", Path, cidJ, pyrID);  FILE *fp = fopen(Fname, "a"); fprintf(fp, "%d %.16e\n", iter, residuals / npts); fclose(fp);
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

	ceres::Solver::Options options;
	//options.update_state_every_iteration = true;
	//options.callbacks.push_back(new MyCallBack(update_Result, iter, earlyTermination, verbose));

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
	MyCallBack *myCallback = new MyCallBack(update_Result, iter, earlyTermination, verbose); myCallback->callback_();

	/*for (int cidJ = 0; cidJ < 1; cidJ++) //for all refImages
	{
	int hb = (int)(0.004*allImgs[cidJ].imgPyr[pyrID].cols);
	hb = hb < 1 ? 1 : hb;

	for (int cidI = 0; cidI < (int)allImgs.size(); cidI++)
	{
	if (cidI == cidJ)
	continue;
	sprintf(Fname, "%s/Res_%d_%d_%.2d.txt", Path, pyrID, cidI, iter);  FILE *fp = fopen(Fname, "w");

	double residuals[3];
	int width = allImgs[cidI].imgPyr[pyrID].cols, height = allImgs[cidI].imgPyr[pyrID].rows, boundary = width / 50 + 1, orgWidth = allImgs[0].width;
	for (int jj = boundary; jj < height - boundary; jj++)
	{
	for (int ii = boundary; ii < width - boundary; ii++)
	{
	if (allImgs[cidJ].validPixelsPyr[pyrID][ii + jj*width] == 0)
	continue;

	float gradMag = allImgs[cidJ].gradPyr[pyrID][ii + jj*width];
	double xycnRef[3] = { 0, 0, 0 }, ij[3] = { ii, jj, 1 }, rayDirRef[3];
	double v = allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*width];

	mat_mul(allCalibInfo[cidJ].activeinvK, ij, xycnRef, 3, 3, 1);
	getRayDir(rayDirRef, allCalibInfo[cidJ].activeinvK, allCalibInfo[cidJ].R, ij);

	double XYZ[3];
	for (int kk = 0; kk < 3; kk++)
	XYZ[kk] = rayDirRef[kk] / v + allCalibInfo[cidJ].C[kk];

	Point2d pt;
	Point3d xyz(XYZ[0], XYZ[1], XYZ[2]);
	ProjectandDistort(xyz, &pt, allCalibInfo[cidI].P, NULL, NULL, 1);
	if (pt.x<hb || pt.x > width - hb - 1 || pt.y< hb || pt.y > height - hb - 1)
	continue;

	ceres::LossFunction *ColorLoss = new ceres::TukeyLoss(alignmentParas.HuberSizeColor);
	ceres::LossFunction *ScaleColorLoss = new ceres::ScaledLoss(ColorLoss, dataWeight*gradMag, ceres::TAKE_OWNERSHIP);

	ceres::CostFunction* cost_function = DepthImgWarping_SSD_BiLinear_RefPose::Create(Point2i(ii, jj), rayDirRef, allCalibInfo[cidJ].C, allImgs[cidJ].imgPyr[pyrID].data, allImgs[cidI].imgPyr[pyrID].data, allCalibInfo[cidI].activeIntrinsic, Point2i(width, height), allImgs[cidI].nchannels, 1.0 / colorSigma, hb, cidI);
	problem.AddResidualBlock(cost_function, ScaleColorLoss, &allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*width], &allCalibInfo[cidI].rt[0]);

	vector<double *> paras; paras.push_back(&allImgs[cidJ].InvDepthPyr[pyrID][ii + jj*width]);
	paras.push_back(&allCalibInfo[cidI].rt[0]);
	cost_function->Evaluate(&paras[0], residuals, NULL);
	fprintf(fp, "%d %d %f %f %f\n", ii, jj, residuals[0], residuals[1], residuals[2]);
	}
	fclose(fp);
	}
	}
	}*/

	//let's normalized everything
	if (lowDepth == -1)
	{
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
	}

	if (WriteOutput == 1)
	{
		printf("Saving data...");
		sprintf(Fname, "%s/pose.txt", Path); FILE *fp = fopen(Fname, "a+");
		for (int cid = 0; cid < (int)allImgs.size(); cid++)
			fprintf(fp, "Cid: %d  PyrID: %d %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %\n", allImgs[cid].frameID, pyrID,
			allCalibInfo[cid].rt[0], allCalibInfo[cid].rt[1], allCalibInfo[cid].rt[2], allCalibInfo[cid].rt[3] * metricScale, allCalibInfo[cid].rt[4] * metricScale, allCalibInfo[cid].rt[5] * metricScale,
			allCalibInfo[cid].wt[0], allCalibInfo[cid].wt[1], allCalibInfo[cid].wt[2], allCalibInfo[cid].wt[3] * metricScale, allCalibInfo[cid].wt[4] * metricScale, allCalibInfo[cid].wt[5] * metricScale);
		fclose(fp);

		for (int cid = 0; cid < 1; cid++)
		{
			int width = allImgs[cid].imgPyr[pyrID].cols, height = allImgs[cid].imgPyr[pyrID].rows, lengthJ = width*height, boundary = width / 50 + 1;
			sprintf(Fname, "%s/invD_%d_%d.dat", Path, cid, pyrID), WriteGridBinary(Fname, allImgs[cid].InvDepthPyr[pyrID], width, height, 1);

			sprintf(Fname, "%s/3d_%d_%d.txt", Path, cid, pyrID); fp = fopen(Fname, "w+");
			for (int jj = boundary; jj < height - boundary; jj++)
			{
				for (int ii = boundary; ii < width - boundary; ii++)
				{
					if (allImgs[cid].validPixelsPyr[pyrID][ii + jj*width] == 1)
					{
						double XYZ[3], rayDir[3], ij[3] = { ii, jj, 1 };
						getRayDir(rayDir, allCalibInfo[cid].activeinvK, allCalibInfo[cid].R, ij);
						double invd = allImgs[cid].InvDepthPyr[pyrID][ii + jj * width];

						for (int kk = 0; kk < 3; kk++)
							XYZ[kk] = (rayDir[kk] / invd + allCalibInfo[cid].C[kk]) * metricScale;
						if (nchannels == 1)
							fprintf(fp, "%.8e %.8e %.8e\n", XYZ[0], XYZ[1], XYZ[2]);
						else
							fprintf(fp, "%.8e %.8e %.8e %d %d %d\n", XYZ[0], XYZ[1], XYZ[2],
							(int)allImgs[cid].imgPyr[pyrID].data[(ii + jj*width)*nchannels + 2], (int)allImgs[cid].imgPyr[pyrID].data[(ii + jj*width)*nchannels + 1], (int)allImgs[cid].imgPyr[pyrID].data[(ii + jj*width)*nchannels]);
					}
				}
			}
			fclose(fp);
		}
	}

	delete[]resImg;
	delete[]synthImg;

	return summary.final_cost;
}
double DirectAlignment(char *Path, DirectAlignPara &alignmentParas, vector<ImgData> &allImgs, vector<CameraData> &allCalibInfo, int smallAngle = 0, double scale = 1.0)
{
	char Fname[512];
	int nscales = 4, //actually 6 scales = 5 down sampled images + org image
		innerIter = 20;

	//find texture region in the refImg and store in the vector
	double meanD = 1.0;
	vector<Point2d>  anchorUV; vector<double> anchorD; vector<int> vpid;
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
			if (allImgs[0].Grad[ii + jj*width] > alignmentParas.gradientThresh)
			allImgs[0].validPixels[ii + jj*width] = 1;*/

			int nPartions = 1, HarrisminDistance = 5;
			vector<Point2f> uv; uv.reserve(100000);
			Mat img; cvtColor(allImgs[cid].color, img, CV_BGR2GRAY);
			BucketGoodFeaturesToTrack(img, uv, nPartions, 100000, 0.001, HarrisminDistance*max(1, width / 1920), 7 * max(1, width / 1920), 0, 0.04);

			for (auto pt : uv)
				allImgs[cid].validPixels[(int)pt.x + (int)(pt.y)*width] = 1;

			/*int i, j, pid; float u, v, s, u2 = 0, v2 = 0, s2, x, y, z;

			sprintf(Fname, "%s/K.txt", Path); FILE *fp = fopen(Fname, "r");
			while (fscanf(fp, "%d %f %f %f %f %f %f ", &pid, &u, &v, &s, &x, &y, &z) != EOF)
			{
			double XYZ[3] = { x, y, z }, rayDir[3], uv1[3] = { u, v, 1 }, depth, depth1, depth2, depth3;
			if (allCalibInfo[0].ShutterModel == 0)
			{
			getRayDir(rayDir, allCalibInfo[0].invK, allCalibInfo[0].R, uv1);
			Point2d pt;  ProjectandDistort(Point3d(x, y, z), &pt, allCalibInfo[0].P, NULL, NULL, 1);

			depth1 = (XYZ[0] - allCalibInfo[0].C[0]) / rayDir[0], depth2 = (XYZ[1] - allCalibInfo[0].C[1]) / rayDir[1], depth3 = (XYZ[2] - allCalibInfo[0].C[2]) / rayDir[2];
			if (abs(rayDir[0]) > abs(rayDir[1]) && abs(rayDir[0] > abs(rayDir[2])))
			depth = depth1;
			else if (abs(rayDir[1]) > abs(rayDir[0]) && abs(rayDir[1] > abs(rayDir[2])))
			depth = depth2;
			else
			depth = depth3;
			}
			else
			{
			Point2d uv(u, v);
			double R[9], T[3], C[3];
			AssembleRT_RS(uv, allCalibInfo[0].K, allCalibInfo[0].R, allCalibInfo[0].T, allCalibInfo[0].wt, R, T);
			getRayDir(rayDir, allCalibInfo[0].invK, R, uv1);
			GetCfromT(R, T, C);

			depth1 = (XYZ[0] - C[0]) / rayDir[0],
			depth2 = (XYZ[1] - C[1]) / rayDir[1],
			depth3 = (XYZ[2] - C[2]) / rayDir[2];
			if (abs(rayDir[0]) > abs(rayDir[1]) && abs(rayDir[0] > abs(rayDir[2])))
			depth = depth1;
			else if (abs(rayDir[1]) > abs(rayDir[0]) && abs(rayDir[1] > abs(rayDir[2])))
			depth = depth2;
			else
			depth = depth3;
			}

			minDepth = min(depth, minDepth);
			maxDepth = max(depth, maxDepth);
			anchorD.push_back(depth);
			anchorUV.push_back(Point2d(u, v));

			i = (int)u, j = int(v), allImgs[0].validPixels[i + j*width] = 1, allImgs[0].anchorUV[i + j*width] = 1, allImgs[0].InvDepth[i + j*width] = 1.0 / depth, allImgs[0].anchorXYZ[i + j*width] = Point3d(x, y, z), allImgs[0].correspondence[i + j*width] = Point2d(u2, v2);
			i = (int)u + 1, j = int(v), allImgs[0].validPixels[i + j*width] = 1, allImgs[0].anchorUV[i + j*width] = 1, allImgs[0].InvDepth[i + j*width] = 1.0 / depth, allImgs[0].anchorXYZ[i + j*width] = Point3d(x, y, z), allImgs[0].correspondence[i + j*width] = Point2d(u2, v2);
			i = (int)u, j = int(v) + 1, allImgs[0].validPixels[i + j*width] = 1, allImgs[0].anchorUV[i + j*width] = 1, allImgs[0].InvDepth[i + j*width] = 1.0 / depth, allImgs[0].anchorXYZ[i + j*width] = Point3d(x, y, z), allImgs[0].correspondence[i + j*width] = Point2d(u2, v2);
			i = (int)u + 1, j = int(v) + 1, allImgs[0].validPixels[i + j*width] = 1, allImgs[0].anchorUV[i + j*width] = 1, allImgs[0].InvDepth[i + j*width] = 1.0 / depth, allImgs[0].anchorXYZ[i + j*width] = Point3d(x, y, z), allImgs[0].correspondence[i + j*width] = Point2d(u2, v2);
			}
			fclose(fp);

			//scale depth to 1 so that init for small translation intialization makes sense
			minDepth = minDepth / 2, maxDepth = maxDepth * 2;
			meanD = MeanArray(anchorD);
			allCalibInfo[0].scale = meanD; minDepth /= meanD, maxDepth /= meanD;
			alignmentParas.lowDepth = minDepth, alignmentParas.highDepth = maxDepth;
			for (size_t ii = 0; ii < anchorD.size(); ii++)
			{
			i = (int)anchorUV[ii].x, j = (int)anchorUV[ii].y, allImgs[0].InvDepth[i + j*width] = meanD / anchorD[ii], allImgs[0].anchorXYZ[i + j*width].x /= meanD, allImgs[0].anchorXYZ[i + j*width].y /= meanD, allImgs[0].anchorXYZ[i + j*width].z /= meanD;
			i = (int)anchorUV[ii].x + 1, j = (int)anchorUV[ii].y, allImgs[0].InvDepth[i + j*width] = meanD / anchorD[ii], allImgs[0].anchorXYZ[i + j*width].x /= meanD, allImgs[0].anchorXYZ[i + j*width].y /= meanD, allImgs[0].anchorXYZ[i + j*width].z /= meanD;
			i = (int)anchorUV[ii].x, j = (int)anchorUV[ii].y + 1, allImgs[0].InvDepth[i + j*width] = meanD / anchorD[ii], allImgs[0].anchorXYZ[i + j*width].x /= meanD, allImgs[0].anchorXYZ[i + j*width].y /= meanD, allImgs[0].anchorXYZ[i + j*width].z /= meanD;
			i = (int)anchorUV[ii].x + 1, j = (int)anchorUV[ii].y + 1, allImgs[0].InvDepth[i + j*width] = meanD / anchorD[ii], allImgs[0].anchorXYZ[i + j*width].x /= meanD, allImgs[0].anchorXYZ[i + j*width].y /= meanD, allImgs[0].anchorXYZ[i + j*width].z /= meanD;
			}

			//re-init invdepth using knn (1-nn) interp of anchor points
			omp_set_num_threads(omp_get_max_threads());
			#pragma omp parallel for
			for (int jj = 0; jj < height; jj++)
			{
			for (int ii = 0; ii < width; ii++)
			{
			double smallestDist = 9e9, interpD = 0;
			for (size_t kk = 0; kk < anchorD.size(); kk++)
			{
			double dist = pow(ii - anchorUV[kk].x, 2) + pow(jj - anchorUV[kk].y, 2);
			if (dist < smallestDist)
			interpD = anchorD[kk], smallestDist = dist;
			}
			allImgs[0].InvDepth[ii + jj*width] = 1.0 / interpD;
			}
			}*/
		}

		BuildDataPyramid(allImgs[0].validPixels, allImgs[0].InvDepth, allImgs[0].Grad, allImgs[0].validPixelsPyr, allImgs[0].InvDepthPyr, allImgs[0].gradPyr, allImgs[0].width, allImgs[0].height, nscales);
	}

	double startTime = omp_get_wtime();
	for (int pyrID = nscales; pyrID >= 0; pyrID--)
	{
		printf("\n\n@ level: %d\n", pyrID);
		for (int cid = 0; cid < (int)allImgs.size(); cid++)
		{
			if (cid > 0)
				continue;
			sprintf(Fname, "%s/Level_%d", Path, pyrID); makeDir(Fname);
			sprintf(Fname, "%s/Level_%d/%03d.png", Path, pyrID, cid); 	imwrite(Fname, allImgs[cid].imgPyr[pyrID]);

			sprintf(Fname, "%s/Level_%d/m_%03d.png", Path, pyrID, cid);
			WriteGridToImage(Fname, allImgs[cid].validPixelsPyr[pyrID], allImgs[cid].imgPyr[pyrID].cols, allImgs[cid].imgPyr[pyrID].rows, 1);

		}

		DirectAlignmentPyr(Path, alignmentParas, allImgs, allCalibInfo, 0, 0, pyrID, meanD, smallAngle, 1, 1);

		if (pyrID != 0)
			for (int cid = 0; cid < 1; cid++)
				UpsamleDepthNN(allImgs[cid].InvDepthPyr[pyrID], allImgs[cid].InvDepthPyr[pyrID - 1], allImgs[cid].validPixelsPyr[pyrID], allImgs[cid].validPixelsPyr[pyrID - 1],
				allImgs[cid].imgPyr[pyrID].cols, allImgs[cid].imgPyr[pyrID].rows, allImgs[cid].imgPyr[pyrID - 1].cols, allImgs[cid].imgPyr[pyrID - 1].rows);
	}
	printf("\n\nTotal time: %.2f\n", omp_get_wtime() - startTime);

	for (int cid = 0; cid < 1; cid++)
		for (int pyrID = 0; pyrID < nscales + 1; pyrID++)
			delete[]allImgs[cid].InvDepthPyr[pyrID], delete[]allImgs[cid].gradPyr[pyrID], delete[]allImgs[cid].validPixelsPyr[pyrID];

	return meanD;
}

double DirectTwoFramesAlignmentPyr(char *Path, DirectAlignPara &alignmentParas, vector<Point2d> &uv, vector<Point3d> &XYZ, ImgData &refImg, ImgData &nonRefImg, CameraData &refCam, CameraData &nonRefCam, int pyrID, int nonRefCid, int verbose = 0)
{
	char Fname[512];

	double dataWeight = alignmentParas.dataWeight, anchorWeight = alignmentParas.anchorWeight;
	double colorSigma = alignmentParas.colorSigma; //expected std of variables (grayscale, mm);
	int nchannels = refImg.nchannels;

	double scale = refImg.scaleFactor[pyrID];
	GetIntrinsicScaled(refCam.intrinsic, refCam.activeIntrinsic, refImg.scaleFactor[pyrID]);
	GetKFromIntrinsic(refCam.activeK, refCam.activeIntrinsic), GetiK(refCam.activeinvK, refCam.activeK);

	GetIntrinsicScaled(nonRefCam.intrinsic, nonRefCam.activeIntrinsic, nonRefImg.scaleFactor[pyrID]);
	GetKFromIntrinsic(nonRefCam.activeK, nonRefCam.activeIntrinsic), GetiK(nonRefCam.activeinvK, nonRefCam.activeK);

	ceres::Problem problem;

	//sprintf(Fname, "%s/Level_%d/%03d.png", Path, pyrID, nonRefCid);  imwrite(Fname, nonRefImg.imgPyr[pyrID]);

	//Data term
	int hb = (int)(0.004*refImg.imgPyr[pyrID].cols);
	hb = hb < 1 ? 1 : hb;
	printf("Using window of %dx%d\n", 2 * hb + 1, 2 * hb + 1);
	int width = refImg.imgPyr[pyrID].cols, height = refImg.imgPyr[pyrID].rows, length = width*height, boundary = width / 50, orgWidth = refImg.width, orgHeight = refImg.height;
	for (size_t kk = 0; kk < XYZ.size(); kk++)
	{
		Point2d suv = uv[kk] * scale;
		if (suv.x<hb + boundary || suv.x>width - hb - 1 - boundary || suv.y<hb + boundary || suv.y > height - hb - 1 - boundary)
			continue;
		int ii = (int)suv.x, jj = (int)suv.y;

		float gradMag = 1;// refImg.gradPyr[pyrID][ii + jj*width];
		ceres::LossFunction *ColorLoss = new ceres::TukeyLoss(alignmentParas.HuberSizeColor);
		ceres::LossFunction *ScaleColorLoss = new ceres::ScaledLoss(ColorLoss, dataWeight*gradMag, ceres::TAKE_OWNERSHIP);
		ceres::CostFunction* cost_function = DepthImgWarping_SSD_BiLinear_PoseOnly3::Create(XYZ[kk], refCam.rt, suv, refImg.imgPyr[pyrID].data, nonRefImg.imgPyr[pyrID].data, nonRefCam.activeIntrinsic, Point2i(width, height), nonRefImg.nchannels, 1.0 / colorSigma, hb);
		problem.AddResidualBlock(cost_function, ScaleColorLoss, &nonRefCam.rt[0]);
	}

	//Set up callback to update residual images
	uchar *resImg = new uchar[width*height* nchannels], *synthImg = new uchar[width*height* nchannels];
	int iter = 0, earlyTermination = 0;
	double oldRes = 9e9, newRes = 9e9;
	class MyCallBack : public ceres::IterationCallback{
	public:
		MyCallBack(std::function<void()> callback, int &iter, int &earlyTermination, int &myVerbose) : callback_(callback), iter(iter), earlyTermination(earlyTermination), myVerbose(myVerbose){}
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
			if (earlyTermination == 0)
				return ceres::SOLVER_CONTINUE;
			else
				return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
		}
		int &iter, &earlyTermination, &myVerbose;
		std::function<void()> callback_;
	};
	auto update_Result = [&]()
	{
		/*if (pyrID == 0)
		{
		sprintf(Fname, "%s/Res_%d_%.2d.txt", Path, pyrID, iter);  FILE *fp = fopen(Fname, "w");
		double residuals[3];
		for (size_t kk = 0; kk < XYZ.size(); kk++)
		{
		Point2d suv = uv[kk] * scale;
		if (suv.x<hb + boundary || suv.x>width - hb - 1 - boundary || suv.y<hb + boundary || suv.y > height - hb - 1 - boundary)
		continue;
		int ii = (int)suv.x, jj = (int)suv.y;

		float gradMag = 1;// refImg.gradPyr[pyrID][ii + jj*width];
		ceres::LossFunction *ColorLoss = new ceres::HuberLoss(alignmentParas.HuberSizeColor);
		ceres::LossFunction *ScaleColorLoss = new ceres::ScaledLoss(ColorLoss, dataWeight*gradMag, ceres::TAKE_OWNERSHIP);
		ceres::CostFunction* cost_function = DepthImgWarping_SSD_BiLinear_PoseOnly3::Create(XYZ[kk], refCam.rt, suv, refImg.imgPyr[pyrID].data, nonRefImg.imgPyr[pyrID].data, nonRefCam.activeIntrinsic, Point2i(width, height), nonRefImg.nchannels, 1.0 / colorSigma, hb);
		problem.AddResidualBlock(cost_function, ScaleColorLoss, &nonRefCam.rt[0]);
		vector<double *> paras; paras.push_back(&nonRefCam.rt[0]);
		cost_function->Evaluate(&paras[0], residuals, NULL);
		fprintf(fp, "%d %f %f %f\n", kk, residuals[0], residuals[1], residuals[2]);
		}
		fclose(fp);
		}*/
		char Fname[512];
		double residuals = 0.0;
		for (int ii = 0; ii < width*height* nchannels; ii++)
			resImg[ii] = (uchar)128, synthImg[ii] = (uchar)128;

		int npts = (int)XYZ.size();
		double tp[3], color1[3], color2[3];

		//sprintf(Fname, "%s/corres_%d_%.2d.txt", Path, pyrID, iter); FILE *fp = fopen(Fname, "w");
		for (size_t kk = 0; kk < XYZ.size(); kk++)
		{
			Point2d suv = uv[kk] * scale;
			if (suv.x<hb + boundary || suv.x>width - hb - 1 - boundary || suv.y<hb + boundary || suv.y > height - hb - 1 - boundary)
				continue;
			int ii = (int)suv.x, jj = (int)suv.y;

			ceres::AngleAxisRotatePoint(nonRefCam.rt, &XYZ[kk].x, tp);
			tp[0] += nonRefCam.rt[3], tp[1] += nonRefCam.rt[4], tp[2] += nonRefCam.rt[5];
			double xcn = tp[0] / tp[2], ycn = tp[1] / tp[2];

			double tu2 = nonRefCam.activeIntrinsic[0] * xcn + nonRefCam.activeIntrinsic[2] * ycn + nonRefCam.activeIntrinsic[3];
			double tv2 = nonRefCam.activeIntrinsic[1] * ycn + nonRefCam.activeIntrinsic[4];
			if (tu2< hb || tu2> width - 1 - hb || tv2< hb || tv2>height - 1 - hb)
				continue;

			//fprintf(fp, "%.2f %.2f %2f %2f\n", suv.x, suv.y, tu2, tv2);
			BilinearInterpolator interp1(refImg.imgPyr[pyrID].data, width, height, nchannels, nchannels),
				interp2(nonRefImg.imgPyr[pyrID].data, width, height, nchannels, nchannels);

			double resi = 0;
			for (int j = -hb; j <= hb; j++)
			{
				for (int i = -hb; i <= hb; i++)
				{
					interp1.Evaluate(suv.x + i, suv.y + j, color1);
					interp2.Evaluate(tu2 + i, tv2 + j, color2);
					for (int kk = 0; kk < 3; kk++)
					{
						double dif = color1[kk] - color2[kk];
						resImg[ii + i + (j + jj) * width + kk*length] = (uchar)(int)(max(min(128.0 + 5.0*dif, 255.0), 0.0)); //magnify 5x
						synthImg[ii + i + (j + jj) * width + kk*length] = (uchar)(int)(color2[kk] + 0.5);
						resi += dif*dif;
					}
				}
			}
			residuals += resi / (2 * hb + 1) / (2 * hb + 1) / nchannels;
		}
		//fclose(fp);

		if (pyrID < 2)
		{
			oldRes = newRes;
			newRes = sqrt(residuals / npts);
			double dif = abs(newRes - oldRes) / oldRes;
			if (dif < 0.005 / pow(2, pyrID))
				earlyTermination = 1;
		}

		sprintf(Fname, "%s/Level_%d/D_R_%03d_%03d_%03d.png", Path, pyrID, 0, nonRefCid, iter), WriteGridToImage(Fname, resImg, width, height, nchannels);
		sprintf(Fname, "%s/Level_%d/D_S_%03d_%03d_%03d.png", Path, pyrID, 0, nonRefCid, iter), WriteGridToImage(Fname, synthImg, width, height, nchannels);
		sprintf(Fname, "%s/iter_%d_%d.txt", Path, nonRefCid, pyrID);  FILE *fp = fopen(Fname, "a"); fprintf(fp, "%d %.16e\n", iter, sqrt(residuals / npts)); fclose(fp);
	};
	if (verbose == 1)
	{
		sprintf(Fname, "%s/iter_%d_%d.txt", Path, nonRefCid, pyrID);
		FILE *fp = fopen(Fname, "w+"); fclose(fp);
	}

	ceres::Solver::Options options;
	/*if (verbose == 1)
	options.update_state_every_iteration = true,
	options.callbacks.push_back(new MyCallBack(update_Result, iter, earlyTermination, verbose));
	*/
	options.num_threads = omp_get_max_threads(); //jacobian eval
	options.num_linear_solver_threads = omp_get_max_threads(); //linear solver
	options.trust_region_strategy_type = ceres::DOGLEG;
	options.linear_solver_type = ceres::DENSE_QR;
	options.use_nonmonotonic_steps = false;
	options.max_num_iterations = pyrID > 1 ? 50 : 200;
	options.minimizer_progress_to_stdout = false;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	if (verbose == 1)
		std::cout << summary.BriefReport() << "\n";

	MyCallBack *myCallback = new MyCallBack(update_Result, iter, earlyTermination, verbose); myCallback->callback_();

	delete[]resImg, delete[]synthImg;

	return summary.final_cost;
}
int DirectTwoFramesAlignment(char *Path, DirectAlignPara &alignmentParas, vector<ImgData> &allImgs, vector<CameraData> &allCalibInfo, int smallAngle = 0, double scale = 1.0)
{
	char Fname[512], Path2[512];

	int images = (int)allImgs.size();
	printf("\n\n\nWorking on (%d, %d, %d)\n", allImgs[images - 2].frameID, allImgs[0].frameID, allImgs[images - 1].frameID);
	sprintf(Path2, "%s/%d", Path, allImgs[0].frameID); makeDir(Fname);

	int nscales = 4, //actually 6 scales = 5 down sampled images + org image
		innerIter = 20;

	vector<Point2d> uv; vector<Point3d>  XYZ;
	double maxDepth = 0, minDepth = 9e9, meanD;
	for (int cid = 0; cid < (int)allImgs.size(); cid++)
	{
		int width = allImgs[cid].width, height = allImgs[cid].height, nchannels = allImgs[cid].nchannels, boundary = width / 50;
		for (int pyrID = 0; pyrID < nscales + 1; pyrID++)
			allImgs[cid].scaleFactor.push_back(1.0 / pow(2, pyrID));

		GaussianBlur(allImgs[cid].color, allImgs[cid].color, Size(5, 5), 0.707);
		buildPyramid(allImgs[cid].color, allImgs[cid].imgPyr, nscales);

		if (cid > 0)
			continue;

		uchar *imgData = allImgs[cid].color.data;
		for (int jj = boundary; jj < height - boundary; jj++)
		{
			for (int ii = boundary; ii < width - boundary; ii++)
			{
				//calculate first order image derivatives: using 1 channel should be enough. This is actually gaussian derivative filter.
				float dx = (float)(int)imgData[(ii + 1)*nchannels + jj*nchannels*width] - (float)(int)imgData[(ii - 1)*nchannels + jj*nchannels*width], //1, 0, --1
					dy = (float)(int)imgData[ii *nchannels + (jj + 1)*nchannels*width] - (float)(int)imgData[ii *nchannels + (jj - 1)*nchannels*width]; //1, 0, -1
				allImgs[cid].Grad[ii + jj*width] = sqrt(dx*dx + dy*dy);
			}
		}

		int i, j, pid, npts = 0; float u, v, s, u2 = 0, v2 = 0, s2, x, y, z;
		meanD = 0;
		sprintf(Fname, "%s/K.txt", Path2); FILE *fp = fopen(Fname, "r");
		while (fscanf(fp, "%d %f %f %f %f %f %f ", &pid, &u, &v, &s, &x, &y, &z) != EOF)
		{
			uv.push_back(Point2d(u, v));
			XYZ.push_back(Point3d(x, y, z));
		}
		fclose(fp);
	}

	for (int pyrID = nscales; pyrID >= 0; pyrID--)
	{
		sprintf(Fname, "%s/Level_%d", Path2, pyrID); makeDir(Fname);
		sprintf(Fname, "%s/Level_%d/%03d.png", Path2, pyrID, 0);  imwrite(Fname, allImgs[0].imgPyr[pyrID]);
	}

	//a better way is to do frame-to-frame tracking and use it to initalize ref-to-frame tracking... maybe not needed for small baseline cases
	for (int fid = 1; fid < allImgs.size(); fid += 2)
	{
		printf("*****@%d (%d)*****\n", allImgs[fid].frameID, fid);
		GetRTFromrt(allCalibInfo[fid].rt, allCalibInfo[fid].R, allCalibInfo[fid].T);
		GetCfromT(allCalibInfo[fid].R, allCalibInfo[fid].T, allCalibInfo[fid].C);

		double startTime = omp_get_wtime();
		for (int pyrID = nscales; pyrID >= 0; pyrID--)
		{
			printf("@ level %d: ", pyrID);
			//sprintf(Fname, "%s/Level_%d/%03d.png", Path2, pyrID, fid);  imwrite(Fname, allImgs[fid].imgPyr[pyrID]);
			DirectTwoFramesAlignmentPyr(Path2, alignmentParas, uv, XYZ, allImgs[0], allImgs[fid], allCalibInfo[0], allCalibInfo[fid], pyrID, fid, 1);
		}

		if (fid + 2 < allImgs.size())
			for (int ii = 0; ii < 6; ii++)
				allCalibInfo[fid + 2].rt[ii] = allCalibInfo[fid].rt[ii];

		printf("Total time: %.2f\n\n", omp_get_wtime() - startTime);
	}
	for (int fid = 2; fid < allImgs.size(); fid += 2)
	{
		printf("*****@%d (%d)*****\n", allImgs[fid].frameID, fid);
		GetRTFromrt(allCalibInfo[fid].rt, allCalibInfo[fid].R, allCalibInfo[fid].T);
		GetCfromT(allCalibInfo[fid].R, allCalibInfo[fid].T, allCalibInfo[fid].C);

		double startTime = omp_get_wtime();
		for (int pyrID = nscales; pyrID >= 0; pyrID--)
		{
			printf("@ level %d: ", pyrID);
			//sprintf(Fname, "%s/Level_%d/%03d.png", Path2, pyrID, fid);  imwrite(Fname, allImgs[fid].imgPyr[pyrID]);
			DirectTwoFramesAlignmentPyr(Path2, alignmentParas, uv, XYZ, allImgs[0], allImgs[fid], allCalibInfo[0], allCalibInfo[fid], pyrID, fid, 1);
		}

		if (fid + 2 < allImgs.size())
			for (int ii = 0; ii < 6; ii++)
				allCalibInfo[fid + 2].rt[ii] = allCalibInfo[fid].rt[ii];

		printf("Total time: %.2f\n\n", omp_get_wtime() - startTime);
	}

	sprintf(Fname, "%s/CamPose_0.txt", Path);  FILE *fp = fopen(Fname, "a+");
	for (int fid = 0; fid < allImgs.size(); fid++)
	{
		fprintf(fp, "%d ", allImgs[fid].frameID);
		for (int ii = 0; ii < 6; ii++)
			fprintf(fp, "%.16f ", allCalibInfo[fid].rt[ii]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	return 1;
}

int PlaneSweepStereo(char *Path, vector<ImgData> &allImgs, vector<CameraData> &allCalibInfo, double minD, double maxD, double scale, int num_label, double lambda, double sigma)
{
	int nimages = allImgs.size();
	GetIntrinsicScaled(allCalibInfo[0].intrinsic, allCalibInfo[0].activeIntrinsic, scale);

	//normalize ref's rt to eye
	double RT[16], iRT[16], RTi[16], RTt[16];
	GetRTFromrt(allCalibInfo[0].rt, allCalibInfo[0].R, allCalibInfo[0].T); AssembleRT(allCalibInfo[0].R, allCalibInfo[0].T, RT);
	RT[12] = 0, RT[13] = 0, RT[14] = 0, RT[15] = 1;
	mat_invert(RT, iRT, 4);

	for (int ii = 0; ii < nimages; ii++)
	{
		GetRTFromrt(allCalibInfo[ii].rt, allCalibInfo[ii].R, allCalibInfo[ii].T);
		AssembleRT(allCalibInfo[0].R, allCalibInfo[0].T, RTi);
		RTi[12] = 0, RTi[13] = 0, RTi[14] = 0, RTi[15] = 1;

		mat_mul(RTi, iRT, RTt, 4, 4, 4);
		for (int jj = 0; jj < 3; jj++)
		{
			for (int kk = 0; kk < 3; kk++)
				allCalibInfo[ii].R[kk + 3 * jj] = RTt[kk + 4 * jj];
			for (int kk = 0; kk < 3; kk++)
				allCalibInfo[ii].T[jj] = RTt[3 + 4 * jj];
		}
		GetrtFromRT(allCalibInfo[ii].rt, allCalibInfo[ii].R, allCalibInfo[ii].T);
	}

	int row = allImgs[0].height*scale, col = allImgs[0].width*scale;
	cout << row << " " << col << endl;

	// set the close end of the depth range to be closer than the min. depth
	double max_inv_depth = 1.0 / minD, min_inv_depth = 1.0 / maxD;
	double min_w = 0.0; // starting inv.depth; 0 means infinity
	double max_w = max_inv_depth*1.1; // you can adjust this scaling factor for depth margin.
	double dw = (max_w - min_w) / (num_label - 1);

	// 3x3 box filter
	float window_[9] = { 1 / 9.0, 1 / 9.0, 1 / 9.0, 1 / 9.0, 1 / 9.0, 1 / 9.0, 1 / 9.0, 1 / 9.0, 1 / 9.0 };
	Mat window = Mat(3, 3, CV_32FC1, window_);

	//prepara data
	Mat rs_rimg, gray;
	Mat *rimg = new Mat[nimages], *gimg = new Mat[nimages], *bimg = new Mat[nimages],
		*gximg = new Mat[nimages], *gyimg = new Mat[nimages];
	for (int i = 0; i < nimages; i++)
	{
		Mat img_dist;
		allImgs[i].color.convertTo(img_dist, CV_32FC3, 1 / 255.0);
		resize(img_dist, rs_rimg, Size(), scale, scale);

		vector<Mat> channels(3);
		split(rs_rimg, channels);
		rimg[i] = Mat(row, col, CV_32FC1), gimg[i] = Mat(row, col, CV_32FC1), bimg[i] = Mat(row, col, CV_32FC1);
		channels[2].copyTo(rimg[i]), channels[1].copyTo(gimg[i]), channels[0].copyTo(bimg[i]);

		cvtColor(rs_rimg, gray, COLOR_BGR2GRAY);
		Sobel(gray, gximg[i], gray.depth(), 1, 0, 1, 1 / 2.0);
		Sobel(gray, gyimg[i], gray.depth(), 0, 1, 1, 1 / 2.0);
	}

	Mat u0 = Mat(row, col, CV_32FC1), v0 = Mat(row, col, CV_32FC1);
	Mat x0 = Mat(row, col, CV_32FC1), y0 = Mat(row, col, CV_32FC1);

	// meshgrid
	double _fx = allCalibInfo[0].activeIntrinsic[0], _fy = allCalibInfo[0].activeIntrinsic[1], _cx = allCalibInfo[0].activeIntrinsic[3], _cy = allCalibInfo[0].activeIntrinsic[4];
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			u0.at<float>(i, j) = j, v0.at<float>(i, j) = i;
			x0.at<float>(i, j) = j / _fx - _cx / _fx, y0.at<float>(i, j) = i / _fy - _cy / _fy;
		}
	}
	Mat *rps = new Mat[nimages], *gps = new Mat[nimages], *bps = new Mat[nimages],
		*gxps = new Mat[nimages], *gyps = new Mat[nimages];

	float *confidence = new float[row*col];
	float *cost = new float[row*col];
	float *minconfidence = new float[row*col];
	float *mincost = new float[row*col];
	int *mincostlabel = new int[row*col];
	for (int i = 0; i < row*col; i++)
		mincost[i] = 9e9, mincostlabel[i] = 0;

	printf("Plane sweeping: ");
	omp_set_num_threads(omp_get_max_threads());
	for (int l = 0; l < num_label; l++)
	{
		printf("(%d/%d) ..", l, num_label);

		float w = min_w + float(l)*dw;

		//#pragma omp parallel for 
		for (int n = 0; n < nimages; n++)
		{
			rps[n] = Mat(row, col, CV_32FC1), gps[n] = Mat(row, col, CV_32FC1), bps[n] = Mat(row, col, CV_32FC1);
			gxps[n] = Mat(row, col, CV_32FC1), gyps[n] = Mat(row, col, CV_32FC1);

			float h00 = allCalibInfo[n].R[0], h01 = allCalibInfo[n].R[1], h02 = allCalibInfo[n].R[2] + allCalibInfo[n].rt[3] * w;
			float h10 = allCalibInfo[n].R[3], h11 = allCalibInfo[n].R[4], h12 = allCalibInfo[n].R[5] + allCalibInfo[n].rt[4] * w;
			float h20 = allCalibInfo[n].R[6], h21 = allCalibInfo[n].R[7], h22 = allCalibInfo[n].R[8] + allCalibInfo[n].rt[5] * w;

			Mat x1 = h00*x0 + h01*y0 + h02;
			Mat y1 = h10*x0 + h11*y0 + h12;
			Mat z1 = h20*x0 + h21*y0 + h22;
			x1 = x1.mul(1 / z1), y1 = y1.mul(1 / z1);

			Mat u1 = _fx*x1 + _cx, v1 = _fy*y1 + _cy;

			Mat rwarp, gwarp, bwarp, gxwarp, gywarp;
			remap(rimg[n], rwarp, u1, v1, INTER_CUBIC), remap(gimg[n], gwarp, u1, v1, INTER_CUBIC), remap(bimg[n], bwarp, u1, v1, INTER_CUBIC);
			remap(gximg[n], gxwarp, u1, v1, INTER_CUBIC), remap(gyimg[n], gywarp, u1, v1, INTER_CUBIC);

			rwarp.copyTo(rps[n]), gwarp.copyTo(gps[n]), bwarp.copyTo(bps[n]);
			gxwarp.copyTo(gxps[n]), gywarp.copyTo(gyps[n]);
		}

		Mat rmean = Mat::zeros(row, col, CV_32FC1), gmean = Mat::zeros(row, col, CV_32FC1), bmean = Mat::zeros(row, col, CV_32FC1);
		Mat gxmean = Mat::zeros(row, col, CV_32FC1), gymean = Mat::zeros(row, col, CV_32FC1);
		for (int n = 0; n < nimages; n++)
		{
			rmean = rmean + rps[n], gmean = gmean + gps[n], bmean = bmean + bps[n];
			gxmean = gxmean + gxps[n], gymean = gymean + gyps[n];
		}
		rmean = rmean / nimages, gmean = gmean / nimages, bmean = bmean / nimages, gxmean = gxmean / nimages, gymean = gymean / nimages;

		Mat rvar = Mat::zeros(row, col, CV_32FC1), gvar = Mat::zeros(row, col, CV_32FC1), bvar = Mat::zeros(row, col, CV_32FC1);
		Mat gxvar = Mat::zeros(row, col, CV_32FC1), gyvar = Mat::zeros(row, col, CV_32FC1);
		for (int n = 0; n < nimages; n++)
		{
			Mat rdiff = rps[n] - rmean, gdiff = gps[n] - gmean, bdiff = bps[n] - bmean,
				gxdiff = gxps[n] - gxmean, gydiff = gyps[n] - gymean;

			rvar = rvar + rdiff.mul(rdiff), gvar = gvar + gdiff.mul(gdiff), bvar = bvar + bdiff.mul(bdiff);
			gxvar = gxvar + gxdiff.mul(gxdiff), gyvar = gyvar + gydiff.mul(gydiff);
		}
		rvar = rvar / nimages, gvar = gvar / nimages, bvar = bvar / nimages;
		gxvar = gxvar / nimages, gyvar = gyvar / nimages;

		Mat rgbcost = rvar + gvar + bvar;
		Mat _confidence = Mat(row, col, CV_32F, confidence);
		rmean = rmean + 1e-16; gmean = gmean + 1e-16; bmean = bmean + 1e-16;
		_confidence = 1 - (rvar.mul(1 / rmean) + gvar.mul(1 / gmean) + bvar.mul(1 / bmean));

		Mat gradcost = gxvar + gyvar;
		Mat _cost = Mat(row, col, CV_32F, cost);
		_cost = rgbcost + lambda*gradcost;
		filter2D(_cost, _cost, _cost.depth(), window);

		for (int i = 0; i < row*col; i++)
			if (mincost[i] > cost[i])
				mincost[i] = cost[i], mincostlabel[i] = l, minconfidence[i] = confidence[i];
	}
	printf("\n");
	char Fname[512];
	sprintf(Fname, "%s/%d/costL.dat", Path, allImgs[0].frameID); WriteGridBinary(Fname, mincostlabel, col, row);
	sprintf(Fname, "%s/%d/costC.dat", Path, allImgs[0].frameID); WriteGridBinary(Fname, minconfidence, col, row);

	//ReadGridBinary("C:/temp/costL.dat", mincostlabel, col, row);
	//ReadGridBinary("C:/temp/costC.dat", minconfidence, col, row);

	Mat depthmapWTA = Mat(row, col, CV_8UC1);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			depthmapWTA.at<unsigned char>(i, j) = (unsigned char)(255 * mincostlabel[i*col + j] / float(num_label - 1));
	sprintf(Fname, "%s/%d/WTA.png", Path, allImgs[0].frameID); imwrite(Fname, depthmapWTA);


	/*Mat white = Mat::ones(image_height, image_width, CV_32F);
	Mat white_undist;
	remap(white, white_undist, ud_mapx, ud_mapy, CV_INTER_CUBIC);
	Mat _white_undist;
	resize(white_undist, _white_undist, Size(), scale, scale);
	bool *mask = new bool[row*col];
	int cnt_valid = 0;
	for (int i = 0; i < row; i++)
	for (int j = 0; j<col; j++)
	if (_white_undist.at<float>(i, j)>0.01)
	mask[i*col + j] = true, cnt_valid++;

	Mat _minconfidence = Mat(row, col, CV_32F, minconfidence);
	confidencemap = _minconfidence.clone();

	// statistical filtering
	_minconfidence = Mat(1, row*col, CV_32F, minconfidence);
	cv::sort(_minconfidence, _minconfidence, SORT_ASCENDING);
	float thresh = minconfidence[row*col - int(cnt_valid*0.90)]; // 10 percent

	cout << "confidence threshold: " << thresh << endl;
	double scalar = 255.0 / (num_label - 1);
	double **disparity = new double*[row];
	depthmapFiltered = depthmapWTA.clone();
	for (int i = 0; i < row; i++) {
	disparity[i] = new double[col];
	for (int j = 0; j < col; j++)
	{
	if (mask[i*col + j] == false || confidencemap.at<float>(i, j) < thresh)
	depthmapFiltered.at<unsigned char>(i, j) = (unsigned char)0;
	disparity[i][j] = (double)(depthmapFiltered.at<unsigned char>(i, j)) / scalar;
	if (disparity[i][j] <= 2)
	disparity[i][j] = 0;
	}
	}
	sprintf(Fname, "%s/SMC/Filtered.png", Path);
	SaveDepthmapFiltered(Fname, true);

	// Refinement: Qingxiong Yang's method, A Non-Local Cost Aggregation Method for Stereo Matching, CVPR 2012
	qx_tree_upsampling m_tree_upsampling;//upsampling class
	m_tree_upsampling.init(row, col, num_label - 1, sigma);

	Mat img0_undist;
	remap(images[0], img0_undist, ud_mapx, ud_mapy, CV_INTER_CUBIC);
	unsigned char ***guidance_img_ = new unsigned char**[1];
	guidance_img_[0] = new unsigned char*[1];
	guidance_img_[0][0] = new unsigned char[row*col * 3];
	Mat guidance_img = Mat(row, col, CV_8UC3, guidance_img_[0][0]);
	resize(img0_undist, guidance_img, Size(), scale, scale);
	imshow("guidance_img", guidance_img);
	waitKey(1);
	m_tree_upsampling.build_minimum_spanning_tree(guidance_img_);
	m_tree_upsampling.disparity_upsampling(disparity);

	depthmapRefined = depthmapFiltered.clone();
	for (int i = 0; i < row; i++)
	for (int j = 0; j < col; j++)
	depthmapRefined.at<unsigned char>(i, j) = (unsigned char)(disparity[i][j] * scalar + 0.5);

	sprintf(Fname, "%s/SMC/Refined.png", Path);
	SaveDepthmapRefined(Fname, true);

	Mat img_ref = images[0];
	unsigned char *imageData = (unsigned char*)img_ref.data;
	sprintf(Fname, "%s/SMC/dense_WTA.txt", Path);
	FILE *fp = fopen(Fname, "w");
	for (int i = 0; i < row; i++)
	{
	for (int j = 0; j < col; j++)
	{
	float depth = 1.0f / (min_w + disparity[i][j] * dw);
	if (depth<1.0 / max_inv_depth || depth > 1.0 / min_inv_depth)
	continue;
	double u0_ = j;
	double v0_ = i;
	double x0 = (u0_ - _cx) / _f;
	double y0 = (v0_ - _cy) / _f;

	float X = x0 *depth;
	float Y = y0 *depth;
	float Z = depth;
	fprintf(fp, "%f %f %f ", X, Y, Z);

	// take the RGB intensities of the closest pixel.
	unsigned char R = imageData[2 * i * 2 * col * 3 + 2 * j * 3 + 2];
	unsigned char G = imageData[2 * i * 2 * col * 3 + 2 * j * 3 + 1];
	unsigned char B = imageData[2 * i * 2 * col * 3 + 2 * j * 3 + 0];
	fprintf(fp, "%d %d %d\n", (int)R, int(G), int(B));
	}
	}
	fclose(fp);*/

	return 0;
}
int main(int argc, char** argv)
{
	//srand(time(NULL));
	srand(1);
	char Fname[512], Path[] = "C:/Data/MSR/CMU2";

	if (argc == 1)
		argv[1] = "0";

	int nchannels = 3, refF = 520, rangeF = 15, stepF = 1;
	//for (int ii = 0; ii < 13; ii++)
	//	GetKeyFrameFeature3D(Path, ii);
	//visualizationDriver(Path, 34, 1000, 1, 35);

	vector<int> vrefF; vrefF.push_back(50), vrefF.push_back(110), vrefF.push_back(220), vrefF.push_back(290), vrefF.push_back(320), vrefF.push_back(390), vrefF.push_back(480), vrefF.push_back(520), vrefF.push_back(610), vrefF.push_back(690), vrefF.push_back(750), vrefF.push_back(850), vrefF.push_back(940);
	//for (int ll = 0; ll < vrefF.size(); ll++)
	size_t ll = 7;
	{
		refF = vrefF[ll];

		vector<ImgData> allImgs;
		vector<CameraData> allCalibInfo;
		int mode = 0;
		if (mode == 0)
		{
			double scale = 1.0, DepthLow = -1, DepthHigh = -1;
			double dataWeight = 1.0, regIntraWeight = 0,//atof(argv[1]),
				anchorWeight = atof(argv[1]),
				regInterWeight = 1.0 - dataWeight - regIntraWeight;
			double reprojectionSigma = 0.3, colorSigma = 3.0, depthSigma = 1.0;  //expected std of variables
			double ImGradientThesh = 20;
			DirectAlignPara alignmentParas(dataWeight, anchorWeight, regIntraWeight, regInterWeight, colorSigma, depthSigma, ImGradientThesh, DepthLow, DepthHigh, reprojectionSigma);
			alignmentParas.HuberSizeColor = 30, alignmentParas.removeSmallRegions = false;

			vector<int> allframes;
			allframes.push_back(refF);
			for (int ii = stepF; ii <= rangeF; ii += stepF)
				allframes.push_back(refF + ii), allframes.push_back(refF - ii);
			for (size_t inst = 0; inst < allframes.size(); inst++)
			{
				CameraData camI;
				camI.frameID = allframes[inst];
				camI.intrinsic[0] = 1161.417020, camI.intrinsic[1] = 1163.372023, camI.intrinsic[3] = 634.003916, camI.intrinsic[4] = 478.896843,
					camI.distortion[0] = 0, camI.distortion[1] = 0, camI.distortion[2] = 0, camI.distortion[3] = 0, camI.distortion[4] = 0;	//CMU2
				//camI.intrinsic[0] = 851.687977, camI.intrinsic[1] = 780.464167, camI.intrinsic[3] = 604.448653, camI.intrinsic[4] = 442.588207,
				//camI.distortion[0] = 0.176961, camI.distortion[1] = -0.376240, camI.distortion[2] = 0.242703, camI.distortion[3] = -0.014976, camI.distortion[4] = -0.004914; //Bench--> does not work
				//camI.intrinsic[0] = 1735.130306, camI.intrinsic[1] = camI.intrinsic[0], camI.intrinsic[3] = 960, camI.intrinsic[4] = 540,
				//camI.distortion[0] = 0, camI.distortion[1] = 0, camI.distortion[2] = 0, camI.distortion[3] = 0, camI.distortion[4] = 0; //Bikes--> work
				//camI.intrinsic[0] = 700.0, camI.intrinsic[1] = 700.0, camI.intrinsic[3] = 960, camI.intrinsic[4] = 540, camI.distortion[0] = 0, camI.distortion[1] = 0.0143828495182; //Synth
				//camI.intrinsic[0] = 1961.243258, camI.intrinsic[1] = 1938.422906, camI.intrinsic[3] = 973.725713, camI.intrinsic[4] = 547.547383,
				//	camI.distortion[0] = 0, camI.distortion[1] = 0, camI.distortion[2] = 0, camI.distortion[3] = 0, camI.distortion[4] = 0;	 //Rocks			
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

				double rt[] = { 0.5717444793618730, -1.0552369013033676, -0.1622362500972539, 0.8866640663893104, 0.2830661540846559, -0.2412381164053521,
					0.5439707289839892, -0.9918018462638920, -0.1744879937891410, 0.5556267910628330, 0.2872065245376633, -0.3886801485053389,
					0.5934464450303739, -0.6044898359341377, -0.0764022473040437, -0.2885288623328824, 0.2005524721565697, -0.3294839183923981,
					0.6199084848829047, -0.4533695187083456, -0.0626347164556905, -0.7412611760197863, 0.1458756464702699, -0.3092946690534463,
					0.6604245963717194, -0.3115118331863033, -0.0077831008163147, -1.1447979401871651, 0.0227065628518012, -0.2048998658451439,
					0.6002159759472727, -0.1131474289925313, 0.0585565216827590, -1.5726396206867819, -0.1625434780561931, 0.0147131212166239,
					0.6333911060517488, -0.0038719087538832, 0.0088023975242986, -1.8748837957814821, -0.1505101677988247, 0.1583037534522604,
					0.6182497934358503, 0.2314353613713688, 0.1199830952573363, -2.1391789454085335, -0.4841871953660551, 0.4962443490429812,
					0.6612106073081232, 0.3446883855656200, 0.1224740108052978, -2.2711612489354147, -0.6208284493501075, 0.7309769165448209,
					0.6494299420985926, 0.3639195934601025, 0.1052711391947941, -2.7880089905081222, -0.6029864587362619, 0.7983865718602350,
					0.6017691806594034, 0.5354370095070070, 0.1914090941309303, -2.8038357983721851, -0.9571106529276910, 1.1774980488616098,
					0.6361582285051434, 0.4800228503473724, 0.2042963100776014, -3.3052607669444289, -1.0561993612149889, 1.0556578634226723,
					0.4833682369258264, 0.8293066279269505, 0.2743314477863007, -2.7955452631781674, -1.4948095861984223, 2.1648940613541812 };

				for (int ii = 0; ii < 6; ii++)
					camI.rt[ii] = rt[ii + ll * 6], camI.wt[ii] = 0;

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

			if (allCalibInfo[0].rt[0] == 0)
				InitializeRT(allImgs, allCalibInfo);

			for (int cid = 0; cid < 1; cid++)
			{
				allImgs[cid].correspondence = new Point2d[allImgs[cid].width*allImgs[cid].height];
				allImgs[cid].validPixels = new int[allImgs[cid].width*allImgs[cid].height];
				allImgs[cid].Grad = new float[allImgs[cid].width*allImgs[cid].height];
				allImgs[cid].InvDepth = new double[allImgs[cid].width*allImgs[cid].height];
				allImgs[cid].DepthConf = new double[allImgs[cid].width*allImgs[cid].height];
				allImgs[cid].anchorUV = new int[allImgs[cid].width*allImgs[cid].height];
				allImgs[cid].anchorXYZ = new Point3d[allImgs[cid].width*allImgs[cid].height];

				for (int jj = 0; jj < allImgs[cid].height; jj++)
					for (int ii = 0; ii < allImgs[cid].width; ii++)
						allImgs[cid].validPixels[ii + jj*allImgs[cid].width] = 0, allImgs[cid].Grad[ii + jj*allImgs[cid].width] = 0.0, allImgs[cid].InvDepth[ii + jj*allImgs[cid].width] = 1.0 / min(2.5, max(0.001, gaussian_noise(1.0, 0.5))),
						allImgs[cid].anchorUV[ii + jj*allImgs[cid].width] = 0, allImgs[cid].anchorXYZ[ii + jj*allImgs[cid].width] = Point3d(0, 0, 0);
			}


			/*if (1)
			{
			sprintf(Fname, "%s/CamPose_0.txt", Path);  FILE *fp = fopen(Fname, "r");
			int fid; double rt[6];
			while (fscanf(fp, "%d ", &fid) != EOF)
			{
			for (int ii = 0; ii < 6; ii++)
			fscanf(fp, "%lf ", &rt[ii]);
			for (int jj = 0; jj < allImgs.size(); jj++)
			{
			if (fid == allImgs[jj].frameID)
			{
			for (int ii = 0; ii < 6; ii++)
			allCalibInfo[jj].rt[ii] = rt[ii];
			break;
			}
			}
			}
			fclose(fp);


			double minD = 9e9, maxD = 0;
			int i, j, pid; float u, v, s, u2 = 0, v2 = 0, s2, x, y, z;
			sprintf(Fname, "%s/%d/K.txt", Path, refF); fp = fopen(Fname, "r");
			while (fscanf(fp, "%d %f %f %f %f %f %f ", &pid, &u, &v, &s, &x, &y, &z) != EOF)
			{
			double XYZ[3] = { x, y, z }, rayDir[3], uv1[3] = { u, v, 1 }, depth, depth1, depth2, depth3;
			if (allCalibInfo[0].ShutterModel == 0)
			{
			getRayDir(rayDir, allCalibInfo[0].invK, allCalibInfo[0].R, uv1);
			Point2d pt;  ProjectandDistort(Point3d(x, y, z), &pt, allCalibInfo[0].P, NULL, NULL, 1);

			depth1 = (XYZ[0] - allCalibInfo[0].C[0]) / rayDir[0], depth2 = (XYZ[1] - allCalibInfo[0].C[1]) / rayDir[1], depth3 = (XYZ[2] - allCalibInfo[0].C[2]) / rayDir[2];
			if (abs(rayDir[0]) > abs(rayDir[1]) && abs(rayDir[0] > abs(rayDir[2])))
			depth = depth1;
			else if (abs(rayDir[1]) > abs(rayDir[0]) && abs(rayDir[1] > abs(rayDir[2])))
			depth = depth2;
			else
			depth = depth3;
			}
			else
			{
			Point2d uv(u, v);
			double R[9], T[3], C[3];
			AssembleRT_RS(uv, allCalibInfo[0].K, allCalibInfo[0].R, allCalibInfo[0].T, allCalibInfo[0].wt, R, T);
			getRayDir(rayDir, allCalibInfo[0].invK, R, uv1);
			GetCfromT(R, T, C);

			depth1 = (XYZ[0] - C[0]) / rayDir[0],
			depth2 = (XYZ[1] - C[1]) / rayDir[1],
			depth3 = (XYZ[2] - C[2]) / rayDir[2];
			if (abs(rayDir[0]) > abs(rayDir[1]) && abs(rayDir[0] > abs(rayDir[2])))
			depth = depth1;
			else if (abs(rayDir[1]) > abs(rayDir[0]) && abs(rayDir[1] > abs(rayDir[2])))
			depth = depth2;
			else
			depth = depth3;
			}

			minD = min(depth, minD);
			maxD = max(depth, maxD);
			}
			fclose(fp);

			PlaneSweepStereo(Path, allImgs, allCalibInfo, minD / 2, maxD * 2, 0.5, 512, 0, 0);
			}
			else*/ if (0)
				DirectTwoFramesAlignment(Path, alignmentParas, allImgs, allCalibInfo, 0, scale);
			else
			{
				sprintf(Fname, "%s/%d", Path, refF);
				double metricScale = DirectAlignment(Fname, alignmentParas, allImgs, allCalibInfo, 0, scale);
				sprintf(Fname, "%s/CamPose_0.txt", Path); FILE *fp = fopen(Fname, "w");
				for (int fid = 0; fid < (int)allImgs.size(); fid++)
					fprintf(fp, "%d %.16e %.16e %.16e %.16e %.16e %.16e\n", allImgs[fid].frameID, allCalibInfo[fid].rt[0], allCalibInfo[fid].rt[1], allCalibInfo[fid].rt[2], allCalibInfo[fid].rt[3] * metricScale, allCalibInfo[fid].rt[4] * metricScale, allCalibInfo[fid].rt[5] * metricScale);
				fclose(fp);
			}



			//visualizationDriver(Path, refF - rangeF, refF + rangeF, 1, refF - rangeF);
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