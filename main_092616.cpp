#include "DataStructure.h"
#include "Visualization.h"
#include "DataIO.h"
#include "ImgUlti.h"
#include "MathUlti.h"


#include "ceres/ceres.h"
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
void LensCorrectionPoint(Point2d *uv, double *K, double *distortion, int npts)
{
	double xcn, ycn;
	double Xcn, Ycn, r2, r4, r6, x2, y2, xy, x0, y0;
	double radial, tangential_x, tangential_y, prism_x, prism_y;
	double a0 = distortion[0], a1 = distortion[1], a2 = distortion[2];
	double p0 = distortion[3], p1 = distortion[4];
	double s0 = distortion[5], s1 = distortion[6];

	for (int ii = 0; ii < npts; ii++)
	{
		Ycn = (uv[ii].y - K[5]) / K[4];
		Xcn = (uv[ii].x - K[2] - K[1] * Ycn) / K[0];

		xcn = Xcn, ycn = Ycn;
		for (int k = 0; k < 20; k++)
		{
			x0 = xcn, y0 = ycn;
			r2 = xcn*xcn + ycn*ycn, r4 = r2*r2, r6 = r2*r4, x2 = xcn*xcn, y2 = ycn*ycn, xy = xcn*ycn;

			radial = 1.0 + a0*r2 + a1*r4 + a2*r6;
			tangential_x = 2.0*p1*xy + p0*(r2 + 2.0*x2);
			tangential_y = p1*(r2 + 2.0*y2) + 2.0*p0*xy;

			prism_x = s0*r2;
			prism_y = s1*r2;

			xcn = (Xcn - tangential_x - prism_x) / radial;
			ycn = (Ycn - tangential_y - prism_y) / radial;

			if (abs((xcn - x0) / xcn) < 1.0e-9 && abs((ycn - y0) / ycn) < 1.0e-9)
				break;
		}

		uv[ii].x = K[0] * xcn + K[1] * ycn + K[2];
		uv[ii].y = K[4] * ycn + K[5];
	}

	return;
}
void LensDistortionPoint(Point2d *img_point, double *K, double *distortion, int npts)
{
	for (int ii = 0; ii < npts; ii++)
	{
		double ycn = (img_point[ii].y - K[5]) / K[4];
		double xcn = (img_point[ii].x - K[2] - K[1] * ycn) / K[0];

		double r2 = xcn*xcn + ycn*ycn, r4 = r2*r2, r6 = r2*r4, X2 = xcn*xcn, Y2 = ycn*ycn, XY = xcn*ycn;

		double a0 = distortion[0], a1 = distortion[1], a2 = distortion[2];
		double p0 = distortion[3], p1 = distortion[4];
		double s0 = distortion[5], s1 = distortion[6];

		double radial = 1 + a0*r2 + a1*r4 + a2*r6;
		double tangential_x = 2.0*p1*XY + p0*(r2 + 2.0*X2);
		double tangential_y = p1*(r2 + 2.0*Y2) + 2.0*p0*XY;
		double prism_x = s0*r2;
		double prism_y = s1*r2;

		double xcn_ = radial*xcn + tangential_x + prism_x;
		double ycn_ = radial*ycn + tangential_y + prism_y;

		img_point[ii].x = K[0] * xcn_ + K[1] * ycn_ + K[2];
		img_point[ii].y = K[4] * ycn_ + K[5];
	}

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
	sprintf(Fname, "%s/sb_CamPose_0.txt", Path); FILE *fp = fopen(Fname, "w+");
	for (int cid = 0; cid < (int)allImgs.size(); cid++)
		fprintf(fp, "%d %.16f %.16f %.16f %.16f %.16f %.16f %\n", allImgs[cid].frameID, allCalibInfo[cid].rt[0], allCalibInfo[cid].rt[1], allCalibInfo[cid].rt[2], allCalibInfo[cid].rt[3], allCalibInfo[cid].rt[4], allCalibInfo[cid].rt[5]);
	fclose(fp);

	if (allImgs[0].nchannels == 1)
		sprintf(Fname, "%s/3d.txt", Path);
	else
		sprintf(Fname, "%s/Corpus/n3dGL.txt", Path);
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
			(int)allImgs[0].color.data[(i + j*width) * 3], (int)allImgs[0].color.data[(i + j*width) * 3 + 1], (int)allImgs[0].color.data[(i + j*width) * 3 + 2]);
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
	sprintf(Fname, "%s/D_%d.dat", Path, allImgs[0].frameID), WriteGridBinary(Fname, depthMap, allImgs[0].width, allImgs[0].height, 1);

	printf("Done\n");

	delete[]invD, delete[]depthMap;

	return 0;
}

struct DepthImgWarping_SSD {
	DepthImgWarping_SSD(Point2i uv, double *xycnRef_, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int hb, int imgID) :
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
				residuals[0] = (T)1000;
			else
			{
				residuals[0] = (T)0;
				for (int jj = -hb; jj <= hb; jj++)
				{
					for (int ii = -hb; ii <= hb; ii++)
					{
						Imgs.Evaluate(tv + (T)jj, tu + (T)ii, color);//ceres takes row, column
						uchar refColor = refImg[uv.x + ii + (uv.y + jj)*imgSize.x];
						residuals[0] += Parameters[3][0] * color[0] + Parameters[3][1] - (T)(double)(int)refColor;
					}
				}

				residuals[0] = (T)isigma*residuals[0];
			}
		}
		else
		{
			Grid2D<uchar, 3>  img(nonRefImgs, 0, imgSize.y, 0, imgSize.x);
			BiCubicInterpolator<Grid2D < uchar, 3 > > Imgs(img);

			if (tu<(T)boundary || tu>(T)(imgSize.x - boundary) || tv<(T)boundary || tv>(T)(imgSize.y - boundary))
				for (int jj = 0; jj < nchannels; jj++)
					residuals[jj] = (T)1000;
			else
			{
				residuals[0] = (T)0, residuals[1] = (T)0, residuals[2] = (T)0;
				T meanRef = (T)0, meanTar = (T)0;

				//FILE *fp2 = fopen("C:/temp/tar.txt", "w");
				for (int jj = -hb; jj <= hb; jj++)
				{
					for (int ii = -hb; ii <= hb; ii++)
					{
						Imgs.Evaluate(tv + (T)jj, tu + (T)ii, color);//ceres takes row, column
						for (int kk = 0; kk < nchannels; kk++)
						{
							uchar refColor = refImg[(uv.x + ii + (uv.y + jj)*imgSize.x)*nchannels + kk];
							residuals[kk] += Parameters[3][0] * color[kk] + Parameters[3][1] - (T)(double)(int)refColor;
							//fprintf(fp2, "%.2f ", color[kk]);
						}
					}
					//	fprintf(fp2, "\n");
				}
				//fclose(fp2);
				for (int kk = 0; kk < nchannels; kk++)
					residuals[kk] = (T)isigma*residuals[kk];

				/*residuals[0] = (T)0, residuals[1] = (T)0, residuals[2] = (T)0;
				T meanRef[3] = { (T)0, (T)0, (T)0 }, meanTar[3] = { (T)0, (T)0, (T)0 };

				for (int jj = -hb; jj <= hb; jj++)
				for (int ii = -hb; ii <= hb; ii++)
				for (int kk = 0; kk < nchannels; kk++)
				meanRef[kk] += (T)refImg[(uv.x + ii + (uv.y + jj) *imgSize.x)*nchannels + kk];

				for (int jj = -hb; jj <= hb; jj++)
				{
				for (int ii = -hb; ii <= hb; ii++)
				{
				Imgs.Evaluate(tv + (T)jj, tu + (T)ii, color);//ceres takes row, column
				for (int kk = 0; kk < nchannels; kk++)
				meanTar[kk] += (T)color[kk];
				}
				}

				T num[3] = { (T)0, (T)0, (T)0 }, denum1[3] = { (T)0, (T)0, (T)0 }, denum2[3] = { (T)0, (T)0, (T)0 };
				for (int jj = -hb; jj <= hb; jj++)
				{
				for (int ii = -hb; ii <= hb; ii++)
				{
				Imgs.Evaluate(tv + (T)jj, tu + (T)ii, color);//ceres takes row, column
				for (int kk = 0; kk < nchannels; kk++)
				{
				T d1 = (T)refImg[(uv.x + ii + (uv.y + jj) *imgSize.x)*nchannels + kk] - meanRef[kk],
				d2 = color[kk] - meanTar[kk];
				num[kk] += d1* d2;
				denum1[kk] += d1*d1, denum2[kk] += d2*d2;
				}
				}
				}
				for (int kk = 0; kk < nchannels; kk++)
				residuals[kk] = num[kk] / sqrt(denum1[kk] * denum2[kk]);*/
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
		residuals[0] = (T)isigma*num / (denum + (T) 1.0e-16); //1/id1 - 1/id2 ~ (id2 - id1)/(id1*id2 + esp)
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
		hb = 0;
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

		if (nchannels == 1)
		{
			Grid2D<uchar, 1>  img(nonRefImgs, 0, imgSize.y, 0, imgSize.x);
			BiCubicInterpolator<Grid2D < uchar, 1 > > Imgs(img);

			if (tu<(T)boundary || tu>(T)(imgSize.x - boundary) || tv<(T)boundary || tv>(T)(imgSize.y - boundary))
				residuals[0] = (T)1000;
			else
			{
				T ssd = (T)0;
				for (int jj = -hb; jj <= hb; jj++)
				{
					for (int ii = -hb; ii <= hb; ii++)
					{
						Imgs.Evaluate(tv + (T)jj, tu + (T)ii, color);//ceres takes row, column
						uchar refColor = refImg[uv.x + ii + (uv.y + jj)*imgSize.x];
						ssd += Parameters[3][0] * color[0] + Parameters[3][1] - (T)(double)(int)refColor;
					}
				}

				residuals[0] = (T)isigma*ssd / (T)((2 * hb + 1)*(2 * hb + 1));
			}
		}
		else
		{
			Grid2D<uchar, 3>  img(nonRefImgs, 0, imgSize.y, 0, imgSize.x);
			BiCubicInterpolator<Grid2D < uchar, 3 > > Imgs(img);

			if (tu<(T)boundary || tu>(T)(imgSize.x - boundary) || tv<(T)boundary || tv>(T)(imgSize.y - boundary))
				for (int jj = 0; jj < nchannels; jj++)
					residuals[jj] = (T)1000;
			else
			{
				residuals[0] = (T)0, residuals[1] = (T)0, residuals[2] = (T)0;
				//T meanRef = (T)0, meanTar = (T)0;
				//FILE *fp2 = fopen("C:/temp/tar.txt", "w");
				for (int jj = -hb; jj <= hb; jj++)
				{
					for (int ii = -hb; ii <= hb; ii++)
					{
						Imgs.Evaluate(tv + (T)jj, tu + (T)ii, color);//ceres takes row, column
						for (int kk = 0; kk < nchannels; kk++)
						{
							uchar refColor = refImg[(uv.x + ii + (uv.y + jj)*imgSize.x)*nchannels + kk];
							residuals[kk] += Parameters[3][0] * color[kk] + Parameters[3][1] - (T)(double)(int)refColor;
							//	fprintf(fp2, "%.2f ", color[kk]);
						}
					}
					//fprintf(fp2, "\n");
				}
				//	fclose(fp2);
				for (int kk = 0; kk < nchannels; kk++)
					residuals[kk] = residuals[kk] * (T)isigma / (T)((2 * hb + 1)*(2 * hb + 1));
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
double DirectAlignmentPyr(char *Path, DirectAlignPara &alignmentParas, vector<ImgData> &allImgs, vector<CameraData> &allCalibInfo, int fixPose, int fixDepth, int pyrID, int SmallAngle = 0, int WriteOutput = 0, int verbose = 0, int HuberSize = 3)
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
		for (int ii = 0; ii < width*height; ii++)
			Grad2[ii] = 0.0;
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

		//remove disconnected small segments
		bool *processedPixels = new bool[width*height];
		for (int ii = 0; ii < width*height; ii++)
			processedPixels[ii] = false;
		double factor = allImgs[cid].scaleFactor[pyrID];
		for (int jj = boundary; jj < height - boundary; jj++)
		{
			for (int ii = boundary; ii < width - boundary; ii++)
			{
				float mag2 = Grad2All[cid][ii + jj*width], depth = allImgs[cid].InvDepthPyr[pyrID][ii + jj*width];
				if (!processedPixels[ii + jj*width] && mag2 > mag2Thresh)// && (pyrID > 3 || (depth > lowDepth - 1e-4 && depth < highDepth + 1e-4)))
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
									mag2 = Grad2All[cid][x + y*width], depth = allImgs[cid].InvDepthPyr[pyrID][x + y*width];
									bool boolean = processedPixels[x + y*width];
									if (!boolean && mag2 > mag2Thresh)///&& (pyrID > 3 || (depth > lowDepth - 1e-4 && depth < highDepth + 1e-4)))
										segQueue.push_back(Point2i(x, y)), processedQueue.push_back(false), processedPixels[x + y*width] = true;
								}
							}
						}
						if ((int)segQueue.size() - npts_before == 0)
							break;
					}

					if (segQueue.size() > (int)(100.0 * factor))//10x10 @ 1920x1080
					{
						for (int kk = 0; kk < (int)segQueue.size(); kk++)
						{
							int x = segQueue[kk].x, y = segQueue[kk].y;
							validPixels[x + y*width] = true;
						}
					}
				}
			}
		}
		delete[]processedPixels;

		for (int jj = boundary; jj < height - boundary; jj++)
		{
			for (int ii = boundary; ii < width - boundary; ii++)
			{
				float mag2 = Grad2All[cid][ii + jj*width], depth = allImgs[cid].InvDepthPyr[pyrID][ii + jj*width];
				if (validPixels[ii + jj*width])
					indIAll[cid].push_back(ii), indJAll[cid].push_back(jj), invDAll[cid].push_back(1.0 / depth), GradAll[cid].push_back(sqrt(mag2)), sub2ind[ii + jj*width] = (int)indIAll[cid].size() - 1;
			}
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
					int *nn = new int[8];
					nn[count] = sub2indAll[cid][ii + jj*width], count++;
					if (validPixelsAll[cid][ii - 1 + jj*width])
						nn[count] = sub2indAll[cid][ii - 1 + jj*width], count++;
					if (validPixelsAll[cid][ii + 1 + jj*width])
						nn[count] = sub2indAll[cid][ii + 1 + jj*width], count++;
					if (validPixelsAll[cid][ii + (jj - 1)*width])
						nn[count] = sub2indAll[cid][ii + (jj - 1)*width], count++;
					if (validPixelsAll[cid][ii + (jj + 1)*width])
						nn[count] = sub2indAll[cid][ii + (jj + 1)*width], count++;
					if (validPixelsAll[cid][ii - 1 + (jj - 1)*width])
						nn[count] = sub2indAll[cid][ii - 1 + (jj - 1)*width], count++;
					if (validPixelsAll[cid][ii + 1 + (jj - 1)*width])
						nn[count] = sub2indAll[cid][ii + 1 + (jj - 1)*width], count++;
					if (validPixelsAll[cid][ii - 1 + (jj + 1)*width])
						nn[count] = sub2indAll[cid][ii - 1 + (jj + 1)*width], count++;
					if (validPixelsAll[cid][ii + 1 + (jj + 1)*width])
						nn[count] = sub2indAll[cid][ii + 1 + (jj + 1)*width], count++;

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

	if (pyrID == 5 && (highDepth > 0 && lowDepth > 0))
	{
		//Bruteforce
		int width = allImgs[0].imgPyr[pyrID].cols, height = allImgs[0].imgPyr[pyrID].rows;

		Grid2D<uchar, 1>  img1(allImgs[0].imgPyr[pyrID].data, 0, height, 0, width);
		BiCubicInterpolator<Grid2D < uchar, 1 > > imgInterp1(img1);

		Grid2D<uchar, 3>  img3(allImgs[0].imgPyr[pyrID].data, 0, height, 0, width);
		BiCubicInterpolator<Grid2D < uchar, 3 > > imgInterp3(img3);

		double color[3], dcolorX[3], dcolorY[3];
		Point2f *refImgGrad = new Point2f[width*height*nchannels];
		for (int jj = 0; jj < height; jj++)
		{
			for (int ii = 0; ii < width; ii++)
			{
				imgInterp3.Evaluate(jj, ii, color, dcolorY, dcolorX);//ceres takes row, column
				for (int kk = 0; kk < 3; kk++)
					refImgGrad[(ii + jj*width)*nchannels + kk].x = dcolorX[kk], refImgGrad[(ii + jj*width)*nchannels + kk].y = dcolorY[kk];
			}
		}

		omp_set_num_threads(omp_get_max_threads());

		int hb = (int)min(1.0, 2.0* allImgs[0].scaleFactor[pyrID]), nsteps = 500, count = 0, per = 10;
		double step = (1.0 / lowDepth - 1.0 / highDepth) / nsteps;
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

	//int i = indIAll[0][ii], j = indJAll[0][ii];
	//FILE *	fp1 = fopen("C:/temp/src.txt", "w+");
	//for (int jj = -hb; jj <= hb; jj++)
	//{
	//for (int ii = -hb; ii <= hb; ii++)
	//for (int kk = 0; kk < nchannels; kk++)
	//fprintf(fp1, "%d ", (int)allImgs[cidJ].imgPyr[pyrID].data[(i + ii + (j + jj)*width)*nchannels + kk]);
	//fprintf(fp1, "\n");
	//}
	//fclose(fp1);

	sprintf(Fname, "C:/temp/costD_%d.txt", pyrID);  FILE *fp = fopen(Fname, "w+");
#pragma omp parallel for schedule(dynamic,1)
	for (int kk = 0; kk < nsteps; kk++)
	{
		double residuals = 0;
		double invD = step*kk + 1.0 / highDepth;
		int i = indIAll[0][ii], j = indJAll[0][ii];
		double xycnRef[3] = { 0, 0, 0 }, ij[3] = { i, j, 1 };
		mat_mul(allCalibInfo[0].activeinvK, ij, xycnRef, 3, 3, 1);

		for (int cid = 1; cid < (int)allImgs.size(); cid++)
		{
			ceres::DynamicAutoDiffCostFunction<DepthImgWarping_SSD, 4> *cost_function = new ceres::DynamicAutoDiffCostFunction < DepthImgWarping_SSD, 4 >
				(new DepthImgWarping_SSD(Point2i(i, j), xycnRef, allImgs[0].imgPyr[pyrID].data, allImgs[cid].imgPyr[pyrID].data, allCalibInfo[cid].activeIntrinsic, Point2i(width, height), nchannels, 1.0 / colorSigma, hb, cid));
			cost_function->SetNumResiduals(nchannels);

			//ceres::DynamicAutoDiffCostFunction<DepthImgWarping_SSD2, 4> *cost_function = new ceres::DynamicAutoDiffCostFunction < DepthImgWarping_SSD2, 4 >
			//(new DepthImgWarping_SSD2(Point2i(i, j), xycnRef, allImgs[0].imgPyr[pyrID].data, refImgGrad, allImgs[cid].imgPyr[pyrID].data, allCalibInfo[cid].activeIntrinsic, Point2i(width, height), nchannels, 1.0 / colorSigma, 1.0 / colorSigma, hb, cid));
			//ceres::DynamicNumericDiffCostFunction<DepthImgWarping_SSD2, ceres::CENTRAL> *cost_function = new ceres::DynamicNumericDiffCostFunction<DepthImgWarping_SSD2, ceres::CENTRAL>
			//(new DepthImgWarping_SSD2(Point2i(i, j), xycnRef, allImgs[0].imgPyr[pyrID].data, refImgGrad, allImgs[cid].imgPyr[pyrID].data, allCalibInfo[cid].activeIntrinsic, Point2i(width, height), nchannels, 1.0 / colorSigma, 1.0 / colorSigma, hb, cid));
			//cost_function->SetNumResiduals(nchannels * 3);

			vector<double*> parameter_blocks;
			parameter_blocks.push_back(&invD), cost_function->AddParameterBlock(1);
			parameter_blocks.push_back(allCalibInfo[0].rt), cost_function->AddParameterBlock(6);
			parameter_blocks.push_back(allCalibInfo[cid].rt), cost_function->AddParameterBlock(6);
			parameter_blocks.push_back(&allCalibInfo[cid].photometric[0]), cost_function->AddParameterBlock(2);

			double resi[9];
			cost_function->Evaluate(&parameter_blocks[0], resi, NULL);
			for (int jj = 0; jj < nchannels * 3; jj++)
				residuals += resi[jj] * resi[jj];
			delete[]cost_function;
		}
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
		return 0;
	}

	//dump to Ceres
	ceres::Problem problem;

	//Data term
	//for (int cidJ = 0; cidJ < (int)allImgs.size(); cidJ++) //for all refImages
	{
		int cidJ = 0;
		bool once = true;
		int hb = (int)min(2.0, 2.0* allImgs[0].scaleFactor[pyrID]);
		for (int cidI = 0; cidI < (int)allImgs.size(); cidI++)
		{
			if (cidI == cidJ)
				continue;
			int width = allImgs[cidI].imgPyr[pyrID].cols, height = allImgs[cidI].imgPyr[pyrID].rows;

			for (int ii = 0; ii < (int)invDAll[cidJ].size(); ii++)
			{
				int i = indIAll[cidJ][ii], j = indJAll[cidJ][ii];
				float gradMag = GradAll[cidJ][ii];
				double xycnRef[3] = { 0, 0, 0 }, ij[3] = { i, j, 1 };
				mat_mul(allCalibInfo[cidJ].activeinvK, ij, xycnRef, 3, 3, 1);

				ceres::LossFunction *ColorLoss = new ceres::HuberLoss(HuberSize);
				ceres::LossFunction *ScaleColorLoss = new ceres::ScaledLoss(ColorLoss, dataWeight*gradMag, ceres::TAKE_OWNERSHIP);

				if (SmallAngle == 1)
				{
					ceres::DynamicAutoDiffCostFunction<DepthImgWarpingSmall, 4> *cost_function = new ceres::DynamicAutoDiffCostFunction < DepthImgWarpingSmall, 4 >
						(new DepthImgWarpingSmall(Point2i(i, j), xycnRef, allImgs[cidJ].imgPyr[pyrID].data, allImgs[cidI].imgPyr[pyrID].data, allCalibInfo[cidI].activeIntrinsic, Point2i(width, height), allImgs[cidI].nchannels, 1.0 / colorSigma, hb, cidI));

					cost_function->SetNumResiduals(nchannels);

					vector<double*> parameter_blocks;
					parameter_blocks.push_back(&invDAll[cidJ][ii]), cost_function->AddParameterBlock(1);
					parameter_blocks.push_back(allCalibInfo[cidJ].rt), cost_function->AddParameterBlock(6);
					parameter_blocks.push_back(allCalibInfo[cidI].rt), cost_function->AddParameterBlock(6);
					parameter_blocks.push_back(&allCalibInfo[cidI].photometric[2 * cidJ]), cost_function->AddParameterBlock(2);
					problem.AddResidualBlock(cost_function, ScaleColorLoss, parameter_blocks);

					if (fixDepth == 1)
						problem.SetParameterBlockConstant(parameter_blocks[0]); //depth
					else
						if (highDepth > 0 && lowDepth > 0)
							problem.SetParameterLowerBound(parameter_blocks[0], 0, 1.0 / highDepth), problem.SetParameterUpperBound(parameter_blocks[0], 0, 1.0 / lowDepth); //bound on depth range
					if (cidJ == 0)
						problem.SetParameterBlockConstant(parameter_blocks[1]); //pose Ref
					if (fixPose == 1)
						problem.SetParameterBlockConstant(parameter_blocks[2]); //pose nonRef
					problem.SetParameterBlockConstant(parameter_blocks[3]); //photometric
				}
				else
				{
					//ceres::DynamicNumericDiffCostFunction<DepthImgWarping_SSD, ceres::CENTRAL> *cost_function = new ceres::DynamicNumericDiffCostFunction<DepthImgWarping_SSD, ceres::CENTRAL>
					//	(new DepthImgWarping_SSD(Point2i(i, j), xycnRef, allImgs[cidJ].imgPyr[pyrID].data, allImgs[cidI].imgPyr[pyrID].data, allCalibInfo[cidI].intrinsic, Point2i(allImgs[cidI].width, allImgs[cidI].height), allImgs[cidI].nchannels, 1.0 / colorSigma, ii, cidI));
					ceres::DynamicAutoDiffCostFunction<DepthImgWarping_SSD, 4> *cost_function = new ceres::DynamicAutoDiffCostFunction < DepthImgWarping_SSD, 4 >
						(new DepthImgWarping_SSD(Point2i(i, j), xycnRef, allImgs[cidJ].imgPyr[pyrID].data, allImgs[cidI].imgPyr[pyrID].data, allCalibInfo[cidI].activeIntrinsic, Point2i(width, height), allImgs[cidI].nchannels, 1.0 / colorSigma, hb, cidI));

					cost_function->SetNumResiduals(nchannels);

					vector<double*> parameter_blocks;
					parameter_blocks.push_back(&invDAll[cidJ][ii]), cost_function->AddParameterBlock(1);
					parameter_blocks.push_back(allCalibInfo[cidJ].rt), cost_function->AddParameterBlock(6);
					parameter_blocks.push_back(allCalibInfo[cidI].rt), cost_function->AddParameterBlock(6);
					parameter_blocks.push_back(&allCalibInfo[cidI].photometric[2 * cidJ]), cost_function->AddParameterBlock(2);
					problem.AddResidualBlock(cost_function, ScaleColorLoss, parameter_blocks);

					if (fixDepth == 1)
						problem.SetParameterBlockConstant(parameter_blocks[0]); //depth
					else
						if (highDepth > 0 && lowDepth > 0)
							problem.SetParameterLowerBound(parameter_blocks[0], 0, 1.0 / highDepth), problem.SetParameterUpperBound(parameter_blocks[0], 0, 1.0 / lowDepth); //bound on depth range
					if (cidJ == 0)
						problem.SetParameterBlockConstant(parameter_blocks[1]); //pose Ref
					if (fixPose == 1)
						problem.SetParameterBlockConstant(parameter_blocks[2]); //pose nonRef
					problem.SetParameterBlockConstant(parameter_blocks[3]); //photometric
					problem.SetParameterLowerBound(parameter_blocks[0], 0, 1e-9); //positive depth

				}
			}
		}
	}
	if (fixDepth == 0)
	{
		//Intra depth regularization term
		//for (int cid = 0; cid < (int)allImgs.size(); cid++)
		{
			int cid = 0;
			double idepthSigma = 1.0 / depthSigma;
			for (int ii = 0; ii < (int)NNAll[cid].size(); ii++)
			{
				for (int jj = 1; jj < (int)nNNAll[cid][ii]; jj++)
				{

					int refi = NNAll[cid][ii][0], refj = NNAll[cid][ii][jj];
					int iref = indIAll[cid][refi], jref = indJAll[cid][refi],
						i = indIAll[cid][refj], j = indJAll[cid][refj], width = allImgs[cid].imgPyr[pyrID].cols;
					double colorDif = 0;
					for (int kk = 0; kk < nchannels; kk++)
						colorDif += (double)((int)allImgs[cid].imgPyr[pyrID].data[kk + (iref + jref* width)*nchannels] - (int)allImgs[cid].imgPyr[pyrID].data[kk + (i + j* width)*nchannels]);
					colorDif /= nchannels;
					double edgePreservingWeight = std::exp(-abs(colorDif) / alignmentParas.colorSigma);

					ceres::LossFunction *RegIntraLoss = new ceres::HuberLoss(3);
					ceres::LossFunction *ScaleRegIntraLoss = new ceres::ScaledLoss(RegIntraLoss, regIntraWeight*edgePreservingWeight, ceres::TAKE_OWNERSHIP);

					ceres::CostFunction* cost_function = IntraDepthRegularize::Create(idepthSigma);
					problem.AddResidualBlock(cost_function, ScaleRegIntraLoss, &invDAll[cid][refi], &invDAll[cid][refj]);
				}
			}
		}
	}
	
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

		//for (int cidJ = 0; cidJ < (int)allImgs.size(); cidJ++)
		{
			int cidJ = 0;
			int widthJ = allImgs[cidJ].imgPyr[pyrID].cols, heightJ = allImgs[cidJ].imgPyr[pyrID].rows, lengthJ = widthJ*heightJ, npts = 0;
			double residuals = 0.0;

			uchar *resImg = new uchar[widthJ*heightJ* nchannels];
			uchar *synthImg = new uchar[widthJ*heightJ* nchannels];
			for (int cidI = 0; cidI < (int)allImgs.size(); cidI++)
			{
				if (cidI == cidJ)
					continue;

				for (int ii = 0; ii < widthJ*heightJ* nchannels; ii++)
					resImg[ii] = (uchar)128, synthImg[ii] = (uchar)128;

				//int myX = 0, myY = 0;
				int widthI = allImgs[cidI].imgPyr[pyrID].cols, heightI = allImgs[cidI].imgPyr[pyrID].rows;
				for (int ii = 0; ii < (int)invDAll[cidJ].size(); ii++)
				{
					//back-project ref depth to 3D
					int i = indIAll[cidJ][ii], j = indJAll[cidJ][ii];
					//if (i == myX && j == myY)
					//	int a = 0;
					double XYZ[3], rayDirRef[3], ij[3] = { i, j, 1 };
					double d = invDAll[cidJ][ii];

					getRfromr(allCalibInfo[cidJ].rt, allCalibInfo[cidJ].R);
					getRayDir(rayDirRef, allCalibInfo[cidJ].activeinvK, allCalibInfo[cidJ].R, ij);
					GetCfromT(allCalibInfo[cidJ].R, allCalibInfo[cidJ].rt + 3, allCalibInfo[cidJ].C);
					for (int jj = 0; jj < 3; jj++)
						XYZ[jj] = rayDirRef[jj] / d + allCalibInfo[cidJ].C[jj];

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

					if (nchannels == 1)
					{
						Grid2D<uchar, 1>  img(allImgs[cidI].imgPyr[pyrID].data, 0, heightI, 0, widthI);
						BiCubicInterpolator<Grid2D < uchar, 1 > > imgInterp(img);

						imgInterp.Evaluate(tv, tu, color);//ceres takes row, column
						double fcolor = allCalibInfo[cidI].photometric[2 * cidJ] * color[0] + allCalibInfo[cidI].photometric[2 * cidJ + 1];
						double dif = fcolor - (double)(int)allImgs[cidJ].imgPyr[pyrID].data[i + j * widthJ];

						resImg[indIAll[cidJ][ii] + indJAll[cidJ][ii] * widthJ] = (uchar)(int)(128.0 + 5.0*dif); //magnify 5x
						synthImg[indIAll[cidJ][ii] + indJAll[cidJ][ii] * widthJ] = (uchar)(int)(fcolor + 0.5);

						residuals += dif*dif;
						npts++;
					}
					else
					{
						Grid2D<uchar, 3>  img(allImgs[cidI].imgPyr[pyrID].data, 0, heightI, 0, widthI);
						BiCubicInterpolator<Grid2D < uchar, 3 > > imgInterp(img);

						imgInterp.Evaluate(tv, tu, color);//ceres takes row, column

						for (int kk = 0; kk < 3; kk++)
						{
							double fcolor = allCalibInfo[cidI].photometric[2 * cidJ] * color[kk] + allCalibInfo[cidI].photometric[2 * cidJ + 1];
							uchar refColor = allImgs[cidJ].imgPyr[pyrID].data[kk + (i + j* widthJ)*nchannels];
							double dif = fcolor - (double)(int)refColor;

							resImg[indIAll[cidJ][ii] + indJAll[cidJ][ii] * widthJ + kk*lengthJ] = (uchar)(int)(128.0 + 5.0*dif); //magnify 5x
							synthImg[indIAll[cidJ][ii] + indJAll[cidJ][ii] * widthJ + kk*lengthJ] = (uchar)(int)(fcolor + 0.5);
							residuals += dif*dif;
						}
						npts += 3;
					}
				}
				sprintf(Fname, "%s/Level_%d/R_%03d_%03d_%03d.png", Path, pyrID, cidJ, cidI, iter), WriteGridToImage(Fname, resImg, widthJ, heightJ, nchannels);
				sprintf(Fname, "%s/Level_%d/S_%03d_%03d_%03d.png", Path, pyrID, cidJ, cidI, iter), WriteGridToImage(Fname, synthImg, widthJ, heightJ, nchannels);

				sprintf(Fname, "%s/pose_%d.txt", Path, cidJ); FILE *fp = fopen(Fname, "a+");
				fprintf(fp, "Iter %d PyrID: %d Cid: %d %.16f %.16f %.16f %.16f %.16f %.16f %.4f %.4f \n", iter, pyrID, cidI, allCalibInfo[cidI].rt[0], allCalibInfo[cidI].rt[1], allCalibInfo[cidI].rt[2], allCalibInfo[cidI].rt[3], allCalibInfo[cidI].rt[4], allCalibInfo[cidI].rt[5], allCalibInfo[cidI].photometric[2 * cidJ], allCalibInfo[cidI].photometric[2 * cidJ + 1]);
				fclose(fp);
			}
			sprintf(Fname, "%s/iter_%d_%d.txt", Path, cidJ, pyrID);  FILE *fp = fopen(Fname, "a"); fprintf(fp, "%d %.16e\n", iter, sqrt(residuals / npts)); fclose(fp);

			delete[]resImg, delete[]synthImg;
		}
	};
	if (verbose == 1)
	{
		//for (int cid = 0; cid < (int)allImgs.size() - 1; cid++)
		{
			int cid = 0;
			sprintf(Fname, "%s/iter_%d_%d.txt", Path, cid, pyrID);
			FILE *fp = fopen(Fname, "w+"); fclose(fp);
		}
	}

	MyCallBack *myCallback = new MyCallBack(update_Result, iter, verbose);
	myCallback->callback_();

	ceres::Solver::Options options;
	options.update_state_every_iteration = true;
	options.callbacks.push_back(new MyCallBack(update_Result, iter, verbose));

	options.num_threads = omp_get_max_threads(); //jacobian eval
	options.num_linear_solver_threads = omp_get_max_threads(); //linear solver
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.linear_solver_type = ceres::CGNR;
	options.preconditioner_type = ceres::JACOBI;
	options.use_inner_iterations = true;
	options.use_nonmonotonic_steps = false;
	options.max_num_iterations = pyrID > 1 ? 50 : 200;
	options.parameter_tolerance = 1.0e-9;
	options.minimizer_progress_to_stdout = verbose == 1 ? true : false;

	ceres::Solver::Summary summary;
	//ceres::Solve(options, &problem, &summary);

	//if (verbose == 1)
	std::cout << "\n" << summary.BriefReport() << "\n";

	printf("Saving data...");
	for (int cid = 0; cid < (int)allImgs.size(); cid++)
	{
		int width = allImgs[cid].imgPyr[pyrID].cols;
		for (int ii = 0; ii < (int)invDAll[cid].size(); ii++)
			allImgs[cid].InvDepthPyr[pyrID][indIAll[cid][ii] + indJAll[cid][ii] * width] = 1.0 / invDAll[cid][ii];
	}

	if (WriteOutput == 1)
	{
		sprintf(Fname, "%s/pose.txt", Path); FILE *fp = fopen(Fname, "a+");
		for (int cid = 0; cid < (int)allImgs.size(); cid++)
			fprintf(fp, "Cid: %d  PyrID: %d %.16f %.16f %.16f %.16f %.16f %.16f %\n", allImgs[cid].frameID, pyrID,
			allCalibInfo[cid].rt[0], allCalibInfo[cid].rt[1], allCalibInfo[cid].rt[2], allCalibInfo[cid].rt[3], allCalibInfo[cid].rt[4], allCalibInfo[cid].rt[5]);
		fclose(fp);

		//for (int cid = 0; cid < (int)allImgs.size(); cid++)
		{
			int cid = 0;
			int width = allImgs[cid].imgPyr[pyrID].cols, height = allImgs[cid].imgPyr[pyrID].rows, lengthJ = width*height;
			float *depthMap = new float[width*height];

			for (int ii = 0; ii < width*height; ii++)
				depthMap[ii] = 0.0f;
			for (int ii = 0; ii < (int)invDAll[cid].size(); ii++)
			{
				int i = indIAll[cid][ii], j = indJAll[cid][ii];
				depthMap[i + j*width] = 1.0 / invDAll[cid][ii];
			}
			sprintf(Fname, "%s/D_%d_%d.dat", Path, cid, pyrID), WriteGridBinary(Fname, allImgs[cid].InvDepthPyr[pyrID], width, height, 1);

			double rayDir[3], opticalAxis[3], ij[3] = { allCalibInfo[cid].activeK[2], allCalibInfo[cid].activeK[5], 1 };
			getRayDir(opticalAxis, allCalibInfo[cid].activeinvK, allCalibInfo[cid].R, ij);
			double normOptical = sqrt(pow(opticalAxis[0], 2) + pow(opticalAxis[1], 2) + pow(opticalAxis[2], 2));

			for (int ii = 0; ii < width*height; ii++)
				depthMap[ii] = 0.f;
			for (int ii = 0; ii < (int)invDAll[cid].size(); ii++)
			{
				int i = indIAll[cid][ii], j = indJAll[cid][ii];
				double ij[3] = { i, j, 1 };
				getRayDir(rayDir, allCalibInfo[cid].activeinvK, allCalibInfo[cid].R, ij);
				double cos = (rayDir[0] * opticalAxis[0] + rayDir[1] * opticalAxis[1] + rayDir[2] * opticalAxis[2]) /
					sqrt(pow(rayDir[0], 2) + pow(rayDir[1], 2) + pow(rayDir[2], 2)) / normOptical;

				double d = allImgs[cid].InvDepthPyr[pyrID][i + j * width];
				if (abs(d) < 1e-16)
					depthMap[j + j * width] = 1e16;
				else
					depthMap[i + j* width] = 1.0 / (cos * d + 1.0e-16);
			}
			sprintf(Fname, "%s/invfrontoD_%d_%d.dat", Path, cid, pyrID), WriteGridBinary(Fname, depthMap, width, height);
			delete[]depthMap;

			if (nchannels == 1)
				sprintf(Fname, "%s/3d_%d_%d.txt", Path, cid, pyrID);
			else
				sprintf(Fname, "%s/3d_%d_%d.txt", Path, cid, pyrID);
			fp = fopen(Fname, "w+");
			for (int ii = 0; ii < (int)invDAll[cid].size(); ii++)
			{
				int i = indIAll[cid][ii], j = indJAll[cid][ii];
				double XYZ[3], rayDir[3], ij[3] = { i, j, 1 };
				getRayDir(rayDir, allCalibInfo[cid].activeinvK, allCalibInfo[cid].R, ij);
				double d = allImgs[cid].InvDepthPyr[pyrID][i + j * width];
				//if (highDepth > 0 && lowDepth > 0)
				//	if (abs(d) < lowDepth || abs(d) > highDepth)
				//		continue;

				for (int kk = 0; kk < 3; kk++)
					XYZ[kk] = rayDir[kk] * d + allCalibInfo[cid].C[kk];
				if (nchannels == 1)
					fprintf(fp, "%.8e %.8e %.8e\n", XYZ[0], XYZ[1], XYZ[2]);
				else
					fprintf(fp, "%.8e %.8e %.8e %d %d %d\n", XYZ[0], XYZ[1], XYZ[2],
					(int)allImgs[cid].imgPyr[pyrID].data[(i + j*width)*nchannels + 2], (int)allImgs[cid].imgPyr[pyrID].data[(i + j*width)*nchannels + 1], (int)allImgs[cid].imgPyr[pyrID].data[(i + j*width)*nchannels]);
			}
			fclose(fp);
		}
	}

	for (int ii = 0; ii < (int)validPixelsAll.size(); ii++)
		delete[] validPixelsAll[ii];
	for (int ii = 0; ii < (int)sub2indAll.size(); ii++)
		delete[] sub2indAll[ii];
	delete[]indIAll, delete[]indJAll, delete[]invDAll, delete[]GradAll;
	for (int ii = 0; ii < (int)Grad2All.size(); ii++)
		delete[]Grad2All[ii];

	return summary.final_cost;
}
int DirectAlignment(char *Path, DirectAlignPara &alignmentParas, vector<ImgData> &allImgs, vector<CameraData> &allCalibInfo, int smallAngle = 0, double scale = 1.0, int HuberSize = 3)
{
	char Fname2[512];  sprintf(Fname2, "%s/%d", Path, HuberSize);
	makeDir(Fname2);
	printf("Current Path: %s\n", Fname2);

	int nscales = 1, //actually 6 scales = 5 down sampled images + org image
		innerIter = 20;

	//find texture region in the refImg and store in the vector
	vector<float *> Grad2All;
	for (int cid = 0; cid < 1; cid++)
	{
		int width = allImgs[cid].width, height = allImgs[cid].height, nchannels = allImgs[cid].nchannels, boundary = width / 50;
		float *Grad2 = new float[width*height], dx, dy;
		for (int ii = 0; ii < width*height; ii++)
			Grad2[ii] = 0.0;
		for (int jj = boundary; jj < height - boundary; jj++)
		{
			for (int ii = boundary; ii < width - boundary; ii++)
			{
				//calculate first order image derivatives: using 1 channel should be enough
				dx = (float)(int)allImgs[cid].color.data[(ii + 1)*nchannels + jj*nchannels*width] - (float)(int)allImgs[cid].color.data[(ii - 1)*nchannels + jj*nchannels*width]; //1, 0, -1
				dy = (float)(int)allImgs[cid].color.data[ii *nchannels + (jj + 1)*nchannels*width] - (float)(int)allImgs[cid].color.data[ii *nchannels + (jj - 1)*nchannels*width]; //1, 0, -1
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
	for (int cid = 0; cid < 1; cid++)
	{
		int width = allImgs[cid].width, height = allImgs[cid].height, nchannels = allImgs[cid].nchannels, boundary = width / 50;
		int *sub2ind = new int[width*height];
		bool *validPixels = new bool[width*height];
		for (int ii = 0; ii < width*height; ii++)
			validPixels[ii] = false, sub2ind[ii] = -1;

		invDAll[cid].reserve(width*height), GradAll[cid].reserve(width*height);
		indIAll[cid].reserve(width*height), indJAll[cid].reserve(width*height);

		//remove disconnected small segments
		bool *processedPixels = new bool[width*height];
		for (int ii = 0; ii < width*height; ii++)
			processedPixels[ii] = false;
		for (int jj = boundary; jj < height - boundary; jj++)
		{
			for (int ii = boundary; ii < width - boundary; ii++)
			{
				float mag2 = Grad2All[cid][ii + jj*width], depth = allImgs[cid].InvDepthPyr[pyrID][ii + jj*width];
				if (!processedPixels[ii + jj*width] && mag2 > mag2Thresh)// && (pyrID > 3 || (depth > lowDepth - 1e-4 && depth < highDepth + 1e-4)))
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
									mag2 = Grad2All[cid][x + y*width], depth = allImgs[cid].InvDepthPyr[pyrID][x + y*width];
									bool boolean = processedPixels[x + y*width];
									if (!boolean && mag2 > mag2Thresh)///&& (pyrID > 3 || (depth > lowDepth - 1e-4 && depth < highDepth + 1e-4)))
										segQueue.push_back(Point2i(x, y)), processedQueue.push_back(false), processedPixels[x + y*width] = true;
								}
							}
						}
						if ((int)segQueue.size() - npts_before == 0)
							break;
					}

					if (segQueue.size() > (int)(100.0kd))//10x10 @ 1920x1080
					{
						for (int kk = 0; kk < (int)segQueue.size(); kk++)
						{
							int x = segQueue[kk].x, y = segQueue[kk].y;
							validPixels[x + y*width] = true;
						}
					}
				}
			}
		}
		delete[]processedPixels;

		for (int jj = boundary; jj < height - boundary; jj++)
		{
			for (int ii = boundary; ii < width - boundary; ii++)
			{
				float mag2 = Grad2All[cid][ii + jj*width], depth = allImgs[cid].InvDepthPyr[pyrID][ii + jj*width];
				if (validPixels[ii + jj*width])
					indIAll[cid].push_back(ii), indJAll[cid].push_back(jj), invDAll[cid].push_back(1.0 / depth), GradAll[cid].push_back(sqrt(mag2)), sub2ind[ii + jj*width] = (int)indIAll[cid].size() - 1;
			}
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
					int *nn = new int[8];
					nn[count] = sub2indAll[cid][ii + jj*width], count++;
					if (validPixelsAll[cid][ii - 1 + jj*width])
						nn[count] = sub2indAll[cid][ii - 1 + jj*width], count++;
					if (validPixelsAll[cid][ii + 1 + jj*width])
						nn[count] = sub2indAll[cid][ii + 1 + jj*width], count++;
					if (validPixelsAll[cid][ii + (jj - 1)*width])
						nn[count] = sub2indAll[cid][ii + (jj - 1)*width], count++;
					if (validPixelsAll[cid][ii + (jj + 1)*width])
						nn[count] = sub2indAll[cid][ii + (jj + 1)*width], count++;
					if (validPixelsAll[cid][ii - 1 + (jj - 1)*width])
						nn[count] = sub2indAll[cid][ii - 1 + (jj - 1)*width], count++;
					if (validPixelsAll[cid][ii + 1 + (jj - 1)*width])
						nn[count] = sub2indAll[cid][ii + 1 + (jj - 1)*width], count++;
					if (validPixelsAll[cid][ii - 1 + (jj + 1)*width])
						nn[count] = sub2indAll[cid][ii - 1 + (jj + 1)*width], count++;
					if (validPixelsAll[cid][ii + 1 + (jj + 1)*width])
						nn[count] = sub2indAll[cid][ii + 1 + (jj + 1)*width], count++;

					NN.push_back(nn);
					nNN.push_back(count);
				}
			}
		}
		nNNAll.push_back(nNN), NNAll.push_back(NN);

	}

	for (int cid = 0; cid < (int)allImgs.size(); cid++)
	{
		for (int pyrID = 0; pyrID < nscales + 1; pyrID++)
			allImgs[cid].scaleFactor.push_back(1.0 / pow(2, pyrID));

		GaussianBlur(allImgs[cid].color, allImgs[cid].color, Size(5, 5), 0.707);
		buildPyramid(allImgs[cid].color, allImgs[cid].imgPyr, nscales);

		BuildInvDepthPyramid(allImgs[cid].InvDepth, allImgs[cid].InvDepthPyr, allImgs[cid].width, allImgs[cid].height, nscales);
	}

	/*int pyrID = 1;
	//for (int cid = 0; cid < (int)allImgs.size(); cid++)
	{
	int cid = 0;
	char Fname[512];  sprintf(Fname, "%s/D_%d_%d.dat", Path, cid, pyrID);
	ReadGridBinary(Fname, allImgs[cid].InvDepthPyr[pyrID], allImgs[cid].imgPyr[pyrID].cols, allImgs[cid].imgPyr[pyrID].rows);
	for (int ii = 0; ii < allImgs[cid].imgPyr[pyrID].cols* allImgs[cid].imgPyr[pyrID].rows; ii++)
	allImgs[cid].InvDepthPyr[pyrID][ii] *= scale;
	//UpsamleDepth(allImgs[cid].InvDepthPyr[pyrID], allImgs[cid].InvDepthPyr[pyrID - 1], allImgs[cid].imgPyr[pyrID].cols, allImgs[cid].imgPyr[pyrID].rows, allImgs[cid].imgPyr[pyrID - 1].cols, allImgs[cid].imgPyr[pyrID - 1].rows);
	}
	char Fname[512];  sprintf(Fname, "%s/p1.txt", Path);
	FILE *fp = fopen(Fname, "r");
	for (int cid = 0; cid < (int)allImgs.size(); cid++)
	{
	int dummy;fscanf(fp, "%d ", &dummy);
	if (dummy != allCalibInfo[cid].frameID)
	int a = 0;
	for (int ii = 0; ii < 6; ii++)
	fscanf(fp, "%lf ", &allCalibInfo[cid].rt[ii]);
	}
	fclose(fp);*/

	double startTime = omp_get_wtime();

	for (int pyrID = nscales; pyrID >= 1; pyrID--)
	{
		printf("\n\n@ level: %d\n", pyrID);
		for (int cid = 0; cid < (int)allImgs.size(); cid++)
		{
			char Fname[512];
			sprintf(Fname, "%s/Level_%d", Fname2, pyrID); makeDir(Fname);
			sprintf(Fname, "%s/Level_%d/%03d.png", Fname2, pyrID, cid);
			imwrite(Fname, allImgs[cid].imgPyr[pyrID]);
		}

		double startTimeI = omp_get_wtime();
		/*double percentage, innerCost1, innerCost2, pinnerCost1 = 1e16, pinnerCost2 = 1e16;
		for (int innerID = 0; innerID < innerIter; innerID++) //Pose is mostly well estimated even with nosiy depth. Depth is improved with good pose. -->Alternating optim, which is extremely helpful in practice
		{
		printf("\n@(Pose) Iter %d: ", innerID);
		innerCost2 = DirectAlignmentPyr(Path, alignmentParas, allImgs, allCalibInfo, 0, 1, pyrID, smallAngle, 0, 1);

		printf("@(Depth) Iter %d: ", innerID); //at low res, calibration error is not servere, let's optim depth first
		innerCost1 = DirectAlignmentPyr(Path, alignmentParas, allImgs, allCalibInfo, 1, 0, pyrID, smallAngle, 0, 1);

		percentage = abs(innerCost1 + innerCost2 - pinnerCost1 - pinnerCost2) / (pinnerCost1 + pinnerCost2);
		printf("Change: %.3e\n", percentage);
		if (percentage < 1.0e-3)
		break;
		pinnerCost1 = innerCost1, pinnerCost2 = innerCost2;
		}
		printf("Total time: %.2f\n", omp_get_wtime() - startTimeI);

		if (pyrID == 0)
		DirectAlignmentPyr(Path, alignmentParas, allImgs, allCalibInfo, pyrID < 5 ? 0 : 1, 0, pyrID, smallAngle, 1, 0);*/

		DirectAlignmentPyr(Fname2, alignmentParas, allImgs, allCalibInfo, 0, 0, pyrID, smallAngle, 1, 1, HuberSize);

		if (pyrID != 0)
			for (int cid = 0; cid < (int)allImgs.size(); cid++)
				UpsamleDepth(allImgs[cid].InvDepthPyr[pyrID], allImgs[cid].InvDepthPyr[pyrID - 1], allImgs[cid].imgPyr[pyrID].cols, allImgs[cid].imgPyr[pyrID].rows, allImgs[cid].imgPyr[pyrID - 1].cols, allImgs[cid].imgPyr[pyrID - 1].rows);
	}
	printf("\n\nTotal time: %.2f\n", omp_get_wtime() - startTime);

	for (int cid = 0; cid < (int)allImgs.size(); cid++)
		for (int pyrID = 0; pyrID < nscales + 1; pyrID++)
			delete[]allImgs[cid].InvDepthPyr[pyrID];

	return 1;
}


int main(int argc, char** argv)
{
	//srand(time(NULL));
	srand(1);
	char Fname[512], Path[] = "C:/Data/MSR/Micro/Stairs/0";
	int mode = 5;
	//4: init with spline from other key frames; 
	//3:good init but add noise on Sudipta's format; 
	//2: synth; 
	//1: low based; 
	//0: middlebury

	//ExtractVideoFrames("C:/Data/MSR/Drone", "DJI_0017.mov", 450, 510, 1, 0, 3, 0);
	//return 0;
	if (mode == 5)
	{
		int HuberSize = 30;
		//for (int HuberSize = 30; HuberSize < 60; HuberSize += 10)
		{
			int nchannels = 3, refF = 15, rangeF = 15, stepF = 14;
			char Path[] = "C:/Data/MSR/Micro/Stairs";
			vector<ImgData> allImgs;
			vector<CameraData> allCalibInfo;

			//double scale = 625.0 / 0.51, DepthLow = 900, DepthHigh = 1800;// MSR2
			//double scale = 150.0 / 0.0060881017756917542, DepthLow = 2000, DepthHigh = 20000; //Stairs
			//double scale = 150.0 / 0.11, DepthLow = 1000, DepthHigh = 10000; //Desk
			double scale = 1.0, DepthLow = -1, DepthHigh = -1;//Drone
			double dataWeight = 1.0, regIntraWeight = 10000000.0, regInterWeight = 1.0 - dataWeight - regIntraWeight;
			double reprojectionSigma = 0.3, colorSigma = 3.0,   //expected std of variables
				depthSigma = 1;//mm
			double ImGradientThesh2 = 100;
			DirectAlignPara alignmentParas(dataWeight, regIntraWeight, regInterWeight, colorSigma, depthSigma, ImGradientThesh2, DepthLow, DepthHigh, reprojectionSigma);

			/*sprintf(Fname, "%s/SMC/paras.txt", Path);
			FILE *fp = fopen(Fname, "r");
			if (fp == NULL)
			return 1;
			else
			{
			CameraData camI;
			fscanf(fp, "%lf %lf %lf %lf %lf ", &camI.intrinsic[0], &camI.intrinsic[3], &camI.intrinsic[4], &camI.distortion[0], &camI.distortion[1]);

			camI.ShutterModel = 0;
			camI.intrinsic[1] = camI.intrinsic[0], camI.intrinsic[2] = 0;
			GetKFromIntrinsic(camI.K, camI.intrinsic);
			GetiK(camI.invK, camI.K);

			while (fscanf(fp, "%d ", &camI.frameID) != EOF)
			{
			for (int ii = 0; ii < 6; ii++)
			fscanf(fp, "%lf ", &camI.rt[ii]);
			for (int ii = 0; ii < 3; ii++)
			camI.rt[ii + 3] *= scale;
			GetRTFromrt(camI.rt, camI.R, camI.T);
			GetCfromT(camI.R, camI.T, camI.C);
			AssembleP(camI.K, camI.R, camI.T, camI.P);
			allCalibInfo.push_back(camI);
			}
			fclose(fp);
			}*/
			/*vector<int> frames; frames.push_back(0), frames.push_back(14), frames.push_back(15), frames.push_back(27), frames.push_back(28);
			double SfMdistance = TriangulatePointsFromNonCorpusCameras(Path, allCalibInfo, frames, 1, 2, 2.0);
			printf("SfM measured distance: %.6f\n ", SfMdistance);
			return 0;*/

			vector<int> allframes;
			allframes.push_back(refF);
			for (int ii = stepF; ii < rangeF; ii += stepF)
				//allframes.push_back(refF + ii), 
				allframes.push_back(refF - ii);
			for (int fid : allframes)
			{
				CameraData camI;
				camI.frameID = fid;
				//camI.intrinsic[0] = 0.7 * 3840, camI.intrinsic[3] = 3840 / 2, camI.intrinsic[4] = 2160 / 2, camI.distortion[0] = 0, camI.distortion[1] = 0;
				camI.intrinsic[0] = 1233.617524, camI.intrinsic[3] = 640, camI.intrinsic[4] = 360, camI.distortion[0] = 0, camI.distortion[1] = 0;

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
			for (int ii = 0; ii < (int)allCalibInfo.size(); ii++)
			{
				ImgData imgI;
				sprintf(Fname, "%s/0/%d.png", Path, allCalibInfo[ii].frameID);
				imgI.color = imread(Fname, nchannels == 1 ? 0 : 1);
				if (imgI.color.data == NULL)
				{
					printf("Cannot load %s\n", Fname);
					continue;
				}
				imgI.width = imgI.color.cols, imgI.height = imgI.color.rows, imgI.frameID = allCalibInfo[ii].frameID, imgI.nchannels = nchannels;
				allImgs.push_back(imgI);
			}

			//InitializeRT(allImgs, allCalibInfo);
			/*sprintf(Fname, "%s/Corpus/n3dGL.xyz", Path);
			Corpus CorpusData; ReadCorpusInfo(Fname, CorpusData, scale);
			float depth[2];
			ExtractDepthFromCorpus(CorpusData, allCalibInfo[0], 0, depth);*/

			for (int ii = 0; ii < allCalibInfo.size(); ii++)
				for (int jj = 0; jj < 3; jj++)
					;// allCalibInfo[ii].rt[jj] = 0; //rotation is not precise, translation is pretty good via bspline of keyframes
			for (int ii = 1; ii < allCalibInfo.size(); ii++)
				for (int jj = 0; jj < 3; jj++)
					;// allCalibInfo[ii].rt[jj + 3] += 0.0025*rand() / RAND_MAX; //approximately 5mm for Micro4 

			for (int cid = 0; cid < (int)allImgs.size(); cid++)
				allImgs[cid].InvDepth = new float[allImgs[0].width*allImgs[0].height];

			for (int cid = 0; cid < (int)allImgs.size(); cid++)
				for (int jj = 0; jj < allImgs[cid].height; jj++)
					for (int ii = 0; ii < allImgs[cid].width; ii++)
						allImgs[cid].InvDepth[ii + jj*allImgs[cid].width] = min(2.5, max(0.001, gaussian_noise(1.0, 0.5)));

			DirectAlignment(Path, alignmentParas, allImgs, allCalibInfo, 0, scale, HuberSize);
		}
		return 0;
	}
	if (mode == 4) //no init
	{
		int nchannels = 3;
		char Path[] = "D:/Data/Micro/MSR2";
		vector<ImgData> allImgs;
		vector<CameraData> allCalibInfo;

		//double scale = 2630;//for micro4
		//double scale = 246;//for micro3
		double scale = 1.0;
		//double scale = 1.0; 
		double dataWeight = 0.01, regIntraWeight = 10.0, regInterWeight = 1.0 - dataWeight - regIntraWeight;
		double DepthLow = 1000, DepthHigh = 1800;
		double reprojectionSigma = 0.3, colorSigma = 3.0,   //expected std of variables
			depthSigma = 10;//mm
		double ImGradientThesh2 = 1;
		DirectAlignPara alignmentParas(dataWeight, regIntraWeight, regInterWeight, colorSigma, depthSigma, ImGradientThesh2, DepthLow, DepthHigh, reprojectionSigma);

		int refF = 130;
		vector<int> camIDs; camIDs.push_back(refF);
		for (int ii = 1; ii < 20; ii++)
			camIDs.push_back(refF + ii), camIDs.push_back(refF - ii);

		VideoData vInfo;
		ReadVideoDataI(Path, vInfo, 0, 0, 1800, scale);
		for (int ii = 0; ii < (int)camIDs.size(); ii++)
			allCalibInfo.push_back(vInfo.VideoInfo[camIDs[ii]]);

		vector<int> frames; frames.push_back(refF - 19), frames.push_back(refF - 10), frames.push_back(refF), frames.push_back(refF + 10), frames.push_back(refF + 19);
		double SfMdistance = TriangulatePointsFromNonCorpusCameras(Path, vInfo, frames, 1, 10, 2.0);
		printf("SfM measured distance: %.3f\nPlease input the physcial distance (mm): ", SfMdistance);
		return 0;

		for (int ii = 0; ii < camIDs.size(); ii++)
		{
			ImgData imgI;
			sprintf(Fname, "%s/0/%d.png", Path, camIDs[ii]);
			imgI.color = imread(Fname, nchannels == 1 ? 0 : 1);
			if (imgI.color.data == NULL)
			{
				printf("Cannot load %s\n", Fname);
				return 0;
			}
			imgI.width = imgI.color.cols, imgI.height = imgI.color.rows, imgI.frameID = camIDs[ii], imgI.nchannels = nchannels;
			allImgs.push_back(imgI);
		}

		/*sprintf(Fname, "%s/Corpus/n3dGL.xyz", Path);
		Corpus CorpusData; ReadCorpusInfo(Fname, CorpusData, scale);
		float depth[2];
		ExtractDepthFromCorpus(CorpusData, allCalibInfo[0], 0, depth);*/

		for (int ii = 0; ii < camIDs.size(); ii++)
			for (int jj = 0; jj < 3; jj++)
				;// allCalibInfo[ii].rt[jj] = 0; //rotation is not precise, translation is pretty good via bspline of keyframes
		for (int ii = 1; ii < camIDs.size(); ii++)
			for (int jj = 0; jj < 3; jj++)
				;// allCalibInfo[ii].rt[jj + 3] += 0.0025*rand() / RAND_MAX; //approximately 5mm for Micro4 

		for (int cid = 0; cid < (int)allImgs.size(); cid++)
			allImgs[cid].InvDepth = new float[allImgs[0].width*allImgs[0].height];

		for (int cid = 0; cid < (int)allImgs.size(); cid++)
			for (int jj = 0; jj < allImgs[cid].height; jj++)
				for (int ii = 0; ii < allImgs[cid].width; ii++)
					allImgs[cid].InvDepth[ii + jj*allImgs[cid].width] = 2000;// 9e9;

		DirectAlignment(Path, alignmentParas, allImgs, allCalibInfo, 1, scale);

		visualizationDriver(Path, refF - 19, refF + 19, 1, refF);
		return 0;
	}
	if (mode == 3) //good init but add noise on Sudipta's format
	{
		int nchannels = 3;
		char Path[] = "D:\\Source\\Repos\\Users\\sudipsin\\recon3D\\data\\Micro";
		vector<ImgData> allImgs;
		vector<CameraData> allCalibInfo;

		int SynthID = 0;
		Corpus CorpusData;
		ReadSynthFile(Path, CorpusData, SynthID);

		vector<int> camIDs; camIDs.push_back(45);
		//for (int ii = 1; ii <= 15; ii++)
		//	camIDs.push_back(45 + ii), camIDs.push_back(45 - ii);
		camIDs.push_back(23), camIDs.push_back(27);
		camIDs.push_back(18), camIDs.push_back(25);

		for (int ii = 0; ii < camIDs.size(); ii++)
		{
			ImgData imgI;
			sprintf(Fname, "%s/Recon/urd-%03d.png", Path, camIDs[ii]);
			imgI.color = imread(Fname, nchannels == 1 ? 0 : 1);
			imgI.width = imgI.color.cols, imgI.height = imgI.color.rows, imgI.nchannels = nchannels;
			allImgs.push_back(imgI);
		}

		for (int ii = 0; ii < camIDs.size(); ii++)
			allCalibInfo.push_back(CorpusData.camera[camIDs[ii]]);

		for (int cid = 0; cid < (int)allImgs.size(); cid++)
			allImgs[cid].InvDepth = new float[allImgs[0].width*allImgs[0].height];

		ReadSudiptaDepth(Path, allImgs[0], camIDs[0]);
		ConvertFrontoDepth2LineOfSightDepth(allImgs[0], allCalibInfo[0]);

		/*for (int cid = 0; cid < (int)allImgs.size(); cid++)
		{
		int cid = 0;
		for (int jj = 0; jj < allImgs[cid].height; jj++)
		for (int ii = 0; ii < allImgs[cid].width; ii++)
		if (abs(allImgs[cid].InvDepth[ii + jj*allImgs[cid].width]) >0.001)
		allImgs[cid].InvDepth[ii + jj*allImgs[cid].width] = 9.0*rand() / RAND_MAX - 4.5;
		}*/
		//DirectAlignment(Path, allImgs, allCalibInfo);
	}
	if (mode == 2) //synth
	{
		char Path[] = "D:/Data/DirectBA";

		int nchannels = 3;
		vector<int> selectedCamID; selectedCamID.push_back(8), selectedCamID.push_back(4), selectedCamID.push_back(12), selectedCamID.push_back(0), selectedCamID.push_back(16);

		Corpus CorpusData;
		sprintf(Fname, "%s/Corpus/calibInfo.txt", Path);
		ReadCalibInfo(Fname, CorpusData);

		vector<CameraData> allCalibInfo;
		for (int ii = 0; ii < (int)selectedCamID.size(); ii++)
			allCalibInfo.push_back(CorpusData.camera[selectedCamID[ii]]);

		ImgData imgI;
		vector<ImgData> allImgs;
		for (int ii = 0; ii < (int)selectedCamID.size(); ii++)
		{
			sprintf(Fname, "%s/Image/C_%d_1.png", Path, selectedCamID[ii]);
			imgI.color = imread(Fname, nchannels == 1 ? 0 : 1);
			imgI.width = imgI.color.cols, imgI.height = imgI.color.rows, imgI.nchannels = nchannels;

			if (ii == 0)
			{
				sprintf(Fname, "%s/GT/CD_%d_1.ijz", Path, selectedCamID[ii]);
				imgI.InvDepth = new float[imgI.width * imgI.height];
				ReadGridBinary(Fname, imgI.InvDepth, imgI.width, imgI.height);
			}

			allImgs.push_back(imgI);
		}

		/*for (int cid = 1; cid < (int)allImgs.size(); cid++)
		{
		for (int ii = 0; ii < 3; ii++)
		allCalibInfo[cid].rt[ii] += min(max(gaussian_noise(0, 0.05), -0.15), 0.15);
		for (int ii = 0; ii < 3; ii++)
		allCalibInfo[cid].rt[ii + 3] += min(max(gaussian_noise(0, 1.), -3.), 3.);
		}
		for (int cid = 0; cid < (int)allImgs.size(); cid++)
		{
		for (int jj = 0; jj < allImgs[cid].height; jj++)
		for (int ii = 0; ii < allImgs[cid].width; ii++)
		{
		if (allImgs[cid].InvDepth[ii + jj*allImgs[cid].width] < -0.001)
		;// allImgs[cid].InvDepth[ii + jj*allImgs[cid].width] += min(max(gaussian_noise(0, 0.5), -1.5), 1.5);
		}
		}*/

		//DirectAlignment(Path, allImgs, allCalibInfo);
	}
	if (mode == 1) //flow based
	{
		int nchannels = 3;
		char Path[] = "D:/Data/Micro3";
		vector<ImgData> allImgs;
		vector<CameraData> allCalibInfo;

		int refF = 470;
		vector<int> camIDs; camIDs.push_back(refF);
		for (int ii = 1; ii < 20; ii++)
			camIDs.push_back(refF + ii), camIDs.push_back(refF - ii);

		for (int ii = 0; ii < camIDs.size(); ii++)
		{
			ImgData imgI;
			sprintf(Fname, "%s/0/%d.png", Path, camIDs[ii]);
			imgI.color = imread(Fname, nchannels == 1 ? 0 : 1);
			if (imgI.color.data == NULL)
				continue;
			imgI.frameID = camIDs[ii], imgI.width = imgI.color.cols, imgI.height = imgI.color.rows, imgI.nchannels = nchannels;
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

		VideoData vInfo;
		ReadVideoDataI(Path, vInfo, 0, 0, 1800);
		for (int ii = 0; ii < camIDs.size(); ii++)
			allCalibInfo.push_back(vInfo.VideoInfo[camIDs[ii]]);

		for (int ii = 0; ii < camIDs.size(); ii++)
			for (int jj = 0; jj < 3; jj++)
				allCalibInfo[ii].rt[jj] = 0; //rotation is not precise, translation is pretty good via bspline of keyframes

		double reProjectionSigma = 1.0;
		FlowBasedBundleAdjustment(Path, allImgs, sfd, allCalibInfo, reProjectionSigma, 0, 0, 1);
		waitKey(0);

		return 0;
	}
	if (mode == 0) //middlebury
	{
		double focal = 3968.297, baseline = 236.922, doffs = 77.215;

		CameraData Cam1, Cam2;
		Cam1.width = 2960, Cam1.height = 1924;
		Cam2.width = 2960, Cam2.height = 1924;
		Cam1.intrinsic[0] = focal, Cam1.intrinsic[1] = focal, Cam1.intrinsic[2] = 0.0, Cam1.intrinsic[3] = 1188.925, Cam1.intrinsic[4] = 979.657;
		Cam2.intrinsic[0] = focal, Cam2.intrinsic[1] = focal, Cam2.intrinsic[2] = 0.0, Cam2.intrinsic[3] = 1266.14, Cam2.intrinsic[4] = 979.657;
		Cam1.rt[0] = 0, Cam1.rt[1] = 0, Cam1.rt[2] = 0, Cam1.rt[3] = 0, Cam1.rt[4] = 0, Cam1.rt[5] = 0;
		Cam2.rt[0] = 0.01, Cam2.rt[1] = 0.0, Cam2.rt[2] = 0.0, Cam2.rt[3] = -baseline, Cam2.rt[4] = 0, Cam2.rt[5] = 0;

		GetKFromIntrinsic(Cam1.K, Cam1.intrinsic), GetiK(Cam1.invK, Cam1.K);
		GetKFromIntrinsic(Cam2.K, Cam2.intrinsic), GetiK(Cam2.invK, Cam2.K);

		GetRTFromrt(Cam1.rt, Cam1.R, Cam1.T), GetCfromT(Cam1.R, Cam1.T, Cam1.C), AssembleRT(Cam1.R, Cam1.T, Cam1.RT);
		GetRTFromrt(Cam2.rt, Cam2.R, Cam2.T), GetCfromT(Cam2.R, Cam2.T, Cam2.C), AssembleRT(Cam2.R, Cam2.T, Cam2.RT);

		AssembleP(Cam1.K, Cam1.RT, Cam1.P);
		AssembleP(Cam2.K, Cam2.RT, Cam2.P);

		vector<CameraData> allCalibInfo;
		allCalibInfo.push_back(Cam1), allCalibInfo.push_back(Cam2);

		ImgData imgI;
		vector<ImgData> allImgs;

		int nchannels = 1;
		imgI.color = imread("C:/temp/Pipes-perfect/im0.png", nchannels == 1 ? 0 : 1);
		imgI.width = imgI.color.cols, imgI.height = imgI.color.rows, imgI.nchannels = nchannels;
		allImgs.push_back(imgI);

		imgI.color = imread("C:/temp/Pipes-perfect/im1.png", nchannels == 1 ? 0 : 1);
		imgI.width = imgI.color.cols, imgI.height = imgI.color.rows, imgI.nchannels = nchannels;
		allImgs.push_back(imgI);

		read_pfm_file("C:/temp/Pipes-perfect/disp0.pfm", allImgs[0]);
		ConvertDisparirty2DepthMap(allImgs[0], focal, baseline, doffs);

		read_pfm_file("C:/temp/Pipes-perfect/disp1.pfm", allImgs[1]);
		ConvertDisparirty2DepthMap(allImgs[1], focal, baseline, doffs);

		//DirectAlignment("C:/temp/Pipes-perfect", allImgs, allCalibInfo);

		return 0;
	}


	return 0;
}

/*ImgData img;
img.color = imread("C:/Data/Micro4/0/130.png");
img.width = img.color.cols, img.nchannels = 3;

cvNamedWindow("Image", CV_WINDOW_NORMAL); setMouseCallback("Image", onMouse);
Mat weightImg, gray; cvtColor(img.color, gray, CV_BGR2GRAY);
cvtColor(gray, weightImg, CV_GRAY2BGR);

int hb = 29;
vector<float> localWeights; localWeights.resize((2 * hb + 1)* (2 * hb + 1));
bool *mask = new bool[1920 * 1080];
for (int ii = 0; ii < 1920 * 1080; ii++)
mask[ii] = true;

int i, j;
vector<Point2i> ij; ij.reserve(13000);
FILE *fp = fopen("C:/temp/validIJ.txt", "r");
while (fscanf(fp, "%d %d", &i, &j) != EOF)
ij.push_back(Point2i(i, j));
fclose(fp);


for (int ll = 0; ll < (int)ij.size(); ll++)
{
ComputeGeodesicWeight(img.color.data, mask, 1920, 3, &localWeights[0], ij[ll].x, ij[ll].y, hb, 10);

cvtColor(gray, weightImg, CV_GRAY2BGR);
for (int jj = -hb; jj <= hb; jj++)
for (int ii = -hb; ii <= hb; ii++)
for (int kk = 0; kk < 3; kk++)
weightImg.data[(ij[ll].x + ii + (ij[ll].y + jj) * 1920) * 3 + kk] = (uchar)(255.0*localWeights[ii + hb + (jj + hb) * (2 * hb + 1)]);

imshow("Image", weightImg);
if (waitKey(-1) == 27)
break;
}

/*while (true)
{
if (waitKey(16) == 27)
break;
if (clicked == 1)
{
printf("%d %d ", MousePosX, MousePosY);
ComputeGeodesicWeight(img.color.data, mask, 1920, 3, &localWeights[0], MousePosX, MousePosY, hb);

cvtColor(gray, weightImg, CV_GRAY2BGR);
for (int jj = -hb; jj <= hb; jj++)
for (int ii = -hb; ii <= hb; ii++)
for (int kk = 0; kk < 3; kk++)
weightImg.data[(MousePosX + ii + (MousePosY + jj) * 1920) * 3 + kk] = (uchar)(255.0*localWeights[ii + hb + (jj + hb) * (2 * hb + 1)]);

imshow("Image", weightImg); waitKey(1);
clicked = 0;
}
}
return 0;*/





