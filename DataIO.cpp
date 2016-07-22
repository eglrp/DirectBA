#include "DataIO.h"

using namespace std;
using namespace cv;
using namespace Eigen;

void makeDir(char *Fname)
{
#ifdef _WINDOWS
	_mkdir(Fname);
#else
	mkdir(Fname, 0755);
#endif
	return;
}
int IsFileExist(char *Fname, bool silient)
{
	std::ifstream test(Fname);
	if (test.is_open())
	{
		test.close();
		return 1;
	}
	else
	{
		if (!silient)
			printf("Cannot load %s\n", Fname);
		return 0;
	}
}
int freadi(FILE * fIn)
{
	int iTemp;
	fread(&iTemp, sizeof(int), 1, fIn);
	return iTemp;
}
float freadf(FILE * fIn)
{
	float fTemp;
	fread(&fTemp, sizeof(float), 1, fIn);
	return fTemp;
}
double freadd(FILE * fIn)
{
	double dTemp;
	fread(&dTemp, sizeof(double), 1, fIn);
	return dTemp;
}
int is_little_endian()
{
	if (sizeof(float) != 4)
	{
		printf("Bad float size.\n"); exit(1);
	}
	uchar b[4] = { 255, 0, 0, 0 };
	return *((float *)b) < 1.0;
}


int readCalibInfo(char *BAfileName, Corpus &CorpusData)
{
	FILE *fp = fopen(BAfileName, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", BAfileName);
		return 1;
	}

	char Fname[200];
	int lensType, shutterModel, width, height;
	double fx, fy, skew, u0, v0, r1, r2, r3, t1, t2, p1, p2, omega, DistCtrX, DistCtrY, rt[6];

	fscanf(fp, "%d ", &CorpusData.nCameras);
	CorpusData.camera = new CameraData[CorpusData.nCameras];
	for (int ii = 0; ii < CorpusData.nCameras; ii++)
		CorpusData.camera[ii].valid = false;

	for (int ii = 0; ii < CorpusData.nCameras; ii++)
	{
		if (fscanf(fp, "%s %d %d %d %d", &Fname, &lensType, &shutterModel, &width, &height) == EOF)
			break;
		string filename = Fname;
		size_t dotpos = filename.find("."), slength = filename.length();
		int viewID;
		if (slength - dotpos == 4)
		{
			std::size_t pos = filename.find(".ppm");
			if (pos > 1000)
			{
				pos = filename.find(".png");
				if (pos > 1000)
				{
					pos = filename.find(".jpg");
					if (pos > 1000)
					{
						printf("Something wrong with the image name in the BA file!\n");
						abort();
					}
				}
			}
			filename.erase(pos, 4);
			const char * str = filename.c_str();
			viewID = atoi(str);
		}
		else
		{
			printf("Problem with the calibration file!");
			exit(1);
		}

		CorpusData.camera[viewID].LensModel = lensType, CorpusData.camera[viewID].ShutterModel = shutterModel;
		CorpusData.camera[viewID].width = width, CorpusData.camera[viewID].height = height;
		if (lensType == 0)
		{
			fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf ", &fx, &fy, &skew, &u0, &v0,
				&r1, &r2, &r3, &t1, &t2, &p1, &p2,
				&rt[0], &rt[1], &rt[2], &rt[3], &rt[4], &rt[5]);

			CorpusData.camera[viewID].distortion[0] = r1, CorpusData.camera[viewID].distortion[1] = r2, CorpusData.camera[viewID].distortion[2] = r3,
				CorpusData.camera[viewID].distortion[3] = t1, CorpusData.camera[viewID].distortion[4] = t2,
				CorpusData.camera[viewID].distortion[5] = p1, CorpusData.camera[viewID].distortion[6] = p2;
		}
		else
		{
			fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf ", &fx, &fy, &skew, &u0, &v0,
				&omega, &DistCtrX, &DistCtrY,
				&rt[0], &rt[1], &rt[2], &rt[3], &rt[4], &rt[5]);

			CorpusData.camera[viewID].distortion[0] = omega, CorpusData.camera[viewID].distortion[1] = DistCtrX, CorpusData.camera[viewID].distortion[2] = DistCtrY;
			for (int jj = 3; jj < 7; jj++)
				CorpusData.camera[viewID].distortion[jj] = 0;
		}
		if (CorpusData.camera[viewID].ShutterModel == 1)
			fscanf(fp, "%lf %lf %lf %lf %lf %lf ", &CorpusData.camera[viewID].wt[0], &CorpusData.camera[viewID].wt[1], &CorpusData.camera[viewID].wt[2], &CorpusData.camera[viewID].wt[3], &CorpusData.camera[viewID].wt[4], &CorpusData.camera[viewID].wt[5]);
		else
			for (int jj = 0; jj < 6; jj++)
				CorpusData.camera[viewID].wt[jj] = 0.0;

		CorpusData.camera[viewID].intrinsic[0] = fx, CorpusData.camera[viewID].intrinsic[1] = fy, CorpusData.camera[viewID].intrinsic[2] = skew, CorpusData.camera[viewID].intrinsic[3] = u0, CorpusData.camera[viewID].intrinsic[4] = v0;
		GetKFromIntrinsic(CorpusData.camera[viewID].K, CorpusData.camera[viewID].intrinsic);
		GetiK(CorpusData.camera[viewID].invK, CorpusData.camera[viewID].K);

		for (int jj = 0; jj < 6; jj++)
			CorpusData.camera[viewID].rt[jj] = rt[jj];

		GetRTFromrt(CorpusData.camera[viewID].rt, CorpusData.camera[viewID].R, CorpusData.camera[viewID].T);
		GetCfromT(CorpusData.camera[viewID].R, CorpusData.camera[viewID].rt + 3, CorpusData.camera[viewID].C);

		if (CorpusData.camera[viewID].ShutterModel == 0)
			AssembleP(CorpusData.camera[viewID].K, CorpusData.camera[viewID].RT, CorpusData.camera[viewID].P);
	}
	fclose(fp);

	return 0;
}
int ReadVideoDataI(char *Path, VideoData &vInfo, int viewID, int startTime, int stopTime, double threshold, int ninliersThresh, int silent)
{
	char Fname[200];
	int frameID, LensType, ShutterModel, width, height;
	if (startTime == -1 && stopTime == -1)
		startTime = 0, stopTime = 10000;
	int maxFrameOffset = vInfo.maxFrameOffset, nframes = stopTime + maxFrameOffset + 1;

	vInfo.nframesI = nframes;
	vInfo.startTime = startTime;
	vInfo.stopTime = stopTime;
	vInfo.VideoInfo = new CameraData[nframes];
	for (int ii = 0; ii < nframes; ii++)
	{
		vInfo.VideoInfo[ii].valid = false;
		vInfo.VideoInfo[ii].threshold = threshold;
	}

	//READ INTRINSIC: START
	int validFrame = 0;
	sprintf(Fname, "%s/vHIntrinsic_%d.txt", Path, viewID);
	if (IsFileExist(Fname) == 0)
	{
		sprintf(Fname, "%s/avIntrinsic_%d.txt", Path, viewID);
		if (IsFileExist(Fname) == 0)
		{
			//printf("Cannot find %s...", Fname);
			sprintf(Fname, "%s/vIntrinsic_%d.txt", Path, viewID);
			if (IsFileExist(Fname) == 0)
			{
				//printf("Cannot find %s...", Fname);
				sprintf(Fname, "%s/Intrinsic_%d.txt", Path, viewID);
				if (IsFileExist(Fname) == 0)
				{
					printf("Cannot find %s...\n", Fname);
					return 1;
				}
			}
		}
	}
	FILE *fp = fopen(Fname, "r");
	if (silent == 0)
		printf("Loaded %s\n", Fname);
	double fx, fy, skew, u0, v0, r0, r1, r2, t0, t1, p0, p1, omega, DistCtrX, DistCtrY;
	while (fscanf(fp, "%d %d %d %d %d %lf %lf %lf %lf %lf ", &frameID, &LensType, &ShutterModel, &width, &height, &fx, &fy, &skew, &u0, &v0) != EOF)
	{
		if (frameID >= startTime - maxFrameOffset && frameID <= stopTime + maxFrameOffset)
		{
			vInfo.VideoInfo[frameID].intrinsic[0] = fx, vInfo.VideoInfo[frameID].intrinsic[1] = fy, vInfo.VideoInfo[frameID].intrinsic[2] = skew,
				vInfo.VideoInfo[frameID].intrinsic[3] = u0, vInfo.VideoInfo[frameID].intrinsic[4] = v0;

			GetKFromIntrinsic(vInfo.VideoInfo[frameID].K, vInfo.VideoInfo[frameID].intrinsic);
			mat_invert(vInfo.VideoInfo[frameID].K, vInfo.VideoInfo[frameID].invK);

			vInfo.VideoInfo[frameID].LensModel = LensType, vInfo.VideoInfo[frameID].ShutterModel = ShutterModel, vInfo.VideoInfo[frameID].threshold = threshold, vInfo.VideoInfo[frameID].ninlierThresh = ninliersThresh;
			vInfo.VideoInfo[frameID].hasIntrinsicExtrinisc++;
			validFrame = frameID;
		}

		if (LensType == 0)
		{
			fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf ", &r0, &r1, &r2, &t0, &t1, &p0, &p1);
			if (frameID >= startTime - maxFrameOffset && frameID <= stopTime + maxFrameOffset)
			{
				vInfo.VideoInfo[frameID].distortion[0] = r0, vInfo.VideoInfo[frameID].distortion[1] = r1, vInfo.VideoInfo[frameID].distortion[2] = r2;
				vInfo.VideoInfo[frameID].distortion[3] = t0, vInfo.VideoInfo[frameID].distortion[4] = t1;
				vInfo.VideoInfo[frameID].distortion[5] = p0, vInfo.VideoInfo[frameID].distortion[6] = p1;
			}
		}
		else
		{
			fscanf(fp, "%lf %lf %lf ", &omega, &DistCtrX, &DistCtrY);
			if (frameID >= startTime - maxFrameOffset && frameID <= stopTime + maxFrameOffset)
				vInfo.VideoInfo[frameID].distortion[0] = omega, vInfo.VideoInfo[frameID].distortion[1] = DistCtrX, vInfo.VideoInfo[frameID].distortion[2] = DistCtrY;
		}
		if (frameID >= startTime - maxFrameOffset && frameID <= stopTime + maxFrameOffset)
			vInfo.VideoInfo[frameID].width = width, vInfo.VideoInfo[frameID].height = height;
	}
	fclose(fp);
	//END

	//READ POSE FROM VIDEO POSE: START
	if (vInfo.VideoInfo[validFrame].ShutterModel == 0)
	{
		sprintf(Fname, "%s/vHCamPose_%d.txt", Path, viewID);
		if (IsFileExist(Fname) == 0)
		{
			sprintf(Fname, "%s/avCamPose_%d.txt", Path, viewID);
			if (IsFileExist(Fname) == 0)
			{
				//printf("Cannot find %s...", Fname);
				sprintf(Fname, "%s/vCamPose_%d.txt", Path, viewID);
				if (IsFileExist(Fname) == 0)
				{
					//printf("Cannot find %s...", Fname);
					sprintf(Fname, "%s/CamPose_%d.txt", Path, viewID);
					if (IsFileExist(Fname) == 0)
					{
						printf("Cannot find %s...\n", Fname);
						return 1;
					}
				}
			}
		}
	}
	else if (vInfo.VideoInfo[validFrame].ShutterModel == 1)
	{
		sprintf(Fname, "%s/vHCamPose_RSCayley_%d.txt", Path, viewID);
		if (IsFileExist(Fname) == 0)
		{
			sprintf(Fname, "%s/avCamPose_RSCayley_%d.txt", Path, viewID);
			if (IsFileExist(Fname) == 0)
			{
				//printf("Cannot find %s...", Fname);
				sprintf(Fname, "%s/vCamPose_RSCayley_%d.txt", Path, viewID);
				if (IsFileExist(Fname) == 0)
				{
					//printf("Cannot find %s...", Fname);
					sprintf(Fname, "%s/CamPose_%d.txt", Path, viewID);
					if (IsFileExist(Fname) == 0)
					{
						printf("Cannot find %s...\n", Fname);
						return 1;
					}
				}
			}
		}
	}
	else
	{
		sprintf(Fname, "%s/CamPose_Spline_%d.txt", Path, viewID);
		if (IsFileExist(Fname) == 0)
		{
			printf("Cannot find %s...", Fname);
			return 1;
		}
	}
	fp = fopen(Fname, "r");
	if (silent == 0)
		printf("Loaded %s\n", Fname);
	double rt[6], wt[6];
	while (fscanf(fp, "%d %lf %lf %lf %lf %lf %lf ", &frameID, &rt[0], &rt[1], &rt[2], &rt[3], &rt[4], &rt[5]) != EOF)
	{
		if (vInfo.VideoInfo[validFrame].ShutterModel == 1)
			for (int jj = 0; jj < 6; jj++)
				fscanf(fp, "%lf ", &wt[jj]);

		if (frameID >= startTime - maxFrameOffset && frameID <= stopTime + maxFrameOffset)
		{
			if (vInfo.VideoInfo[frameID].hasIntrinsicExtrinisc < 1 || abs(rt[3]) + abs(rt[4]) + abs(rt[5]) < 0.001)
			{
				vInfo.VideoInfo[frameID].valid = false;
				continue;
			}

			if (vInfo.VideoInfo[frameID].hasIntrinsicExtrinisc > 0)
				vInfo.VideoInfo[frameID].valid = true;

			for (int jj = 0; jj < 6; jj++)
				vInfo.VideoInfo[frameID].rt[jj] = rt[jj];
			GetRTFromrt(vInfo.VideoInfo[frameID].rt, vInfo.VideoInfo[frameID].R, vInfo.VideoInfo[frameID].T);
			GetCfromT(vInfo.VideoInfo[frameID].R, vInfo.VideoInfo[frameID].T, vInfo.VideoInfo[frameID].C);

			if (vInfo.VideoInfo[frameID].ShutterModel == 1)
				for (int jj = 0; jj < 6; jj++)
					vInfo.VideoInfo[frameID].wt[jj] = wt[jj];

			AssembleP(vInfo.VideoInfo[frameID].K, vInfo.VideoInfo[frameID].R, vInfo.VideoInfo[frameID].T, vInfo.VideoInfo[frameID].P);
		}
	}
	fclose(fp);
	//READ FROM VIDEO POSE: END

	return 0;
}
void ReadDepthFile(char *Path, ImgData &imdat, int i)
{
	char Fname[MAX_STRING_SIZE];

	sprintf(Fname, "%s/Recon/pts3d.%04d.dat", Path, i);
	FILE *fd = fopen(Fname, "rb");
	if (fd == NULL)
	{
		printf("Cannot open %s\n", Fname);
		return;
	}
	int numPoints = freadi(fd);
	double scale = freadd(fd);
	int W = freadi(fd), H = freadi(fd);
	int n0 = freadi(fd), n1 = freadi(fd);
	double mindepth = freadd(fd), maxdepth = freadd(fd);
	if (W == 0 || H == 0)
	{
		fclose(fd);
		return;
	}

	imdat.depth = new float[W*H];
	imdat.depthConf = new int[W*H];

	for (int p = 0; p < numPoints; p++)
	{
		float x = freadf(fd), y = freadf(fd), z = freadf(fd);
		int xi = freadi(fd), yi = freadi(fd);
		float d = freadf(fd), c = freadf(fd);
		int l = freadi(fd);

		imdat.depth[xi + yi*W] = d;
		imdat.depthConf[xi + yi*W] = (int)(c + 0.5);
	}
	printf("done.\n");

	return;
}
int ReadSynthFile(char *Path, Corpus &CorpusData, int SynthID)
{
	char Fname[512], dummyT[512];
	sprintf(Fname, "%s/Recon/synth_%d.out", Path, SynthID);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}

	fscanf(fp, "%s %s", dummyT, dummyT);
	fscanf(fp, "%d %d", &CorpusData.nCameras, &CorpusData.n3dPoints);

	vector<int> validCameraIndices(CorpusData.nCameras);
	printf("numCameras %d numPoints %d\n", CorpusData.nCameras, CorpusData.n3dPoints);

	int validCamId = 0;
	CorpusData.camera = new CameraData[CorpusData.nCameras];
	for (int i = 0; i < CorpusData.nCameras; i++)
	{
		validCameraIndices[i] = -1;
		if (fscanf(fp, "%lf %lf %lf ", &CorpusData.camera[i].intrinsic[0], &CorpusData.camera[i].distortion[0], &CorpusData.camera[i].distortion[1]) != 3)
		{
			printf("could not read camera parameters from synth_0.out\n");
			return 1;
		}
		for (int ii = 0; ii < 9; ii++)
			fscanf(fp, "%lf ", &CorpusData.camera[i].R[ii]);

		for (int ii = 0; ii < 3; ii++)
			fscanf(fp, "%lf ", &CorpusData.camera[i].T[ii]);
	}

	/*CorpusData.xyz.reserve(CorpusData.n3dPoints);
	CorpusData.rgb.reserve(CorpusData.n3dPoints);

	for (int i = 0; i < CorpusData.n3dPoints; i++)
	{
	Point3d xyz;
	if (fscanf(fp, "%lf %lf %lf", &xyz.x, &xyz.y, &xyz.z) != 3)
	return 1;
	CorpusData.xyz.push_back(xyz);

	Point3i rgb;
	if (fscanf(fp, "%d %d %d", &rgb.x, &rgb.y, &rgb.z) != 3)
	return 1;
	CorpusData.rgb.push_back(rgb);

	int numMatches = 0;
	if (fscanf(fp, "%d", &numMatches) != 1)
	return 1;

	CorpusData.nEleAll3D.push_back(numMatches);
	int* viewIdPer3D = new int[numMatches]; //3D -> visiable views index
	int* pointIdPer3D = new int[numMatches]; //3D -> 2D index in those visible views
	Point2f*  uvPer3D = new Point2f[numMatches]; //3D -> uv of that point in those visible views
	float *scalePer3D = new float[numMatches]; //3D -> scale of that point in those visible views

	for (int j = 0; j < numMatches; j++)
	{
	int imgId, featureId;
	float scl;
	if (fscanf(fp, "%d %d %f", &imgId, &featureId, &scl) != 3)
	return 1;

	viewIdPer3D[j] = imgId;
	pointIdPer3D[j] = featureId;
	scalePer3D[j] = scl;
	}
	CorpusData.viewIdAll3D.push_back(viewIdPer3D);
	CorpusData.pointIdAll3D.push_back(pointIdPer3D);
	CorpusData.scaleAll3D.push_back(scalePer3D);
	CorpusData.uvAll3D.push_back(uvPer3D);
	} // for i
	fclose(fp);*/

	int width = 1920, height = 1080;
	for (int i = 0; i < CorpusData.nCameras; i++)
	{
		CorpusData.camera[i].width = width, CorpusData.camera[i].height = height;
		CorpusData.camera[i].intrinsic[1] = CorpusData.camera[i].intrinsic[0] = CorpusData.camera[i].intrinsic[0] * max(width, height);
		CorpusData.camera[i].intrinsic[2] = 0.0;
		CorpusData.camera[i].intrinsic[3] = width / 2, CorpusData.camera[i].intrinsic[4] = height / 2;

		GetKFromIntrinsic(CorpusData.camera[i].K, CorpusData.camera[i].intrinsic);
		GetiK(CorpusData.camera[i].invK, CorpusData.camera[i].K);
		mat_invert(CorpusData.camera[i].K, CorpusData.camera[i].invK, 3);
		GetrtFromRT(CorpusData.camera[i].rt, CorpusData.camera[i].R, CorpusData.camera[i].T);
		GetCfromT(CorpusData.camera[i].R, CorpusData.camera[i].T, CorpusData.camera[i].C);
	}

	printf("done.\n");

	return 0;
}
int WriteSynthFile(char *Path, Corpus &CorpusData, int SynthID)
{
	int width = 1920, height = 1080;
	for (int i = 0; i < CorpusData.nCameras; i++)
	{
		GetRTFromrt(CorpusData.camera[i].rt, CorpusData.camera[i].R, CorpusData.camera[i].T);
		GetKFromIntrinsic(CorpusData.camera[i].K, CorpusData.camera[i].intrinsic);
	}

	char Fname[512];
	sprintf(Fname, "%s/Recon/synth_%d.out", Path, SynthID);
	FILE *fp = fopen(Fname, "w+");
	if (fp == NULL)
	{
		printf("Cannot write %s\n", Fname);
		return 1;
	}
	printf("Start saving...");

	fprintf(fp, "drews 1.0\n");
	fprintf(fp, "%d %d\n", CorpusData.nCameras, CorpusData.n3dPoints);

	vector<int> camIDs; camIDs.push_back(26), camIDs.push_back(28), camIDs.push_back(30), camIDs.push_back(32), camIDs.push_back(22), camIDs.push_back(24), camIDs.push_back(20);
	for (int i = 0; i < CorpusData.nCameras; i++)
	{
		/*int found = 0;
		for (int ii = 0; found == 0 && ii < camIDs.size(); ii++)
		{
		if (i == camIDs[ii])
		found = 1;
		}

		if (found ==0)
		{
		fprintf(fp, "%.16f %.16f %.16f\n", CorpusData.camera[i].intrinsic[0] / max(CorpusData.camera[i].width, CorpusData.camera[i].height), CorpusData.camera[i].distortion[0], CorpusData.camera[i].distortion[1]);
		for (int ii = 0; ii < 9; ii++)
		fprintf(fp, "%.16f ", 0.0);
		fprintf(fp, "\n");
		for (int ii = 0; ii < 3; ii++)
		fprintf(fp, "%.16f ", 0.0);
		fprintf(fp, "\n");
		}
		else
		{*/
		fprintf(fp, "%.16f %.16f %.16f\n", CorpusData.camera[i].intrinsic[0] / max(CorpusData.camera[i].width, CorpusData.camera[i].height), CorpusData.camera[i].distortion[0], CorpusData.camera[i].distortion[1]);
		for (int ii = 0; ii < 9; ii++)
			fprintf(fp, "%.16f ", CorpusData.camera[i].R[ii]);
		fprintf(fp, "\n");
		for (int ii = 0; ii < 3; ii++)
			fprintf(fp, "%.16f ", CorpusData.camera[i].T[ii]);
		fprintf(fp, "\n");
		//}
	}

	for (int i = 0; i < CorpusData.n3dPoints; i++)
	{
		Point3d xyz;
		fprintf(fp, "%.16f %.16f %.16f\n", CorpusData.xyz[i].x, CorpusData.xyz[i].y, CorpusData.xyz[i].z);
		fprintf(fp, "%d %d %d\n", CorpusData.rgb[i].x, CorpusData.rgb[i].y, CorpusData.rgb[i].z);

		fprintf(fp, "%d ", CorpusData.nEleAll3D[i]);
		for (int j = 0; j < CorpusData.nEleAll3D[i]; j++)
			fprintf(fp, "%d %d %f ", CorpusData.viewIdAll3D[i][j], CorpusData.pointIdAll3D[i][j], CorpusData.scaleAll3D[i][j]);
		fprintf(fp, "\n");
	} // for i
	fclose(fp);

	printf("done.\n");

	return 0;
}
bool WriteGridToImage(char *fname, unsigned char *Img, int width, int height, int nchannels)
{
	int ii, jj, kk, length = width*height;

	Mat M = Mat::zeros(height, width, nchannels == 1 ? CV_8UC1 : CV_8UC3);
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				M.data[nchannels*ii + kk + nchannels*jj*width] = Img[ii + jj*width + kk*length];

	return imwrite(fname, M);
}
bool WriteGridToImage(char *fname, double *Img, int width, int height, int nchannels)
{
	int ii, jj, kk, length = width*height;

	Mat M = Mat::zeros(height, width, nchannels == 1 ? CV_8UC1 : CV_8UC3);
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				M.data[nchannels*ii + kk + nchannels*jj*width] = (unsigned char)(int)(Img[ii + jj*width + kk*length] + 0.5);

	return imwrite(fname, M);
}

bool ReadFlowDataBinary(char *fnX, char *fnY, Point2f *fxy, int width, int height)
{
	float u, v;

	ifstream fin1, fin2;
	fin1.open(fnX, ios::binary);
	if (!fin1.is_open())
	{
		cout << "Cannot load: " << fnX << endl;
		return false;
	}
	fin2.open(fnY, ios::binary);
	if (!fin2.is_open())
	{
		cout << "Cannot load: " << fnY << endl;
		return false;
	}

	for (int j = 0; j < height; ++j)
	{
		for (int i = 0; i < width; ++i)
		{
			fin1.read(reinterpret_cast<char *>(&u), sizeof(float));
			fin2.read(reinterpret_cast<char *>(&v), sizeof(float));

			fxy[i + j*width].x = u;
			fxy[i + j*width].y = v;
		}
	}
	fin1.close();
	fin2.close();

	return true;
}

int  read_pfm_file(const std::string& filename, ImgData &depthmap)
{
	int w, h, l;
	double scale = 1.0;

	char buf[256];
	FILE *f;
	fopen_s(&f, filename.c_str(), "rb");
	if (f == NULL)
	{
		wprintf(L"PFM file absent: %s\n\n", filename.c_str());
		return 1;
	}

	int channel = 1;
	fscanf(f, "%s\n", buf);
	if (strcmp(buf, "Pf") == 0)
		channel = 1;
	else if (strcmp(buf, "PF") == 0)
		channel = 3;
	else
	{
		printf(buf);
		printf("Not a 1/3 channel PFM file.\n");
		return 2;
	}
	fscanf(f, "%d %d\n", &w, &h);
	l = w*h;
	fscanf(f, "%lf\n", &scale);
	int little_endian = 0;
	if (scale < 0.0)
	{
		little_endian = 1;
		scale = -scale;
	}
	size_t datasize = w*h*channel;
	std::vector<uchar> data(datasize * sizeof(float));

	depthmap.width = w, depthmap.height = h;
	depthmap.depth = new float[w*h*channel];
	size_t count = fread((void *)&data[0], sizeof(float), datasize, f);
	if (count != datasize)
	{
		printf("Could not read ground truth file.\n");
		return 3;
	}
	int native_little_endian = is_little_endian();

	for (int i = 0; i < datasize; i++)
	{
		uchar *p = &data[i * 4];
		if (little_endian != native_little_endian)
		{
			uchar temp;
			temp = p[0]; p[0] = p[3]; p[3] = temp;
			temp = p[1]; p[1] = p[2]; p[2] = temp;
		}
		int jj = (i / channel) % w;
		int ii = (i / channel) / w;
		int ch = i % channel;
		depthmap.depth[jj + (h - 1 - ii)*w + l*ch] = *((float *)p);
	}
	fclose(f);
	return 0;
}
cv::Mat read_pfm_file(const std::string& filename)
{
	int w, h;
	double scale = 1.0;

	char buf[256];
	FILE *f;
	fopen_s(&f, filename.c_str(), "rb");
	if (f == NULL)
	{
		wprintf(L"PFM file absent: %s\n\n", filename.c_str());
		return cv::Mat();
	}

	int channel = 1;
	fscanf(f, "%s\n", buf);
	if (strcmp(buf, "Pf") == 0)
		channel = 1;
	else if (strcmp(buf, "PF") == 0)
		channel = 3;
	else
	{
		printf(buf);
		printf("Not a 1/3 channel PFM file.\n");
		return cv::Mat();
	}
	fscanf(f, "%d %d\n", &w, &h);
	fscanf(f, "%lf\n", &scale);
	int little_endian = 0;
	if (scale < 0.0)
	{
		little_endian = 1;
		scale = -scale;
	}
	size_t datasize = w*h*channel;
	std::vector<uchar> data(datasize * sizeof(float));

	cv::Mat image = cv::Mat(h, w, CV_MAKE_TYPE(CV_32F, channel));
	size_t count = fread((void *)&data[0], sizeof(float), datasize, f);
	if (count != datasize)
	{
		printf("Could not read ground truth file.\n");
		return cv::Mat();
	}
	int native_little_endian = is_little_endian();
	for (int i = 0; i < datasize; i++)
	{
		uchar *p = &data[i * 4];
		if (little_endian != native_little_endian)
		{
			uchar temp;
			temp = p[0]; p[0] = p[3]; p[3] = temp;
			temp = p[1]; p[1] = p[2]; p[2] = temp;
		}
		int jj = (i / channel) % w;
		int ii = (i / channel) / w;
		int ch = i % channel;
		image.at<float>(h - 1 - ii, jj*channel + ch) = *((float *)p);
	}
	fclose(f);
	return image;
}
void save_pfm_file(const std::string& filename, const cv::Mat& image)
{
	int width = image.cols;
	int height = image.rows;

	FILE *stream;
	fopen_s(&stream, filename.c_str(), "rb");
	if (stream == NULL)
	{
		wprintf(L"PFM file absent: %s\n\n", filename.c_str());
		return;
	}
	// write the header: 3 lines: Pf, dimensions, scale factor (negative val == little endian)
	int channel = image.channels();
	if (channel == 1)
		fprintf(stream, "Pf\n%d %d\n%lf\n", width, height, -1.0 / 255.0);
	else if (channel == 3)
		fprintf(stream, "PF\n%d %d\n%lf\n", width, height, -1.0 / 255.0);
	else {
		printf("Channels %d must be 1 or 3\n", image.channels());
		return;
	}


	// pfm stores rows in inverse order!
	int linesize = width*channel;
	std::vector<float> rowBuff(linesize);
	for (int y = height - 1; y >= 0; y--)
	{
		auto ptr = image.ptr<float>(y);
		auto pBuf = &rowBuff[0];
		for (int x = 0; x < linesize; x++)
		{
			float val = (float)(*ptr);
			pBuf[x] = val;
			ptr++;
			/*if (val > 0 && val <= 255)
			rowBuf[x] = val;
			else
			{
			printf("invalid: val %f\n", flo(x,y));
			rowBuf[x] = 0.0f;
			}*/
		}
		if ((int)fwrite(&rowBuff[0], sizeof(float), width, stream) != width)
			printf("[ERROR] problem with fwrite.");

		fflush(stream);
	}

	fclose(stream);
	return;
}






