#if !defined(IMGULTI_H )
#define IMGULTI_H
#pragma once

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <omp.h>
#include <stdarg.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <Eigen/Dense>

#include "DataStructure.h"

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

using namespace cv;
using namespace std;

template <typename T> void BuildDataPyramid(T *Data, vector<T*> &DataPyr, int width, int height, int nscales, bool discrete = false)
{
	//down sample by factor of 2 at every scale

	int upperWidth = width, upperHeight = height;
	DataPyr.push_back(Data); //level 0

	for (int sid = 1; sid < nscales + 1; sid++)
	{
		int activeWidth = (int)ceil(1.0*upperWidth / 2), activeHeight = (int)ceil(1.0*upperHeight / 2);

		T *DataAtSPlus = DataPyr[sid - 1];
		T *DataAtS = new T[activeWidth*activeHeight];

		for (int jj = 0; jj < activeHeight; jj++)
		{
			for (int ii = 0; ii < activeWidth; ii++)
			{
				if (2 * ii + 1 > upperWidth - 1 || 2 * jj + 1>upperHeight - 1)
					DataAtS[ii + jj*activeWidth] = 0.0;
				else
				{
					if (!discrete)
						DataAtS[ii + jj*activeWidth] = 0.25*(DataAtSPlus[2 * ii + 2 * jj*upperWidth] + DataAtSPlus[2 * ii + 1 + 2 * jj*upperWidth] +
						DataAtSPlus[2 * ii + (2 * jj + 1)*upperWidth] + DataAtSPlus[2 * ii + 1 + (2 * jj + 1)*upperWidth]);// DataAtSPlus[2 * ii + 1 + (2 * jj + 1)*upperWidth]; //Detph does not get blurred
					else
					{
						DataAtS[ii + jj*activeWidth] = false;
						if (DataAtSPlus[2 * ii + 2 * jj*upperWidth] || DataAtSPlus[2 * ii + 1 + 2 * jj*upperWidth] ||
							DataAtSPlus[2 * ii + (2 * jj + 1)*upperWidth] || DataAtSPlus[2 * ii + 1 + (2 * jj + 1)*upperWidth])
							DataAtS[ii + jj*activeWidth] = true;
					}
				}
			}
		}

		upperWidth = activeWidth, upperHeight = activeHeight;
		DataPyr.push_back(DataAtS);

	}

	return;
}

template <typename T> void UpsamleDepth(T *depthIn, T* depthOut, int lowerWidth, int lowerHeight, int upperWidth, int upperHeight)
{
	//upsample by factor of 2 at every scale

	for (int jj = 0; jj < lowerHeight; jj++)
	{
		for (int ii = 0; ii < lowerWidth; ii++)
		{
			depthOut[2 * ii + 2 * jj *upperWidth] = depthIn[ii + jj*lowerWidth];
			if (2 * ii + 1 < upperWidth && 2 * jj < upperHeight)
				depthOut[2 * ii + 1 + 2 * jj *upperWidth] = depthIn[ii + jj*lowerWidth];
			if (2 * ii < upperWidth && 2 * jj + 1 < upperHeight)
				depthOut[2 * ii + (2 * jj + 1)*upperWidth] = depthIn[ii + jj*lowerWidth];
			if (2 * ii + 1 < upperWidth && 2 * jj + 1 < upperHeight)
				depthOut[2 * ii + 1 + (2 * jj + 1)*upperWidth] = depthIn[ii + jj*lowerWidth];
		}
	}

	return;
}
template <typename T> void UpsamleDepthNN(T *depthIn, T* depthOut, int *maskIn, int *maskOut, int upperWidth, int upperHeight, int lowerWidth, int lowerHeight)
{
	vector<Point2i> indexList;
	indexList.push_back(Point2i(0, 0));
	for (int range = 1; range < 3; range++)
		for (int r = -range; r <= range; r++)
			for (int c = -range; c <= range; c++)
				if (abs(r) >= range || abs(c) >= range)
					indexList.push_back(Point2i(r, c));

	//upsample by factor of 2 at every scale
	for (int jj = 0; jj < lowerHeight; jj++)
	{
		for (int ii = 0; ii < lowerWidth; ii++)
		{
			if (maskOut[ii + jj*lowerWidth]>0)
			{
				bool found = false;
				int x0 = ii / 2, y0 = jj / 2;
				for (auto rc : indexList)
				{
					if (maskIn[x0 + rc.x + (y0 + rc.y)*upperWidth]>0)
					{
						depthOut[ii + jj *lowerWidth] = depthIn[x0 + rc.x + (y0 + rc.y)*upperWidth];
						found = true;
						break;
					}
				}
				if (!found)
					printf("Problem with depth upsampling @: (%d %d)\n", ii, jj);
			}
		}
	}

	return;
}
void ComputeGeodesicWeight(uchar *img, bool *mask, int width, int nchannels, float *localWeights, int cx, int cy, int hb = 31, double sigma = 50.0, int numPass = 3);
void BucketGoodFeaturesToTrack(Mat Img, vector<Point2f> &Corners, int nImagePartitions, int maxCorners, double qualityLevel, double minDistance, int blockSize = 3, bool useHarrisDetector = false, double harrisK = 0.04);

void LensCorrectionPoint(Point2d *uv, double *K, double *distortion, int npts);
void LensDistortionPoint(Point2d *img_point, double *K, double *distortion, int npts);
void LensCorrection(vector<ImgData> &vImg, double *Intrinsic, double *distortion);

#endif
