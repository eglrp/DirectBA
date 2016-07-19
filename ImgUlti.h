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

template <typename T> void BuildDepthPyramid(T *depth, vector<T*> &depthPyr, int width, int height, int nscales)
{
	//down sample by factor of 2 at every scale

	int upperWidth = width, upperHeight = height;
	depthPyr.push_back(depth); //level 0

	for (int sid = 1; sid < nscales + 1; sid++)
	{
		int activeWidth = (int)ceil(1.0*upperWidth / 2), activeHeight = (int)ceil(1.0*upperHeight / 2);

		T *depthAtSPlus = depthPyr[sid - 1];
		T *depthAtS = new T[activeWidth*activeHeight];

		for (int jj = 0; jj < activeHeight; jj++)
		{
			for (int ii = 0; ii < activeWidth; ii++)
			{
				//depthAtS[ii + jj*activeWidth] = 0.25*(depthAtSPlus[2 * ii + 2 * jj*upperWidth]
				//	+ depthAtSPlus[2 * ii + 1 + 2 * jj*upperWidth] +
				//	depthAtSPlus[2 * ii + (2 * jj + 1)*upperWidth] +
				//	depthAtSPlus[2 * ii + 1 + (2 * jj + 1)*upperWidth]);
				if (2 * ii + 1 > upperWidth - 1 || 2 * jj + 1>upperHeight - 1)
					depthAtS[ii + jj*activeWidth] = 0.0;
				else
					depthAtS[ii + jj*activeWidth] = depthAtSPlus[2 * ii + 1 + (2 * jj + 1)*upperWidth]; //depth does not get blurred
			}
		}

		upperWidth = activeWidth, upperHeight = activeHeight;
		depthPyr.push_back(depthAtS);

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

#endif
