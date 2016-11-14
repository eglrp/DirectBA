#include "ImgUlti.h"

using namespace std;
using namespace cv;
using namespace Eigen;

void ComputeGeodesicWeight(uchar *img, bool *mask, int width, int nchannels, float *localWeights, int cx, int cy, int hb, double sigma, int numPass)
{
	int kernelSize = 8;
	double K1[8] = { -1, -1, 0, -1, 1, -1, -1, 0 }, K2[8] = { -1, 1, 0, 1, 1, 1, 1, 0 };

	int winSize = 2 * hb + 1;

	// Initialize localWeights to large values, and center pixel to 0
	for (int y = 0; y < winSize*winSize; ++y)
		localWeights[y] = 1e9;
	localWeights[hb + hb*winSize] = 0.0;

	uchar color1[3], color2[3];
	// Approximate geodesic distance
	for (int iter = 0; iter < numPass; ++iter)
	{
		// Forward pass
		for (int y = -hb; y <= hb; ++y)
		{
			for (int x = -hb; x <= hb; ++x)
			{
				int xcx = x + cx, ycy = y + cy;
				if (!mask[xcx + ycy*width])
					continue;
				for (int kk = 0; kk < nchannels; kk++)
					color1[kk] = img[(xcx + (ycy)*width)*nchannels + kk];

				float &weight = localWeights[x + hb + (y + hb)*winSize];
				for (int ind = 0; ind < kernelSize; ind += 2)
				{
					int dx = K1[ind + 0], dy = K1[ind + 1], xdx = x + dx, ydy = y + dy;
					if (xdx > hb || ydy > hb || xdx < -hb || ydy < -hb)
						continue;
					if (!mask[xcx + dx + (ycy + dy)*width])
						continue;

					for (int kk = 0; kk < nchannels; kk++)
						color2[kk] = img[(xcx + dx + (ycy + dy)*width)*nchannels + kk];

					float diff = 0;
					for (int kk = 0; kk < nchannels; kk++)
						diff += pow((float)(color1[kk] - color2[kk]), 2);
					float cost = localWeights[xdx + hb + (ydy + hb)*winSize];
					weight = min(weight, cost + diff);
				}
			}
		}

		// Backwards pass
		for (int y = hb; y >= -hb; --y)
		{
			for (int x = hb; x >= -hb; --x)
			{
				int xcx = x + cx, ycy = y + cy;
				if (!mask[xcx + ycy*width])
					continue;
				for (int kk = 0; kk < nchannels; kk++)
					color1[kk] = img[(xcx + (ycy)*width)*nchannels + kk];

				float &weight = localWeights[x + hb + (y + hb)*winSize];
				for (int ind = 0; ind < kernelSize; ind += 2)
				{
					int dx = K2[ind + 0], dy = K2[ind + 1], xdx = x + dx, ydy = y + dy;
					if (xdx > hb || ydy > hb || xdx < -hb || ydy < -hb)
						continue;
					if (!mask[xcx + dx + (ycy + dy)*width])
						continue;

					for (int kk = 0; kk < nchannels; kk++)
						color2[kk] = img[(xcx + dx + (ycy + dy)*width)*nchannels + kk];

					float diff = 0;
					for (int kk = 0; kk < nchannels; kk++)
						diff += pow((float)(color1[kk] - color2[kk]), 2);
					float cost = localWeights[xdx + hb + (ydy + hb)*winSize];
					weight = min(weight, cost + diff);
				}
			}
		}
	}

	// Exponential weighting
	for (int y = 0; y < winSize; ++y)
		for (int x = 0; x < winSize; ++x)
			localWeights[x + y*winSize] = exp(-sqrt(localWeights[x + y*winSize]) / sigma);

	return;
}
void BucketGoodFeaturesToTrack(Mat Img, vector<Point2f> &Corners, int nImagePartitions, int maxCorners, double qualityLevel, double minDistance, int blockSize, bool useHarrisDetector, double harrisK)
{
	int width = Img.cols, height = Img.rows;
	Mat Img2;

	Corners.reserve(maxCorners);

	int partionSize = max(width, height) / nImagePartitions;
	for (int jj = 0; jj < nImagePartitions; jj++)
	{
		for (int ii = 0; ii < nImagePartitions; ii++)
		{
			vector<Point2f> RegionCorners; RegionCorners.reserve(maxCorners);

			Mat mask(height, width, CV_8UC1, Scalar(0, 0, 0));
			for (int kk = jj*partionSize; kk < (jj + 1)*partionSize; kk++)
			{
				for (int ll = ii*partionSize; ll < (ii + 1)*partionSize; ll++)
				{
					if (kk > height - 1 || ll > width - 1)
						continue;
					mask.data[ll + kk*width] = 255;
				}
			}
			goodFeaturesToTrack(Img, RegionCorners, maxCorners, qualityLevel, max(minDistance, minDistance* width / 1920), mask, max(blockSize, blockSize* width / 1920), useHarrisDetector, harrisK);
			/*cvtColor(Img, Img2, CV_GRAY2BGR);
			for (int kk = 0; kk < (int)uvRef.size(); kk++)
			circle(Img2, uvRef[kk], 5, Scalar(0, 255, 0), 2, 8, 0);
			cvNamedWindow("X", WINDOW_NORMAL);
			imshow("X", Img2); waitKey(0);*/

			for (int kk = 0; kk < (int)RegionCorners.size(); kk++)
				Corners.push_back(RegionCorners[kk]);
		}
	}

	return;
}
void DetectCornersHarris(char *img, int width, int height, Point2d *HarrisC, int &npts, double sigma, double sigmaD, double thresh, double alpha, int SuppressType, double AMN_thresh)
{
	/*int i, j, ii, jj;

	const int nMaxCorners = 100000;

	int size = 2 * (int)(3.0*sigma + 0.5) + 1, kk = (size - 1) / 2;
	double *GKernel = new double[size];
	double *GDKernel = new double[size];
	double t, sigma2 = sigma*sigma, sigma3 = sigma*sigma*sigma;

	for (ii = -kk; ii <= kk; ii++)
	{
	GKernel[ii + kk] = exp(-(ii*ii) / (2.0*sigma2)) / (sqrt(2.0*Pi)*sigma);
	GDKernel[ii + kk] = 1.0*ii*exp(-(ii*ii) / (2.0*sigma2)) / (sqrt(2.0*Pi)*sigma3);
	}

	//Compute Ix, Iy throught gaussian filters
	double *temp = new double[width*height];
	double *Ix = new double[width*height];
	double *Iy = new double[width*height];

	filter1D_row(GDKernel, size, img, temp, width, height);
	filter1D_col(GKernel, size, temp, Ix, width, height, t);

	filter1D_row(GKernel, size, img, temp, width, height);
	filter1D_col(GDKernel, size, temp, Iy, width, height, t);

	//Compute Ix2, Iy2, Ixy throught gaussian filters
	size = 2 * (int)(3.0*sigmaD + 0.5) + 1, kk = (size - 1) / 2;
	double *GKernel2 = new double[size];


	for (ii = -kk; ii <= kk; ii++)
	GKernel2[ii + kk] = exp(-(ii*ii) / (2.0*sigmaD*sigmaD)) / (sqrt(2.0*Pi)*sigmaD);

	double *Ix2 = new double[width*height];
	double *Iy2 = new double[width*height];
	double *Ixy = new double[width*height];

	for (jj = 0; jj < height; jj++)
	{
	for (ii = 0; ii < width; ii++)
	{
	Ix2[ii + jj*width] = Ix[ii + jj*width] * Ix[ii + jj*width];
	Iy2[ii + jj*width] = Iy[ii + jj*width] * Iy[ii + jj*width];
	Ixy[ii + jj*width] = Ix[ii + jj*width] * Iy[ii + jj*width];
	}
	}

	filter1D_row_Double(GKernel2, size, Ix2, temp, width, height);
	filter1D_col(GKernel2, size, temp, Ix2, width, height, t);

	filter1D_row_Double(GKernel2, size, Iy2, temp, width, height);
	filter1D_col(GKernel2, size, temp, Iy2, width, height, t);

	filter1D_row_Double(GKernel2, size, Ixy, temp, width, height);
	filter1D_col(GKernel2, size, temp, Ixy, width, height, t);

	double *tr = new double[width*height];
	double *Det = new double[width*height];
	for (jj = kk + 10; jj < height - kk - 10; jj++)
	{
	for (ii = kk + 10; ii < width - kk - 10; ii++)
	{
	tr[ii + jj*width] = Ix2[ii + jj*width] + Iy2[ii + jj*width];
	Det[ii + jj*width] = Ix2[ii + jj*width] * Iy2[ii + jj*width] - Ixy[ii + jj*width] * Ixy[ii + jj*width];
	}
	}

	double maxRes = 0.0;
	double *Cornerness = new double[width*height];
	for (jj = kk + 10; jj < height - kk - 10; jj++)
	{
	for (ii = kk + 10; ii < width - kk - 10; ii++)
	{
	Cornerness[ii + jj*width] = Det[ii + jj*width] - alpha*tr[ii + jj*width] * tr[ii + jj*width];
	if (maxRes < Cornerness[ii + jj*width])
	maxRes = Cornerness[ii + jj*width];
	}
	}*/
	/*
	FILE *fp = fopen("C:/temp/cornerness.txt", "w+");
	for(jj=kk+10; jj<height-kk-10; jj++)
	{
	for(ii=kk+10; ii<width-kk-10; ii++)
	{
	fprintf(fp, "%.4f ", Cornerness[ii+jj*width]);
	if(Cornerness[ii+jj*width] <thresh*maxRes)
	Cornerness[ii+jj*width] = 0.0;
	}
	fprintf(fp, "\n");
	}
	fclose(fp);
	*/
	/*int npotentialCorners = 0;
	int *potentialCorners = new int[2 * nMaxCorners];
	float *Response = new float[nMaxCorners];
	if (SuppressType == 0)
	{
	for (jj = kk + 10; jj < height - kk - 10; jj++)
	{
	for (ii = kk + 10; ii < width - kk - 10; ii++)
	{
	if (Cornerness[ii + jj*width] > maxRes*0.2)
	{
	potentialCorners[2 * npotentialCorners] = ii;
	potentialCorners[2 * npotentialCorners + 1] = jj;
	Response[npotentialCorners] = (float)Cornerness[ii + jj*width];
	npotentialCorners++;
	}
	if (npotentialCorners + 1 > nMaxCorners)
	break;
	}
	if (npotentialCorners + 1 > nMaxCorners)
	break;
	}


	// The orginal paper recommends a threshold of 0.9 but that results in some very close points, which have similar responses, to pass the test.
	float *Mdist = new float[npotentialCorners*(npotentialCorners + 1) / 2]; // Special handling needed for this square symetric matrix due to large nkpts
	float *pt_uv = new float[2 * npotentialCorners];
	int *ind = new int[npotentialCorners];
	float *radius2 = new float[npotentialCorners];

	int pt1u, pt1v, pt2u, pt2v;
	int index = 0;
	for (ii = 0; ii < npotentialCorners; ii++)
	{
	pt1u = potentialCorners[2 * ii];
	pt1v = potentialCorners[2 * ii + 1];
	index++;

	for (jj = ii + 1; jj < npotentialCorners; jj++)
	{
	pt2u = potentialCorners[2 * jj];
	pt2v = potentialCorners[2 * jj + 1];
	Mdist[index] = 1.0f*(pt1u - pt2u)*(pt1u - pt2u) + (pt1v - pt2v)*(pt1v - pt2v);
	index++;
	}
	}

	float response1, response2, r2;
	for (i = 0; i < npotentialCorners; i++)
	{
	r2 = 1.0e9;
	response1 = Response[i];

	for (j = 0; j < npotentialCorners; j++)
	{
	if (i == j)
	continue;

	response2 = Response[j];
	if (response1 < AMN_thresh * response2)
	{
	if (i > j)
	{
	ii = j;
	jj = i;
	}
	else
	{
	ii = i;
	jj = j;
	}
	index = (2 * npotentialCorners - ii + 1)*ii / 2 + jj - ii;
	r2 = min(r2, Mdist[index]);
	}
	}
	radius2[i] = -r2; //descending order
	ind[i] = i;
	}

	int ANM_pts = 1000;
	Quick_Sort_Float(radius2, ind, 0, npotentialCorners - 1);
	for (i = 0; i < ANM_pts; i++)
	{
	HarrisC[i].x = potentialCorners[2 * ind[i]];
	HarrisC[i].y = potentialCorners[2 * ind[i] + 1];
	}
	npts = ANM_pts;

	delete[]Mdist;
	delete[]pt_uv;
	delete[]ind;
	delete[]radius2;
	}
	else if (SuppressType == 1)
	{
	for (jj = kk + 10; jj < height - kk - 10; jj++)
	{
	for (ii = kk + 10; ii < width - kk - 10; ii++)
	{
	if (Cornerness[ii + jj*width] > maxRes*0.2)
	{
	potentialCorners[2 * npotentialCorners] = ii;
	potentialCorners[2 * npotentialCorners + 1] = jj;
	Response[npotentialCorners] = (float)Cornerness[ii + jj*width];
	npotentialCorners++;
	}
	if (npotentialCorners + 1 > nMaxCorners)
	break;
	}
	if (npotentialCorners + 1 > nMaxCorners)
	break;
	}

	bool breakflag;
	int x, y;
	float *Response2 = new float[npotentialCorners];
	for (kk = 0; kk < npotentialCorners; kk++)
	{
	x = potentialCorners[2 * kk];
	y = potentialCorners[2 * kk + 1];
	Response2[kk] = Response[kk];
	breakflag = false;
	for (jj = -1; jj < 2; jj++)
	{
	for (ii = -1; ii < 2; ii++)
	{
	if (Response[kk] < Cornerness[x + ii + (y + jj)*width] - 1.0)
	{
	Response2[kk] = 0.0;
	breakflag = true;
	break;
	}
	}
	if (breakflag == true)
	break;
	}
	}

	npts = 0;
	for (kk = 0; kk<npotentialCorners; kk++)
	{
	if (Response2[kk] > maxRes*0.2)
	{
	HarrisC[npts].x = potentialCorners[2 * kk];
	HarrisC[npts].y = potentialCorners[2 * kk + 1];
	npts++;
	}
	if (npts > nMaxCorners)
	break;
	}
	delete[]Response2;
	}
	else
	{
	npts = 0;
	for (jj = kk + 10; jj < height - kk - 10; jj++)
	{
	for (ii = kk + 10; ii < width - kk - 10; ii++)
	{
	if (Cornerness[ii + jj*width] > maxRes*0.2)
	{
	HarrisC[npts].x = ii;
	HarrisC[npts].y = jj;
	npts++;
	}
	if (npts > nMaxCorners)
	break;
	}
	if (npts > nMaxCorners)
	break;
	}
	}

	delete[]GKernel;
	delete[]GDKernel;
	delete[]GKernel2;
	delete[]temp;
	delete[]Ix;
	delete[]Iy;
	delete[]Ix2;
	delete[]Iy2;
	delete[]Ixy;
	delete[]tr;
	delete[]Det;
	delete[]Cornerness;
	delete[]potentialCorners;
	delete[]Response;*/

	return;
}

double InitialCausalCoefficient(double *sample, int length, double pole, double tolerance)
{
	double zn, iz, z2n;
	double FirstCausalCoef;
	int n, horizon;
	horizon = (int)(ceil(log(tolerance) / log(fabs(pole))) + 0.01);
	if (horizon < length) {
		/* accelerated loop */
		zn = pole;
		FirstCausalCoef = *(sample);
		for (n = 1; n < horizon; n++) {
			FirstCausalCoef += zn * (*(sample + n));
			zn *= pole;
		}
	}
	else {
		/* full loop */
		zn = pole;
		iz = 1.0 / pole;
		z2n = pow(pole, (double)(length - 1));
		FirstCausalCoef = sample[0] + z2n * sample[length - 1];
		z2n *= z2n * iz;
		for (n = 1; n <= length - 2; n++) {
			FirstCausalCoef += (zn + z2n) * sample[n];
			zn *= pole;
			z2n *= iz;
		}
	}
	return FirstCausalCoef;
}
double InitialAnticausalCoefficient(double *CausalCoef, int length, double pole)
{
	return((pole / (pole * pole - 1.0)) * (pole * CausalCoef[length - 2] + CausalCoef[length - 1]));
}
// prefilter for 4-tap, 6-tap, 8-tap, optimized 4-tap, and optimized 6-tap
void Prefilter_1D(double *coefficient, int length, double *pole, double tolerance, int nPoles)
{
	int i, n, k;
	double Lambda;
	Lambda = 1;
	if (length == 1)
		return;
	/* compute the overall gain */
	for (k = 0; k < nPoles; k++)
		Lambda = Lambda * (1.0 - pole[k]) * (1.0 - 1.0 / pole[k]);

	// Applying the gain to original image
	for (i = 0; i < length; i++)
		*(coefficient + i) = (*(coefficient + i)) * Lambda;

	for (k = 0; k < nPoles; k++)
	{
		// Compute the first causal coefficient
		*(coefficient) = InitialCausalCoefficient(coefficient, length, pole[k], tolerance);

		// Causal prefilter
		for (n = 1; n < length; n++)
			coefficient[n] += pole[k] * coefficient[n - 1];

		//Compute the first anticausal coefficient
		*(coefficient + length - 1) = InitialAnticausalCoefficient(coefficient, length, pole[k]);

		//Anticausal prefilter
		for (n = length - 2; n >= 0; n--)
			coefficient[n] = pole[k] * (coefficient[n + 1] - coefficient[n]);
	}
}
// Prefilter for modified 4-tap
void Prefilter_1Dm(double *coefficient, int length, double *pole, double tolerance, double gamma)
{
	int i, n, k;
	double Lambda;
	Lambda = 6.0 / (6.0 * gamma + 1.0);
	if (length == 1)
		return;

	// Applying the gain to original image
	for (i = 0; i < length; i++)
		*(coefficient + i) = (*(coefficient + i)) * Lambda;

	for (k = 0; k < 1; k++)
	{
		// Compute the first causal coefficient
		*(coefficient) = InitialCausalCoefficient(coefficient, length, pole[k], tolerance);

		// Causal prefilter
		for (n = 1; n < length; n++)
			coefficient[n] += pole[k] * coefficient[n - 1];

		//Compute the first anticausal coefficient
		*(coefficient + length - 1) = InitialAnticausalCoefficient(coefficient, length, pole[k]);

		//Anticausal prefilter
		for (n = length - 2; n >= 0; n--)
			coefficient[n] = pole[k] * (coefficient[n + 1] - coefficient[n]);
	}
}

void Generate_Para_Spline(double *Image, double *Para, int width, int height, int Interpolation_Algorithm)
{
	double tolerance;
	int i, j, nPoles;
	int length = width * height;
	double pole[2], a, gamma;
	tolerance = 1e-4;
	if (Interpolation_Algorithm == 1) // 4-tap
	{
		nPoles = 1;
		pole[0] = sqrt(3.0) - 2.0;
	}
	else if (Interpolation_Algorithm == 2) // 6-tap
	{
		nPoles = 2;
		pole[0] = sqrt(664.0 - sqrt(438976.0)) + sqrt(304.0) - 19.0;
		pole[1] = sqrt(664.0 + sqrt(438976.0)) - sqrt(304.0) - 19.0;
	}
	else if (Interpolation_Algorithm == 3) // modified 4-tap
	{
		gamma = 0.0409;
		a = (4.0 - 12.0 * gamma) / (6.0 * gamma + 1.0);
		pole[0] = (-a + sqrt(a * a - 4)) / 2.0;
	}
	else if (Interpolation_Algorithm == 4) // optimized 4-tap
	{
		nPoles = 1;
		pole[0] = (-13.0 + sqrt(105.0)) / 8.0;
	}
	else if (Interpolation_Algorithm == 5) // optimized 6-tap
	{
		nPoles = 2;
		pole[0] = -0.410549185795627524168;
		pole[1] = -0.0316849091024414351363;
	}
	else if (Interpolation_Algorithm == 6) // 8-tap
	{
		nPoles = 2;
		pole[0] = sqrt(135.0 / 2.0 - sqrt(17745.0 / 4.0)) + sqrt(105.0 / 4.0) - 13.0 / 2.0;
		pole[1] = sqrt(135.0 / 2.0 + sqrt(17745.0 / 4.0)) - sqrt(105.0 / 4.0) - 13.0 / 2.0;
	}

	//Perform the 1D prefiltering along the rows
	double *LineWidth = new double[width];
	for (i = 0; i < height; i++)
	{
		//Prefiltering each row
		for (j = 0; j < width; j++)
		{
			*(LineWidth + j) = *(Image + i * width + j);
		}
		if (Interpolation_Algorithm == 3)
			Prefilter_1Dm(LineWidth, width, pole, tolerance, gamma);
		else
			Prefilter_1D(LineWidth, width, pole, tolerance, nPoles);

		// Put the prefiltered coeffiecients into Para array
		for (j = 0; j < width; j++)
		{
			*(Para + i * width + j) = (*(LineWidth + j));
		}
	}
	delete[]LineWidth;

	//Perform the 1D prefiltering along the columns
	double *LineHeight = new double[height];
	for (i = 0; i < width; i++)
	{
		//Prefiltering each comlumn
		for (j = 0; j < height; j++)
		{
			*(LineHeight + j) = (*(Para + j * width + i));
		}
		if (Interpolation_Algorithm == 3)
			Prefilter_1Dm(LineHeight, height, pole, tolerance, gamma);
		else
			Prefilter_1D(LineHeight, height, pole, tolerance, nPoles);

		//Put the prefilterd coefficients into the Para array
		for (j = 0; j < height; j++)
		{
			*(Para + j * width + i) = (*(LineHeight + j));
		}
	}
	delete[]LineHeight;
	return;
}
void Generate_Para_Spline(char *Image, double *Para, int width, int height, int Interpolation_Algorithm)
{
	int i, length = width*height;
	double *Image_2 = new double[length];
	for (i = 0; i < length; i++)
		*(Image_2 + i) = (double)((int)((unsigned char)(*(Image + i))));

	Generate_Para_Spline(Image_2, Para, width, height, Interpolation_Algorithm);

	delete[]Image_2;
	return;
}
void Generate_Para_Spline(unsigned char *Image, double *Para, int width, int height, int Interpolation_Algorithm)
{
	int i, length = width*height;
	double *Image_2 = new double[length];
	for (i = 0; i < length; i++)
		*(Image_2 + i) = (double)((int)(*(Image + i)));

	Generate_Para_Spline(Image_2, Para, width, height, Interpolation_Algorithm);

	delete[]Image_2;
	return;
}
void Generate_Para_Spline(int *Image, double *Para, int width, int height, int Interpolation_Algorithm)
{
	int i, length = width*height;
	double *Image_2 = new double[length];
	for (i = 0; i < length; i++)
		*(Image_2 + i) = (double)(*(Image + i));

	Generate_Para_Spline(Image_2, Para, width, height, Interpolation_Algorithm);

	delete[]Image_2;
	return;
}
void Get_Value_Spline(double *Para, int width, int height, double X, double Y, double *S, int S_Flag, int Interpolation_Algorithm)
{
	int i, j, width2, height2, xIndex[6], yIndex[6];
	double Para_Value, xWeight[6], yWeight[6], xWeightGradient[6], yWeightGradient[6], w, w2, w3, w4, t, t0, t1, gamma;
	double oneSix = 1.0 / 6.0;

	width2 = 2 * width - 2;
	height2 = 2 * height - 2;

	if (Interpolation_Algorithm == 6)
	{
		xIndex[0] = int(X) - 2;
		yIndex[0] = int(Y) - 2;
		for (i = 1; i < 6; i++)
		{
			xIndex[i] = xIndex[i - 1] + 1;
			yIndex[i] = yIndex[i - 1] + 1;
		}
	}
	else if ((Interpolation_Algorithm == 2) || (Interpolation_Algorithm == 5))
	{
		xIndex[0] = int(X + 0.5) - 2;
		yIndex[0] = int(Y + 0.5) - 2;
		for (i = 1; i < 5; i++)
		{
			xIndex[i] = xIndex[i - 1] + 1;
			yIndex[i] = yIndex[i - 1] + 1;
		}
	}
	else
	{
		xIndex[0] = int(X) - 1;
		yIndex[0] = int(Y) - 1;
		for (i = 1; i < 4; i++)
		{
			xIndex[i] = xIndex[i - 1] + 1;
			yIndex[i] = yIndex[i - 1] + 1;
		}
	}

	//Calculate the weights of x,y and their derivatives
	if (Interpolation_Algorithm == 1)
	{
		w = X - (double)xIndex[1];
		w2 = w*w; w3 = w2*w;
		xWeight[3] = oneSix * w3;
		xWeight[0] = oneSix + 0.5 * (w2 - w) - xWeight[3];
		xWeight[2] = w + xWeight[0] - 2.0 * xWeight[3];
		xWeight[1] = 1.0 - xWeight[0] - xWeight[2] - xWeight[3];

		if (S_Flag > -1)
		{
			xWeightGradient[3] = w2 / 2.0;
			xWeightGradient[0] = w - 0.5 - xWeightGradient[3];
			xWeightGradient[2] = 1.0 + xWeightGradient[0] - 2.0 * xWeightGradient[3];
			xWeightGradient[1] = -xWeightGradient[0] - xWeightGradient[2] - xWeightGradient[3];
		}

		/* y */
		w = Y - (double)yIndex[1];
		w2 = w*w; w3 = w2*w;
		yWeight[3] = oneSix * w3;
		yWeight[0] = oneSix + 0.5 * (w2 - w) - yWeight[3];
		yWeight[2] = w + yWeight[0] - 2.0 * yWeight[3];
		yWeight[1] = 1.0 - yWeight[0] - yWeight[2] - yWeight[3];

		if (S_Flag > -1)
		{
			yWeightGradient[3] = w2 / 2.0;
			yWeightGradient[0] = w - 0.5 - yWeightGradient[3];
			yWeightGradient[2] = 1.0 + yWeightGradient[0] - 2.0 * yWeightGradient[3];
			yWeightGradient[1] = -yWeightGradient[0] - yWeightGradient[2] - yWeightGradient[3];
		}
	}
	else if (Interpolation_Algorithm == 2)
	{
		w = X - (double)xIndex[2];
		w2 = w * w;
		t = (1.0 / 6.0) * w2;
		xWeight[0] = 1.0 / 2.0 - w;
		xWeight[0] *= xWeight[0];
		xWeight[0] *= (1.0 / 24.0) * xWeight[0];
		t0 = w * (t - 11.0 / 24.0);
		t1 = 19.0 / 96.0 + w2 * (1.0 / 4.0 - t);
		xWeight[1] = t1 + t0;
		xWeight[3] = t1 - t0;
		xWeight[4] = xWeight[0] + t0 + (1.0 / 2.0) * w;
		xWeight[2] = 1.0 - xWeight[0] - xWeight[1] - xWeight[3] - xWeight[4];

		xWeightGradient[0] = -(1.0 / 2.0 - w) * (1.0 / 2.0 - w) * (1.0 / 2.0 - w) / 6.0;
		xWeightGradient[1] = w * w / 2 - 11.0 / 24.0 + w / 2.0 - 2.0 * w * w * w / 3.0;
		xWeightGradient[3] = -w * w / 2 + 11.0 / 24.0 + w / 2.0 - 2.0 * w * w * w / 3.0;
		xWeightGradient[4] = xWeightGradient[0] + w * w / 2.0 + 1.0 / 24.0;
		xWeightGradient[2] = -xWeightGradient[0] - xWeightGradient[1] - xWeightGradient[3] - xWeightGradient[4];

		/* y */
		w = Y - (double)yIndex[2];
		w2 = w * w;
		t = (1.0 / 6.0) * w2;
		yWeight[0] = 1.0 / 2.0 - w;
		yWeight[0] *= yWeight[0];
		yWeight[0] *= (1.0 / 24.0) * yWeight[0];
		t0 = w * (t - 11.0 / 24.0);
		t1 = 19.0 / 96.0 + w2 * (1.0 / 4.0 - t);
		yWeight[1] = t1 + t0;
		yWeight[3] = t1 - t0;
		yWeight[4] = yWeight[0] + t0 + (1.0 / 2.0) * w;
		yWeight[2] = 1.0 - yWeight[0] - yWeight[1] - yWeight[3] - yWeight[4];

		yWeightGradient[0] = -(1.0 / 2.0 - w) * (1.0 / 2.0 - w) * (1.0 / 2.0 - w) / 6.0;
		yWeightGradient[1] = w * w / 2 - 11.0 / 24.0 + w / 2.0 - 2.0 * w * w * w / 3.0;
		yWeightGradient[3] = -w * w / 2 + 11.0 / 24.0 + w / 2.0 - 2.0 * w * w * w / 3.0;
		yWeightGradient[4] = yWeightGradient[0] + w * w / 2.0 + 1.0 / 24.0;
		yWeightGradient[2] = -yWeightGradient[0] - yWeightGradient[1] - yWeightGradient[3] - yWeightGradient[4];
	}
	else if (Interpolation_Algorithm == 3)
	{
		gamma = 0.0409;
		w = X - (double)xIndex[1];
		xWeight[0] = -w * w * w / 6.0 + w * w / 2.0 - (gamma + 0.5) * w + 1.0 / 6.0 + gamma;
		xWeight[1] = w * w * w / 2.0 - w * w + 3 * gamma * w + 2.0 / 3.0 - 2.0 * gamma;
		xWeight[2] = -w * w * w / 2.0 + w * w / 2.0 + (1.0 / 2.0 - 3.0 * gamma) * w + gamma + 1.0 / 6.0;
		xWeight[3] = w * w * w / 6.0 + gamma * w;

		xWeightGradient[0] = -w * w / 2.0 + w - gamma - 0.5;
		xWeightGradient[1] = 3.0 * w * w / 2.0 - 2.0 * w + 3.0 * gamma;
		xWeightGradient[2] = -3.0 * w * w / 2.0 + w + 1.0 / 2.0 - 3.0 * gamma;
		xWeightGradient[3] = w * w / 2.0 + gamma;

		/* y */
		w = Y - (double)yIndex[1];
		yWeight[0] = -w * w * w / 6.0 + w * w / 2.0 - (gamma + 0.5) * w + 1.0 / 6.0 + gamma;
		yWeight[1] = w * w * w / 2.0 - w * w + 3 * gamma * w + 2.0 / 3.0 - 2.0 * gamma;
		yWeight[2] = -w * w * w / 2.0 + w * w / 2.0 + (1.0 / 2.0 - 3.0 * gamma) * w + gamma + 1.0 / 6.0;
		yWeight[3] = w * w * w / 6.0 + gamma * w;

		yWeightGradient[0] = -w * w / 2.0 + w - gamma - 0.5;
		yWeightGradient[1] = 3.0 * w * w / 2.0 - 2.0 * w + 3.0 * gamma;
		yWeightGradient[2] = -3.0 * w * w / 2.0 + w + 1.0 / 2.0 - 3.0 * gamma;
		yWeightGradient[3] = w * w / 2.0 + gamma;
	}
	else if (Interpolation_Algorithm == 4)
	{
		w = X - (double)xIndex[1];
		xWeight[0] = -w * w * w / 6.0 + w * w / 2.0 - 11.0 * w / 21.0 + 4.0 / 21.0;
		xWeight[1] = w * w * w / 2.0 - w * w + 3.0 * w / 42.0 + 13.0 / 21.0;
		xWeight[2] = -w * w * w / 2.0 + w * w / 2.0 + 3.0 * w / 7.0 + 4.0 / 21.0;
		xWeight[3] = w * w * w / 6.0 + w / 42.0;

		xWeightGradient[0] = -w * w / 2.0 + w - 11.0 / 21.0;
		xWeightGradient[1] = 3.0 * w * w / 2.0 - 2.0 * w + 3.0 / 42.0;
		xWeightGradient[2] = -3.0 * w * w / 2.0 + w + 3.0 / 7.0;
		xWeightGradient[3] = w * w / 2.0 + 1.0 / 42.0;

		/* y */
		w = Y - (double)yIndex[1];
		yWeight[0] = -w * w * w / 6.0 + w * w / 2.0 - 11.0 * w / 21.0 + 4.0 / 21.0;
		yWeight[1] = w * w * w / 2.0 - w * w + 3.0 * w / 42.0 + 13.0 / 21.0;
		yWeight[2] = -w * w * w / 2.0 + w * w / 2.0 + 3.0 * w / 7.0 + 4.0 / 21.0;
		yWeight[3] = w * w * w / 6.0 + w / 42.0;

		yWeightGradient[0] = -w * w / 2.0 + w - 11.0 / 21.0;
		yWeightGradient[1] = 3.0 * w * w / 2.0 - 2.0 * w + 3.0 / 42.0;
		yWeightGradient[2] = -3.0 * w * w / 2.0 + w + 3.0 / 7.0;
		yWeightGradient[3] = w * w / 2.0 + 1.0 / 42.0;
	}
	else if (Interpolation_Algorithm == 5)
	{
		w = X - (double)xIndex[2];
		w2 = w*w; w3 = w2*w; w4 = w2*w2;
		double coeff1 = 743.0 / 120960.0, coeff2 = 6397.0 / 30240.0, coeff3 = 5.0 / 144.0, coeff4 = 31.0 / 72.0, coeff5 = 11383.0 / 20160.0;
		double coeff6 = 11.0 / 144.0, coeff7 = 5.0 / 144.0, coeff8 = 7.0 / 36.0, coeff9 = 31.0 / 72.0, coeff10 = 13.0 / 24.0, coeff11 = 11.0 / 72.0, coeff12 = 7.0 / 18.0;
		xWeight[0] = w4 / 24.0 - w3 / 12.0 + w2 * coeff6 - w *coeff7 + coeff1;
		xWeight[1] = -w4 / 6.0 + w3 / 6.0 + w2*coeff8 - w*coeff9 + coeff2;
		xWeight[2] = w4 / 4.0 - w2 *coeff10 + coeff5;
		xWeight[3] = -w4 / 6.0 - w3 / 6.0 + w2*coeff8 + w*coeff9 + coeff2;
		xWeight[4] = w4 / 24.0 + w3 / 12.0 + w2 * coeff6 + w *coeff7 + coeff1;
		//xWeight[2] = 1.0 - xWeight[0] - xWeight[1] - xWeight[3] - xWeight[4]; 

		if (S_Flag > -1)
		{
			xWeightGradient[0] = w3 / 6.0 - w2 / 4.0 + w *coeff11 - coeff3;
			xWeightGradient[1] = -2.0 * w3 / 3.0 + w2 / 2.0 + w*coeff12 - coeff4;
			xWeightGradient[2] = w3 - 13.0 * w / 12.0;
			xWeightGradient[3] = -2.0 * w3 / 3.0 - w2 / 2.0 + w*coeff12 + coeff4;
			xWeightGradient[4] = w3 / 6.0 + w2 / 4.0 + w *coeff11 + coeff3;
		}

		/* y */
		w = Y - (double)yIndex[2];
		w2 = w*w; w3 = w2*w; w4 = w2*w2;
		yWeight[0] = w4 / 24.0 - w3 / 12.0 + w2 * coeff6 - w *coeff7 + coeff1;
		yWeight[1] = -w4 / 6.0 + w3 / 6.0 + w2*coeff8 - w*coeff9 + coeff2;
		yWeight[2] = w4 / 4.0 - w2 *coeff10 + coeff5;
		yWeight[3] = -w4 / 6.0 - w3 / 6.0 + w2*coeff8 + w*coeff9 + coeff2;
		yWeight[4] = w4 / 24.0 + w3 / 12.0 + w2 * coeff6 + w *coeff7 + coeff1;
		//yWeight[2] = 1.0 - xWeight[0] - xWeight[1] - xWeight[3] - xWeight[4]; 

		if (S_Flag > -1)
		{
			yWeightGradient[0] = w3 / 6.0 - w2 / 4.0 + w *coeff11 - coeff3;
			yWeightGradient[1] = -2.0 * w3 / 3.0 + w2 / 2.0 + w*coeff12 - coeff4;
			yWeightGradient[2] = w3 - 13.0 * w / 12.0;
			yWeightGradient[3] = -2.0 * w3 / 3.0 - w2 / 2.0 + w*coeff12 + coeff4;
			yWeightGradient[4] = w3 / 6.0 + w2 / 4.0 + w *coeff11 + coeff3;
		}
	}
	else if (Interpolation_Algorithm == 6)
	{
		w = X - (double)xIndex[2];
		w2 = w * w;
		xWeight[5] = (1.0 / 120.0) * w * w2 * w2;
		w2 -= w;
		w4 = w2 * w2;
		w -= 1.0 / 2.0;
		t = w2 * (w2 - 3.0);
		xWeight[0] = (1.0 / 24.0) * (1.0 / 5.0 + w2 + w4) - xWeight[5];
		t0 = (1.0 / 24.0) * (w2 * (w2 - 5.0) + 46.0 / 5.0);
		t1 = (-1.0 / 12.0) * w * (t + 4.0);
		xWeight[2] = t0 + t1;
		xWeight[3] = t0 - t1;
		t0 = (1.0 / 16.0) * (9.0 / 5.0 - t);
		t1 = (1.0 / 24.0) * w * (w4 - w2 - 5.0);
		xWeight[1] = t0 + t1;
		xWeight[4] = t0 - t1;

		xWeightGradient[5] = w * w * w * w / 24.0;
		xWeightGradient[0] = (4 * w * w * w - 6 * w * w + 4 * w - 1) / 24.0 - xWeightGradient[5];
		t0 = (4.0 * w * w * w - 6.0 * w * w - 8.0 * w + 5.0) / 24.0;
		t1 = -(5.0 * w * w * w * w - 10.0 * w * w * w - 3.0 * w * w + 8.0 * w + 5.0 / 2.0) / 12.0;
		xWeightGradient[2] = t0 + t1;
		xWeightGradient[3] = t0 - t1;
		t0 = (-4.0 * w * w * w + 6.0 * w * w + 4.0 * w - 3) / 16.0;
		t1 = (5.0 * w * w * w * w - 10.0 * w * w * w + 3.0 * w * w + 2 * w - 11.0 / 2.0) / 24.0;
		xWeightGradient[1] = t0 + t1;
		xWeightGradient[4] = t0 - t1;

		/* y */
		w = Y - (double)yIndex[2];
		w2 = w * w;
		yWeight[5] = (1.0 / 120.0) * w * w2 * w2;
		w2 -= w;
		w4 = w2 * w2;
		w -= 1.0 / 2.0;
		t = w2 * (w2 - 3.0);
		yWeight[0] = (1.0 / 24.0) * (1.0 / 5.0 + w2 + w4) - yWeight[5];
		t0 = (1.0 / 24.0) * (w2 * (w2 - 5.0) + 46.0 / 5.0);
		t1 = (-1.0 / 12.0) * w * (t + 4.0);
		yWeight[2] = t0 + t1;
		yWeight[3] = t0 - t1;
		t0 = (1.0 / 16.0) * (9.0 / 5.0 - t);
		t1 = (1.0 / 24.0) * w * (w4 - w2 - 5.0);
		yWeight[1] = t0 + t1;
		yWeight[4] = t0 - t1;

		yWeightGradient[5] = w * w * w * w / 24.0;
		yWeightGradient[0] = (4 * w * w * w - 6 * w * w + 4 * w - 1) / 24.0 - yWeightGradient[5];
		t0 = (4.0 * w * w * w - 6.0 * w * w - 8.0 * w + 5.0) / 24.0;
		t1 = -(5.0 * w * w * w * w - 10.0 * w * w * w - 3.0 * w * w + 8.0 * w + 5.0 / 2.0) / 12.0;
		yWeightGradient[2] = t0 + t1;
		yWeightGradient[3] = t0 - t1;
		t0 = (-4.0 * w * w * w + 6.0 * w * w + 4.0 * w - 3) / 16.0;
		t1 = (5.0 * w * w * w * w - 10.0 * w * w * w + 3.0 * w * w + 2 * w - 11.0 / 2.0) / 24.0;
		yWeightGradient[1] = t0 + t1;
		yWeightGradient[4] = t0 - t1;
	}
	//***********************************

	/* apply the mirror boundary conditions and calculate the interpolated values */
	S[0] = 0;
	if (S_Flag > -1)
		S[1] = 0, S[2] = 0;

	if (Interpolation_Algorithm == 6)
	{
		for (i = 0; i < 6; i++)
		{
			xIndex[i] = (width == 1) ? (0) : ((xIndex[i] < 0) ?
				(-xIndex[i] - width2 * ((-xIndex[i]) / width2))
				: (xIndex[i] - width2 * (xIndex[i] / width2)));
			if (width <= xIndex[i]) {
				xIndex[i] = width2 - xIndex[i];
			}
			yIndex[i] = (height == 1) ? (0) : ((yIndex[i] < 0) ?
				(-yIndex[i] - height2 * ((-yIndex[i]) / height2))
				: (yIndex[i] - height2 * (yIndex[i] / height2)));
			if (height <= yIndex[i]) {
				yIndex[i] = height2 - yIndex[i];
			}
		}

		for (i = 0; i < 6; i++)
		{
			for (j = 0; j < 6; j++)
			{
				Para_Value = (*(Para + width * yIndex[i] + xIndex[j]));
				S[0] = S[0] + Para_Value * xWeight[j] * yWeight[i];
				if (S_Flag > -1)
				{
					S[1] = S[1] + Para_Value * xWeightGradient[j] * yWeight[i];
					S[2] = S[2] + Para_Value * xWeight[j] * yWeightGradient[i];
				}
			}
		}
	}
	else if ((Interpolation_Algorithm == 2) || (Interpolation_Algorithm == 5))
	{
		for (i = 0; i < 5; i++)
		{
			xIndex[i] = (width == 1) ? (0) : ((xIndex[i] < 0) ?
				(-xIndex[i] - width2 * ((-xIndex[i]) / width2))
				: (xIndex[i] - width2 * (xIndex[i] / width2)));
			if (width <= xIndex[i]) {
				xIndex[i] = width2 - xIndex[i];
			}
			yIndex[i] = (height == 1) ? (0) : ((yIndex[i] < 0) ?
				(-yIndex[i] - height2 * ((-yIndex[i]) / height2))
				: (yIndex[i] - height2 * (yIndex[i] / height2)));
			if (height <= yIndex[i]) {
				yIndex[i] = height2 - yIndex[i];
			}
		}
		for (i = 0; i < 5; i++)
		{
			for (j = 0; j < 5; j++)
			{
				Para_Value = (*(Para + width * yIndex[i] + xIndex[j]));
				S[0] = S[0] + Para_Value * xWeight[j] * yWeight[i];
				if (S_Flag > -1)
				{
					S[1] = S[1] + Para_Value * xWeightGradient[j] * yWeight[i];
					S[2] = S[2] + Para_Value * xWeight[j] * yWeightGradient[i];
				}
			}
		}
	}
	else
	{
		for (i = 0; i < 4; i++)
		{
			xIndex[i] = (width == 1) ? (0) : ((xIndex[i] < 0) ?
				(-xIndex[i] - width2 * ((-xIndex[i]) / width2))
				: (xIndex[i] - width2 * (xIndex[i] / width2)));
			if (width <= xIndex[i]) {
				xIndex[i] = width2 - xIndex[i];
			}
			yIndex[i] = (height == 1) ? (0) : ((yIndex[i] < 0) ?
				(-yIndex[i] - height2 * ((-yIndex[i]) / height2))
				: (yIndex[i] - height2 * (yIndex[i] / height2)));
			if (height <= yIndex[i]) {
				yIndex[i] = height2 - yIndex[i];
			}
		}
		for (i = 0; i < 4; i++)
		{
			for (j = 0; j < 4; j++)
			{
				Para_Value = (*(Para + width * yIndex[i] + xIndex[j]));
				S[0] = S[0] + Para_Value * xWeight[j] * yWeight[i];
				if (S_Flag > -1)
				{
					S[1] = S[1] + Para_Value * xWeightGradient[j] * yWeight[i];
					S[2] = S[2] + Para_Value * xWeight[j] * yWeightGradient[i];
				}
			}
		}
	}

	return;
}

//Float is to save memory, so, its input and output are a bit different
void Generate_Para_Spline(uchar *Image, float *Para, int width, int height, int Interpolation_Algorithm)
{
	double tolerance;
	int i, j, nPoles;
	int length = width * height;
	double pole[2], a, gamma;
	tolerance = 1e-4;
	if (Interpolation_Algorithm == 1) // 4-tap
	{
		nPoles = 1;
		pole[0] = sqrt(3.0) - 2.0;
	}
	else if (Interpolation_Algorithm == 2) // 6-tap
	{
		nPoles = 2;
		pole[0] = sqrt(664.0 - sqrt(438976.0)) + sqrt(304.0) - 19.0;
		pole[1] = sqrt(664.0 + sqrt(438976.0)) - sqrt(304.0) - 19.0;
	}
	else if (Interpolation_Algorithm == 3) // modified 4-tap
	{
		gamma = 0.0409;
		a = (4.0 - 12.0 * gamma) / (6.0 * gamma + 1.0);
		pole[0] = (-a + sqrt(a * a - 4)) / 2.0;
	}
	else if (Interpolation_Algorithm == 4) // optimized 4-tap
	{
		nPoles = 1;
		pole[0] = (-13.0 + sqrt(105.0)) / 8.0;
	}
	else if (Interpolation_Algorithm == 5) // optimized 6-tap
	{
		nPoles = 2;
		pole[0] = -0.410549185795627524168;
		pole[1] = -0.0316849091024414351363;
	}
	else if (Interpolation_Algorithm == 6) // 8-tap
	{
		nPoles = 2;
		pole[0] = sqrt(135.0 / 2.0 - sqrt(17745.0 / 4.0)) + sqrt(105.0 / 4.0) - 13.0 / 2.0;
		pole[1] = sqrt(135.0 / 2.0 + sqrt(17745.0 / 4.0)) - sqrt(105.0 / 4.0) - 13.0 / 2.0;
	}

	//Perform the 1D prefiltering along the rows
	double *LineWidth = new double[width];
	for (i = 0; i < height; i++)
	{
		//Prefiltering each row
		for (j = 0; j < width; j++)
		{
			*(LineWidth + j) = (float)(int)Image[i * width + j];
		}
		if (Interpolation_Algorithm == 3)
			Prefilter_1Dm(LineWidth, width, pole, tolerance, gamma);
		else
			Prefilter_1D(LineWidth, width, pole, tolerance, nPoles);

		// Put the prefiltered coeffiecients into Para array
		for (j = 0; j < width; j++)
			Para[i * width + j] = (float)LineWidth[j];
	}
	delete[]LineWidth;

	//Perform the 1D prefiltering along the columns
	double *LineHeight = new double[height];
	for (i = 0; i < width; i++)
	{
		//Prefiltering each comlumn
		for (j = 0; j < height; j++)
		{
			*(LineHeight + j) = (*(Para + j * width + i));
		}
		if (Interpolation_Algorithm == 3)
			Prefilter_1Dm(LineHeight, height, pole, tolerance, gamma);
		else
			Prefilter_1D(LineHeight, height, pole, tolerance, nPoles);

		//Put the prefilterd coefficients into the Para array
		for (j = 0; j < height; j++)
			Para[j * width + i] = (float)LineHeight[j];
	}
	delete[]LineHeight;

	return;
}
void Generate_Para_Spline(char *Image, float *Para, int width, int height, int Interpolation_Algorithm)
{
	double tolerance;
	int i, j, nPoles;
	int length = width * height;
	double pole[2], a, gamma;
	tolerance = 1e-4;
	if (Interpolation_Algorithm == 1) // 4-tap
	{
		nPoles = 1;
		pole[0] = sqrt(3.0) - 2.0;
	}
	else if (Interpolation_Algorithm == 2) // 6-tap
	{
		nPoles = 2;
		pole[0] = sqrt(664.0 - sqrt(438976.0)) + sqrt(304.0) - 19.0;
		pole[1] = sqrt(664.0 + sqrt(438976.0)) - sqrt(304.0) - 19.0;
	}
	else if (Interpolation_Algorithm == 3) // modified 4-tap
	{
		gamma = 0.0409;
		a = (4.0 - 12.0 * gamma) / (6.0 * gamma + 1.0);
		pole[0] = (-a + sqrt(a * a - 4)) / 2.0;
	}
	else if (Interpolation_Algorithm == 4) // optimized 4-tap
	{
		nPoles = 1;
		pole[0] = (-13.0 + sqrt(105.0)) / 8.0;
	}
	else if (Interpolation_Algorithm == 5) // optimized 6-tap
	{
		nPoles = 2;
		pole[0] = -0.410549185795627524168;
		pole[1] = -0.0316849091024414351363;
	}
	else if (Interpolation_Algorithm == 6) // 8-tap
	{
		nPoles = 2;
		pole[0] = sqrt(135.0 / 2.0 - sqrt(17745.0 / 4.0)) + sqrt(105.0 / 4.0) - 13.0 / 2.0;
		pole[1] = sqrt(135.0 / 2.0 + sqrt(17745.0 / 4.0)) - sqrt(105.0 / 4.0) - 13.0 / 2.0;
	}

	//Perform the 1D prefiltering along the rows
	double *LineWidth = new double[width];
	for (i = 0; i < height; i++)
	{
		//Prefiltering each row
		for (j = 0; j < width; j++)
		{
			*(LineWidth + j) = (float)(int)(unsigned char)Image[i * width + j];
		}
		if (Interpolation_Algorithm == 3)
			Prefilter_1Dm(LineWidth, width, pole, tolerance, gamma);
		else
			Prefilter_1D(LineWidth, width, pole, tolerance, nPoles);

		// Put the prefiltered coeffiecients into Para array
		for (j = 0; j < width; j++)
			Para[i * width + j] = (float)LineWidth[j];
	}
	delete[]LineWidth;

	//Perform the 1D prefiltering along the columns
	double *LineHeight = new double[height];
	for (i = 0; i < width; i++)
	{
		//Prefiltering each comlumn
		for (j = 0; j < height; j++)
		{
			*(LineHeight + j) = (*(Para + j * width + i));
		}
		if (Interpolation_Algorithm == 3)
			Prefilter_1Dm(LineHeight, height, pole, tolerance, gamma);
		else
			Prefilter_1D(LineHeight, height, pole, tolerance, nPoles);

		//Put the prefilterd coefficients into the Para array
		for (j = 0; j < height; j++)
			Para[j * width + i] = (float)LineHeight[j];
	}
	delete[]LineHeight;

	return;
}
void Generate_Para_Spline(float *Image, float *Para, int width, int height, int Interpolation_Algorithm)
{
	double tolerance;
	int i, j, nPoles;
	int length = width * height;
	double pole[2], a, gamma;
	tolerance = 1e-4;
	if (Interpolation_Algorithm == 1) // 4-tap
	{
		nPoles = 1;
		pole[0] = sqrt(3.0) - 2.0;
	}
	else if (Interpolation_Algorithm == 2) // 6-tap
	{
		nPoles = 2;
		pole[0] = sqrt(664.0 - sqrt(438976.0)) + sqrt(304.0) - 19.0;
		pole[1] = sqrt(664.0 + sqrt(438976.0)) - sqrt(304.0) - 19.0;
	}
	else if (Interpolation_Algorithm == 3) // modified 4-tap
	{
		gamma = 0.0409;
		a = (4.0 - 12.0 * gamma) / (6.0 * gamma + 1.0);
		pole[0] = (-a + sqrt(a * a - 4)) / 2.0;
	}
	else if (Interpolation_Algorithm == 4) // optimized 4-tap
	{
		nPoles = 1;
		pole[0] = (-13.0 + sqrt(105.0)) / 8.0;
	}
	else if (Interpolation_Algorithm == 5) // optimized 6-tap
	{
		nPoles = 2;
		pole[0] = -0.410549185795627524168;
		pole[1] = -0.0316849091024414351363;
	}
	else if (Interpolation_Algorithm == 6) // 8-tap
	{
		nPoles = 2;
		pole[0] = sqrt(135.0 / 2.0 - sqrt(17745.0 / 4.0)) + sqrt(105.0 / 4.0) - 13.0 / 2.0;
		pole[1] = sqrt(135.0 / 2.0 + sqrt(17745.0 / 4.0)) - sqrt(105.0 / 4.0) - 13.0 / 2.0;
	}

	//Perform the 1D prefiltering along the rows
	double *LineWidth = new double[width];
	for (i = 0; i < height; i++)
	{
		//Prefiltering each row
		for (j = 0; j < width; j++)
		{
			*(LineWidth + j) = *(Image + i * width + j);
		}
		if (Interpolation_Algorithm == 3)
			Prefilter_1Dm(LineWidth, width, pole, tolerance, gamma);
		else
			Prefilter_1D(LineWidth, width, pole, tolerance, nPoles);

		// Put the prefiltered coeffiecients into Para array
		for (j = 0; j < width; j++)
			Para[i * width + j] = (float)LineWidth[j];
	}
	delete[]LineWidth;

	//Perform the 1D prefiltering along the columns
	double *LineHeight = new double[height];
	for (i = 0; i < width; i++)
	{
		//Prefiltering each comlumn
		for (j = 0; j < height; j++)
		{
			*(LineHeight + j) = (*(Para + j * width + i));
		}
		if (Interpolation_Algorithm == 3)
			Prefilter_1Dm(LineHeight, height, pole, tolerance, gamma);
		else
			Prefilter_1D(LineHeight, height, pole, tolerance, nPoles);

		//Put the prefilterd coefficients into the Para array
		for (j = 0; j < height; j++)
			Para[j * width + i] = (float)LineHeight[j];
	}
	delete[]LineHeight;

	return;
}
void Get_Value_Spline(float *Para, int width, int height, double X, double Y, double *S, int S_Flag, int Interpolation_Algorithm)
{
	int i, j, width2, height2, xIndex[6], yIndex[6];
	double Para_Value, xWeight[6], yWeight[6], xWeightGradient[6], yWeightGradient[6], w, w2, w3, w4, t, t0, t1, gamma;
	double oneSix = 1.0 / 6.0;

	width2 = 2 * width - 2;
	height2 = 2 * height - 2;

	if (Interpolation_Algorithm == 6)
	{
		xIndex[0] = int(X) - 2;
		yIndex[0] = int(Y) - 2;
		for (i = 1; i < 6; i++)
		{
			xIndex[i] = xIndex[i - 1] + 1;
			yIndex[i] = yIndex[i - 1] + 1;
		}
	}
	else if ((Interpolation_Algorithm == 2) || (Interpolation_Algorithm == 5))
	{
		xIndex[0] = int(X + 0.5) - 2;
		yIndex[0] = int(Y + 0.5) - 2;
		for (i = 1; i < 5; i++)
		{
			xIndex[i] = xIndex[i - 1] + 1;
			yIndex[i] = yIndex[i - 1] + 1;
		}
	}
	else
	{
		xIndex[0] = int(X) - 1;
		yIndex[0] = int(Y) - 1;
		for (i = 1; i < 4; i++)
		{
			xIndex[i] = xIndex[i - 1] + 1;
			yIndex[i] = yIndex[i - 1] + 1;
		}
	}

	//Calculate the weights of x,y and their derivatives
	if (Interpolation_Algorithm == 1)
	{
		w = X - (double)xIndex[1];
		w2 = w*w; w3 = w2*w;
		xWeight[3] = oneSix * w3;
		xWeight[0] = oneSix + 0.5 * (w2 - w) - xWeight[3];
		xWeight[2] = w + xWeight[0] - 2.0 * xWeight[3];
		xWeight[1] = 1.0 - xWeight[0] - xWeight[2] - xWeight[3];

		xWeightGradient[3] = w2 / 2.0;
		xWeightGradient[0] = w - 0.5 - xWeightGradient[3];
		xWeightGradient[2] = 1.0 + xWeightGradient[0] - 2.0 * xWeightGradient[3];
		xWeightGradient[1] = -xWeightGradient[0] - xWeightGradient[2] - xWeightGradient[3];

		/* y */
		w = Y - (double)yIndex[1];
		w2 = w*w; w3 = w2*w;
		yWeight[3] = oneSix * w3;
		yWeight[0] = oneSix + 0.5 * (w2 - w) - yWeight[3];
		yWeight[2] = w + yWeight[0] - 2.0 * yWeight[3];
		yWeight[1] = 1.0 - yWeight[0] - yWeight[2] - yWeight[3];

		yWeightGradient[3] = w2 / 2.0;
		yWeightGradient[0] = w - 0.5 - yWeightGradient[3];
		yWeightGradient[2] = 1.0 + yWeightGradient[0] - 2.0 * yWeightGradient[3];
		yWeightGradient[1] = -yWeightGradient[0] - yWeightGradient[2] - yWeightGradient[3];
	}
	else if (Interpolation_Algorithm == 2)
	{
		w = X - (double)xIndex[2];
		w2 = w * w;
		t = (1.0 / 6.0) * w2;
		xWeight[0] = 1.0 / 2.0 - w;
		xWeight[0] *= xWeight[0];
		xWeight[0] *= (1.0 / 24.0) * xWeight[0];
		t0 = w * (t - 11.0 / 24.0);
		t1 = 19.0 / 96.0 + w2 * (1.0 / 4.0 - t);
		xWeight[1] = t1 + t0;
		xWeight[3] = t1 - t0;
		xWeight[4] = xWeight[0] + t0 + (1.0 / 2.0) * w;
		xWeight[2] = 1.0 - xWeight[0] - xWeight[1] - xWeight[3] - xWeight[4];

		xWeightGradient[0] = -(1.0 / 2.0 - w) * (1.0 / 2.0 - w) * (1.0 / 2.0 - w) / 6.0;
		xWeightGradient[1] = w * w / 2 - 11.0 / 24.0 + w / 2.0 - 2.0 * w * w * w / 3.0;
		xWeightGradient[3] = -w * w / 2 + 11.0 / 24.0 + w / 2.0 - 2.0 * w * w * w / 3.0;
		xWeightGradient[4] = xWeightGradient[0] + w * w / 2.0 + 1.0 / 24.0;
		xWeightGradient[2] = -xWeightGradient[0] - xWeightGradient[1] - xWeightGradient[3] - xWeightGradient[4];

		/* y */
		w = Y - (double)yIndex[2];
		w2 = w * w;
		t = (1.0 / 6.0) * w2;
		yWeight[0] = 1.0 / 2.0 - w;
		yWeight[0] *= yWeight[0];
		yWeight[0] *= (1.0 / 24.0) * yWeight[0];
		t0 = w * (t - 11.0 / 24.0);
		t1 = 19.0 / 96.0 + w2 * (1.0 / 4.0 - t);
		yWeight[1] = t1 + t0;
		yWeight[3] = t1 - t0;
		yWeight[4] = yWeight[0] + t0 + (1.0 / 2.0) * w;
		yWeight[2] = 1.0 - yWeight[0] - yWeight[1] - yWeight[3] - yWeight[4];

		yWeightGradient[0] = -(1.0 / 2.0 - w) * (1.0 / 2.0 - w) * (1.0 / 2.0 - w) / 6.0;
		yWeightGradient[1] = w * w / 2 - 11.0 / 24.0 + w / 2.0 - 2.0 * w * w * w / 3.0;
		yWeightGradient[3] = -w * w / 2 + 11.0 / 24.0 + w / 2.0 - 2.0 * w * w * w / 3.0;
		yWeightGradient[4] = yWeightGradient[0] + w * w / 2.0 + 1.0 / 24.0;
		yWeightGradient[2] = -yWeightGradient[0] - yWeightGradient[1] - yWeightGradient[3] - yWeightGradient[4];
	}
	else if (Interpolation_Algorithm == 3)
	{
		gamma = 0.0409;
		w = X - (double)xIndex[1];
		xWeight[0] = -w * w * w / 6.0 + w * w / 2.0 - (gamma + 0.5) * w + 1.0 / 6.0 + gamma;
		xWeight[1] = w * w * w / 2.0 - w * w + 3 * gamma * w + 2.0 / 3.0 - 2.0 * gamma;
		xWeight[2] = -w * w * w / 2.0 + w * w / 2.0 + (1.0 / 2.0 - 3.0 * gamma) * w + gamma + 1.0 / 6.0;
		xWeight[3] = w * w * w / 6.0 + gamma * w;

		xWeightGradient[0] = -w * w / 2.0 + w - gamma - 0.5;
		xWeightGradient[1] = 3.0 * w * w / 2.0 - 2.0 * w + 3.0 * gamma;
		xWeightGradient[2] = -3.0 * w * w / 2.0 + w + 1.0 / 2.0 - 3.0 * gamma;
		xWeightGradient[3] = w * w / 2.0 + gamma;

		/* y */
		w = Y - (double)yIndex[1];
		yWeight[0] = -w * w * w / 6.0 + w * w / 2.0 - (gamma + 0.5) * w + 1.0 / 6.0 + gamma;
		yWeight[1] = w * w * w / 2.0 - w * w + 3 * gamma * w + 2.0 / 3.0 - 2.0 * gamma;
		yWeight[2] = -w * w * w / 2.0 + w * w / 2.0 + (1.0 / 2.0 - 3.0 * gamma) * w + gamma + 1.0 / 6.0;
		yWeight[3] = w * w * w / 6.0 + gamma * w;

		yWeightGradient[0] = -w * w / 2.0 + w - gamma - 0.5;
		yWeightGradient[1] = 3.0 * w * w / 2.0 - 2.0 * w + 3.0 * gamma;
		yWeightGradient[2] = -3.0 * w * w / 2.0 + w + 1.0 / 2.0 - 3.0 * gamma;
		yWeightGradient[3] = w * w / 2.0 + gamma;
	}
	else if (Interpolation_Algorithm == 4)
	{
		w = X - (double)xIndex[1];
		xWeight[0] = -w * w * w / 6.0 + w * w / 2.0 - 11.0 * w / 21.0 + 4.0 / 21.0;
		xWeight[1] = w * w * w / 2.0 - w * w + 3.0 * w / 42.0 + 13.0 / 21.0;
		xWeight[2] = -w * w * w / 2.0 + w * w / 2.0 + 3.0 * w / 7.0 + 4.0 / 21.0;
		xWeight[3] = w * w * w / 6.0 + w / 42.0;

		xWeightGradient[0] = -w * w / 2.0 + w - 11.0 / 21.0;
		xWeightGradient[1] = 3.0 * w * w / 2.0 - 2.0 * w + 3.0 / 42.0;
		xWeightGradient[2] = -3.0 * w * w / 2.0 + w + 3.0 / 7.0;
		xWeightGradient[3] = w * w / 2.0 + 1.0 / 42.0;

		/* y */
		w = Y - (double)yIndex[1];
		yWeight[0] = -w * w * w / 6.0 + w * w / 2.0 - 11.0 * w / 21.0 + 4.0 / 21.0;
		yWeight[1] = w * w * w / 2.0 - w * w + 3.0 * w / 42.0 + 13.0 / 21.0;
		yWeight[2] = -w * w * w / 2.0 + w * w / 2.0 + 3.0 * w / 7.0 + 4.0 / 21.0;
		yWeight[3] = w * w * w / 6.0 + w / 42.0;

		yWeightGradient[0] = -w * w / 2.0 + w - 11.0 / 21.0;
		yWeightGradient[1] = 3.0 * w * w / 2.0 - 2.0 * w + 3.0 / 42.0;
		yWeightGradient[2] = -3.0 * w * w / 2.0 + w + 3.0 / 7.0;
		yWeightGradient[3] = w * w / 2.0 + 1.0 / 42.0;
	}
	else if (Interpolation_Algorithm == 5)
	{
		w = X - (double)xIndex[2];
		w2 = w*w; w3 = w2*w; w4 = w2*w2;
		double coeff1 = 743.0 / 120960.0, coeff2 = 6397.0 / 30240.0, coeff3 = 5.0 / 144.0, coeff4 = 31.0 / 72.0, coeff5 = 11383.0 / 20160.0;
		double coeff6 = 11.0 / 144.0, coeff7 = 5.0 / 144.0, coeff8 = 7.0 / 36.0, coeff9 = 31.0 / 72.0, coeff10 = 13.0 / 24.0, coeff11 = 11.0 / 72.0, coeff12 = 7.0 / 18.0;
		xWeight[0] = w4 / 24.0 - w3 / 12.0 + w2 * coeff6 - w *coeff7 + coeff1;
		xWeight[1] = -w4 / 6.0 + w3 / 6.0 + w2*coeff8 - w*coeff9 + coeff2;
		xWeight[2] = w4 / 4.0 - w2 *coeff10 + coeff5;
		xWeight[3] = -w4 / 6.0 - w3 / 6.0 + w2*coeff8 + w*coeff9 + coeff2;
		xWeight[4] = w4 / 24.0 + w3 / 12.0 + w2 * coeff6 + w *coeff7 + coeff1;
		//xWeight[2] = 1.0 - xWeight[0] - xWeight[1] - xWeight[3] - xWeight[4]; 

		xWeightGradient[0] = w3 / 6.0 - w2 / 4.0 + w *coeff11 - coeff3;
		xWeightGradient[1] = -2.0 * w3 / 3.0 + w2 / 2.0 + w*coeff12 - coeff4;
		xWeightGradient[2] = w3 - 13.0 * w / 12.0;
		xWeightGradient[3] = -2.0 * w3 / 3.0 - w2 / 2.0 + w*coeff12 + coeff4;
		xWeightGradient[4] = w3 / 6.0 + w2 / 4.0 + w *coeff11 + coeff3;

		/* y */
		w = Y - (double)yIndex[2];
		w2 = w*w; w3 = w2*w; w4 = w2*w2;
		yWeight[0] = w4 / 24.0 - w3 / 12.0 + w2 * coeff6 - w *coeff7 + coeff1;
		yWeight[1] = -w4 / 6.0 + w3 / 6.0 + w2*coeff8 - w*coeff9 + coeff2;
		yWeight[2] = w4 / 4.0 - w2 *coeff10 + coeff5;
		yWeight[3] = -w4 / 6.0 - w3 / 6.0 + w2*coeff8 + w*coeff9 + coeff2;
		yWeight[4] = w4 / 24.0 + w3 / 12.0 + w2 * coeff6 + w *coeff7 + coeff1;
		//yWeight[2] = 1.0 - xWeight[0] - xWeight[1] - xWeight[3] - xWeight[4]; 

		yWeightGradient[0] = w3 / 6.0 - w2 / 4.0 + w *coeff11 - coeff3;
		yWeightGradient[1] = -2.0 * w3 / 3.0 + w2 / 2.0 + w*coeff12 - coeff4;
		yWeightGradient[2] = w3 - 13.0 * w / 12.0;
		yWeightGradient[3] = -2.0 * w3 / 3.0 - w2 / 2.0 + w*coeff12 + coeff4;
		yWeightGradient[4] = w3 / 6.0 + w2 / 4.0 + w *coeff11 + coeff3;
	}
	else if (Interpolation_Algorithm == 6)
	{
		w = X - (double)xIndex[2];
		w2 = w * w;
		xWeight[5] = (1.0 / 120.0) * w * w2 * w2;
		w2 -= w;
		w4 = w2 * w2;
		w -= 1.0 / 2.0;
		t = w2 * (w2 - 3.0);
		xWeight[0] = (1.0 / 24.0) * (1.0 / 5.0 + w2 + w4) - xWeight[5];
		t0 = (1.0 / 24.0) * (w2 * (w2 - 5.0) + 46.0 / 5.0);
		t1 = (-1.0 / 12.0) * w * (t + 4.0);
		xWeight[2] = t0 + t1;
		xWeight[3] = t0 - t1;
		t0 = (1.0 / 16.0) * (9.0 / 5.0 - t);
		t1 = (1.0 / 24.0) * w * (w4 - w2 - 5.0);
		xWeight[1] = t0 + t1;
		xWeight[4] = t0 - t1;

		xWeightGradient[5] = w * w * w * w / 24.0;
		xWeightGradient[0] = (4 * w * w * w - 6 * w * w + 4 * w - 1) / 24.0 - xWeightGradient[5];
		t0 = (4.0 * w * w * w - 6.0 * w * w - 8.0 * w + 5.0) / 24.0;
		t1 = -(5.0 * w * w * w * w - 10.0 * w * w * w - 3.0 * w * w + 8.0 * w + 5.0 / 2.0) / 12.0;
		xWeightGradient[2] = t0 + t1;
		xWeightGradient[3] = t0 - t1;
		t0 = (-4.0 * w * w * w + 6.0 * w * w + 4.0 * w - 3) / 16.0;
		t1 = (5.0 * w * w * w * w - 10.0 * w * w * w + 3.0 * w * w + 2 * w - 11.0 / 2.0) / 24.0;
		xWeightGradient[1] = t0 + t1;
		xWeightGradient[4] = t0 - t1;

		/* y */
		w = Y - (double)yIndex[2];
		w2 = w * w;
		yWeight[5] = (1.0 / 120.0) * w * w2 * w2;
		w2 -= w;
		w4 = w2 * w2;
		w -= 1.0 / 2.0;
		t = w2 * (w2 - 3.0);
		yWeight[0] = (1.0 / 24.0) * (1.0 / 5.0 + w2 + w4) - yWeight[5];
		t0 = (1.0 / 24.0) * (w2 * (w2 - 5.0) + 46.0 / 5.0);
		t1 = (-1.0 / 12.0) * w * (t + 4.0);
		yWeight[2] = t0 + t1;
		yWeight[3] = t0 - t1;
		t0 = (1.0 / 16.0) * (9.0 / 5.0 - t);
		t1 = (1.0 / 24.0) * w * (w4 - w2 - 5.0);
		yWeight[1] = t0 + t1;
		yWeight[4] = t0 - t1;

		yWeightGradient[5] = w * w * w * w / 24.0;
		yWeightGradient[0] = (4 * w * w * w - 6 * w * w + 4 * w - 1) / 24.0 - yWeightGradient[5];
		t0 = (4.0 * w * w * w - 6.0 * w * w - 8.0 * w + 5.0) / 24.0;
		t1 = -(5.0 * w * w * w * w - 10.0 * w * w * w - 3.0 * w * w + 8.0 * w + 5.0 / 2.0) / 12.0;
		yWeightGradient[2] = t0 + t1;
		yWeightGradient[3] = t0 - t1;
		t0 = (-4.0 * w * w * w + 6.0 * w * w + 4.0 * w - 3) / 16.0;
		t1 = (5.0 * w * w * w * w - 10.0 * w * w * w + 3.0 * w * w + 2 * w - 11.0 / 2.0) / 24.0;
		yWeightGradient[1] = t0 + t1;
		yWeightGradient[4] = t0 - t1;
	}
	//***********************************

	/* apply the mirror boundary conditions and calculate the interpolated values */
	S[0] = 0;
	S[1] = 0;
	S[2] = 0;

	if (Interpolation_Algorithm == 6)
	{
		for (i = 0; i < 6; i++)
		{
			xIndex[i] = (width == 1) ? (0) : ((xIndex[i] < 0) ?
				(-xIndex[i] - width2 * ((-xIndex[i]) / width2))
				: (xIndex[i] - width2 * (xIndex[i] / width2)));
			if (width <= xIndex[i]) {
				xIndex[i] = width2 - xIndex[i];
			}
			yIndex[i] = (height == 1) ? (0) : ((yIndex[i] < 0) ?
				(-yIndex[i] - height2 * ((-yIndex[i]) / height2))
				: (yIndex[i] - height2 * (yIndex[i] / height2)));
			if (height <= yIndex[i]) {
				yIndex[i] = height2 - yIndex[i];
			}
		}

		for (i = 0; i < 6; i++)
		{
			for (j = 0; j < 6; j++)
			{
				Para_Value = (*(Para + width * yIndex[i] + xIndex[j]));
				S[0] = S[0] + Para_Value * xWeight[j] * yWeight[i];
				if (S_Flag > -1)
				{
					S[1] = S[1] + Para_Value * xWeightGradient[j] * yWeight[i];
					S[2] = S[2] + Para_Value * xWeight[j] * yWeightGradient[i];
				}
			}
		}
	}
	else if ((Interpolation_Algorithm == 2) || (Interpolation_Algorithm == 5))
	{
		for (i = 0; i < 5; i++)
		{
			xIndex[i] = (width == 1) ? (0) : ((xIndex[i] < 0) ?
				(-xIndex[i] - width2 * ((-xIndex[i]) / width2))
				: (xIndex[i] - width2 * (xIndex[i] / width2)));
			if (width <= xIndex[i]) {
				xIndex[i] = width2 - xIndex[i];
			}
			yIndex[i] = (height == 1) ? (0) : ((yIndex[i] < 0) ?
				(-yIndex[i] - height2 * ((-yIndex[i]) / height2))
				: (yIndex[i] - height2 * (yIndex[i] / height2)));
			if (height <= yIndex[i]) {
				yIndex[i] = height2 - yIndex[i];
			}
		}
		for (i = 0; i < 5; i++)
		{
			for (j = 0; j < 5; j++)
			{
				Para_Value = (*(Para + width * yIndex[i] + xIndex[j]));
				S[0] = S[0] + Para_Value * xWeight[j] * yWeight[i];
				if (S_Flag > -1)
				{
					S[1] = S[1] + Para_Value * xWeightGradient[j] * yWeight[i];
					S[2] = S[2] + Para_Value * xWeight[j] * yWeightGradient[i];
				}
			}
		}
	}
	else
	{
		for (i = 0; i < 4; i++)
		{
			xIndex[i] = (width == 1) ? (0) : ((xIndex[i] < 0) ?
				(-xIndex[i] - width2 * ((-xIndex[i]) / width2))
				: (xIndex[i] - width2 * (xIndex[i] / width2)));
			if (width <= xIndex[i]) {
				xIndex[i] = width2 - xIndex[i];
			}
			yIndex[i] = (height == 1) ? (0) : ((yIndex[i] < 0) ?
				(-yIndex[i] - height2 * ((-yIndex[i]) / height2))
				: (yIndex[i] - height2 * (yIndex[i] / height2)));
			if (height <= yIndex[i]) {
				yIndex[i] = height2 - yIndex[i];
			}
		}
		for (i = 0; i < 4; i++)
		{
			for (j = 0; j < 4; j++)
			{
				Para_Value = (*(Para + width * yIndex[i] + xIndex[j]));
				S[0] = S[0] + Para_Value * xWeight[j] * yWeight[i];
				if (S_Flag > -1)
				{
					S[1] = S[1] + Para_Value * xWeightGradient[j] * yWeight[i];
					S[2] = S[2] + Para_Value * xWeight[j] * yWeightGradient[i];
				}
			}
		}
	}

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
void LensCorrection(vector<ImgData> &vImg, double *Intrinsic, double *distortion)
{
	int width = vImg[0].width, height = vImg[0].height, length = width*height, nchannels = vImg[0].nchannels;

	//Generate mapping data
	Point2d *MapXY = new Point2d[width*height];

	double K[9] = { Intrinsic[0], Intrinsic[2], Intrinsic[3], 0, Intrinsic[1], Intrinsic[4] };
	for (int jj = 0; jj < height; jj++)
		for (int ii = 0; ii < width; ii++)
			MapXY[ii + jj*width] = Point2d(ii, jj);

	LensDistortionPoint(MapXY, K, distortion, width*height);

	//undistort
	double *Para = new double[length*nchannels];
	unsigned char *Img = new unsigned char[length*nchannels];

	for (int ll = 0; ll < (int)vImg.size(); ll++)
	{
		for (int kk = 0; kk < nchannels; kk++)
		{
			for (int jj = 0; jj < height; jj++)
				for (int ii = 0; ii < width; ii++)
					Img[ii + jj*width + kk*width*height] = vImg[ll].color.data[ii*nchannels + jj*width*nchannels + kk];
			Generate_Para_Spline(Img + kk*width*height, Para + kk*width*height, width, height, 1);
		}

		double S[3];
		for (int jj = 0; jj < height; jj++)
		{
			for (int ii = 0; ii < width; ii++)
			{
				Point2d ImgPt = MapXY[ii + jj*width];
				if (ImgPt.x < 0 || ImgPt.x > width - 1 || ImgPt.y<0.0 || ImgPt.y > height - 1)
					for (int kk = 0; kk < nchannels; kk++)
						Img[ii + jj*width + kk*length] = (unsigned char)0;
				else
				{
					for (int kk = 0; kk < nchannels; kk++)
					{
						Get_Value_Spline(Para + kk*length, width, height, ImgPt.x, ImgPt.y, S, -1, 1);
						S[0] = min(max(S[0], 0.0), 255.0);
						vImg[ll].color.data[ii*nchannels + jj*width*nchannels + kk] = (unsigned char)(S[0] + 0.5);
					}
				}
			}
		}
	}

	delete[]Para, delete[]Img, delete[]MapXY;
	return;
}







