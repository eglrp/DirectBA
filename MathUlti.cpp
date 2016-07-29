#include "MathUlti.h"

using namespace std;
using namespace cv;
using namespace Eigen;


bool IsNumber(double x)
{
	// This looks like it should always be true, but it's false if x is a NaN.
	return (x == x);
}
bool IsFiniteNumber(double x)
{
	return (x <= DBL_MAX && x >= -DBL_MAX);
}

double UniformNoise(double High, double Low)
{
	double noise = 1.0*rand() / RAND_MAX;
	return (High - Low)*noise + Low;
}
double gaussian_noise(double mean, double std)
{
	double u1 = 0.0, u2 = 0.0;
	while (abs(u1) < DBL_EPSILON || abs(u2) < DBL_EPSILON) //avoid 0.0 case since log(0) = inf
	{
		u1 = 1.0 * rand() / RAND_MAX;
		u2 = 1.0 * rand() / RAND_MAX;
	}

	double normal_noise = sqrt(-2.0 * log(u1)) * cos(2.0 * Pi * u2);
	return mean + std * normal_noise;
}

void normalize(double *x, int dim)
{
	double tt = 0;
	for (int ii = 0; ii < dim; ii++)
		tt += x[ii] * x[ii];
	tt = sqrt(tt);
	if (tt < FLT_EPSILON)
		return;
	for (int ii = 0; ii < dim; ii++)
		x[ii] = x[ii] / tt;
	return;
}
void mat_invert(double* mat, double* imat, int dims)
{
	if (dims == 2)
	{
		double a0 = mat[0], a1 = mat[1], a2 = mat[2], a3 = mat[3];
		double det = a0*a3 - a1*a2;
		if (abs(det) < 1e-9)
			printf("Caution. Matrix is ill-condition/n");

		imat[0] = a3 / det, imat[1] = -a1 / det;
		imat[2] = -a2 / det, imat[3] = a0 / det;
	}
	if (dims == 3)
	{
		// only work for 3x3
		double a = mat[0], b = mat[1], c = mat[2], d = mat[3], e = mat[4], f = mat[5], g = mat[6], h = mat[7], k = mat[8];
		double A = e*k - f*h, B = c*h - b*k, C = b*f - c*e;
		double D = f*g - d*k, E = a*k - c*g, F = c*d - a*f;
		double G = d*h - e*g, H = b*g - a*h, K = a*e - b*d;
		double DET = a*A + b*D + c*G;
		imat[0] = A / DET, imat[1] = B / DET, imat[2] = C / DET;
		imat[3] = D / DET, imat[4] = E / DET, imat[5] = F / DET,
			imat[6] = G / DET, imat[7] = H / DET, imat[8] = K / DET;
	}
	else
	{
		Mat inMat = Mat(dims, dims, CV_64FC1, mat);
		Mat outMat = inMat.inv(DECOMP_SVD);
		for (int jj = 0; jj < dims; jj++)
			for (int ii = 0; ii < dims; ii++)
				imat[ii + jj*dims] = outMat.at<double>(jj, ii);
	}

	return;
}
void mat_invert(float* mat, float* imat, int dims)
{
	if (dims == 3)
	{
		// only work for 3x3
		float a = mat[0], b = mat[1], c = mat[2], d = mat[3], e = mat[4], f = mat[5], g = mat[6], h = mat[7], k = mat[8];
		float A = e*k - f*h, B = c*h - b*k, C = b*f - c*e;
		float D = f*g - d*k, E = a*k - c*g, F = c*d - a*f;
		float G = d*h - e*g, H = b*g - a*h, K = a*e - b*d;
		float DET = a*A + b*D + c*G;
		imat[0] = A / DET, imat[1] = B / DET, imat[2] = C / DET;
		imat[3] = D / DET, imat[4] = E / DET, imat[5] = F / DET,
			imat[6] = G / DET, imat[7] = H / DET, imat[8] = K / DET;
	}
	else
	{
		Mat inMat = Mat(dims, dims, CV_32FC1, mat);
		Mat outMat = inMat.inv(DECOMP_SVD);
		for (int jj = 0; jj < dims; jj++)
			for (int ii = 0; ii < dims; ii++)
				imat[ii + jj*dims] = outMat.at<float>(jj, ii);
	}

	return;
}

void mat_transpose(double *in, double *out, int row_in, int col_in)
{
	int ii, jj;
	for (jj = 0; jj < row_in; jj++)
		for (ii = 0; ii < col_in; ii++)
			out[ii*row_in + jj] = in[jj*col_in + ii];
	return;
}
void mat_completeSym(double *mat, int size, bool upper)
{
	if (upper)
	{
		for (int jj = 0; jj < size; jj++)
			for (int ii = jj; ii < size; ii++)
				mat[jj + ii*size] = mat[ii + jj*size];
	}
	else
	{
		for (int jj = 0; jj < size; jj++)
			for (int ii = jj; ii < size; ii++)
				mat[ii + jj*size] = mat[jj + ii*size];
	}
	return;
}

void Quick_Sort_Int(int * A, int *B, int low, int high)
{
	m_TemplateClass_1<int> m_TempClass;
	m_TempClass.Quick_Sort(A, B, low, high);
	return;
}
void Quick_Sort_Float(float * A, int *B, int low, int high)
{
	m_TemplateClass_1<float> m_TempClass;
	m_TempClass.Quick_Sort(A, B, low, high);
	return;
}
void Quick_Sort_Double(double * A, int *B, int low, int high)
{
	m_TemplateClass_1<double> m_TempClass;
	m_TempClass.Quick_Sort(A, B, low, high);
	return;
}

void GetIntrinsicScaled(double *Intrinsic, double *IntrinsicScaled, double s)
{
	for (int ii = 0; ii < 5; ii++)
		IntrinsicScaled[ii] = Intrinsic[ii] * s;
}
void GetKFromIntrinsic(double *K, double *intrinsic)
{
	K[0] = intrinsic[0], K[1] = intrinsic[2], K[2] = intrinsic[3];
	K[3] = 0.0, K[4] = intrinsic[1], K[5] = intrinsic[4];
	K[6] = 0.0, K[7] = 0.0, K[8] = 1.0;
	return;
}
void GetiK(double *iK, double *K)
{
	mat_invert(K, iK);
	return;
}

void getTwistFromRT(double *R, double *T, double *twist)
{
	//OpenCV code to handle log map for SO(3)
	Map < Matrix < double, 3, 3, RowMajor > > matR(R); //matR is referenced to R;
	JacobiSVD<MatrixXd> svd(matR, ComputeFullU | ComputeFullV);
	//Matrix3d S = svd.singularValues().asDiagonal();
	matR = svd.matrixU()*svd.matrixV().transpose();//Project R to SO(3)

	double rx = R[7] - R[5], ry = R[2] - R[6], rz = R[3] - R[1];
	double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
	double c = (R[0] + R[4] + R[8] - 1)*0.5;
	c = c > 1. ? 1. : c < -1. ? -1. : c;
	double theta = acos(c);

	if (s < 1e-5)
	{
		double t;
		if (c > 0)
			rx = ry = rz = 0.0;
		else
		{
			t = (R[0] + 1)*0.5, rx = sqrt(MAX(t, 0.));
			t = (R[4] + 1)*0.5, ry = sqrt(MAX(t, 0.))*(R[1] < 0 ? -1. : 1.);
			t = (R[8] + 1)*0.5, rz = sqrt(MAX(t, 0.))*(R[2] < 0 ? -1. : 1.);
			if (fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R[5] > 0) != (ry*rz > 0))
				rz = -rz;
			theta /= sqrt(rx*rx + ry*ry + rz*rz);
			rx *= theta, ry *= theta, rz *= theta;
		}
	}
	else
	{
		double vth = 1.0 / (2.0 * s);
		vth *= theta;
		rx *= vth; ry *= vth; rz *= vth;
	}
	twist[3] = rx, twist[4] = ry, twist[5] = rz;

	//Compute V
	double theta2 = theta* theta;
	double wx[9] = { 0.0, -rz, ry, rz, 0.0, -rx, -ry, rx, 0.0 };
	double wx2[9]; mat_mul(wx, wx, wx2, 3, 3, 3);

	double V[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
	if (theta < 1.0e-9)
		twist[0] = T[0], twist[1] = T[1], twist[2] = T[2];
	else
	{
		double A = sin(theta) / theta, B = (1.0 - cos(theta)) / theta2, C = (1.0 - A) / theta2;
		for (int ii = 0; ii < 9; ii++)
			V[ii] += B*wx[ii] + C*wx2[ii];
	}

	//solve Vt = T;
	Map < Matrix < double, 3, 3, RowMajor > > matV(V);
	Map<Vector3d> matT(T), matt(twist);
	matt = matV.lu().solve(matT);

	return;
}
void getRTFromTwist(double *twist, double *R, double *T)
{
	double t[3] = { twist[0], twist[1], twist[2] };
	double w[3] = { twist[3], twist[4], twist[5] };

	double theta = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]), theta2 = theta* theta;
	double wx[9] = { 0.0, -w[2], w[1], w[2], 0.0, -w[0], -w[1], w[0], 0.0 };
	double wx2[9]; mat_mul(wx, wx, wx2, 3, 3, 3);

	R[0] = 1.0, R[1] = 0.0, R[2] = 0.0;
	R[3] = 0.0, R[4] = 1.0, R[5] = 0.0;
	R[6] = 0.0, R[7] = 0.0, R[8] = 1.0;

	double V[9] = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
	if (theta < 1.0e-9)
		T[0] = t[0], T[1] = t[1], T[2] = t[2]; //Rotation is idenity
	else
	{
		double A = sin(theta) / theta, B = (1.0 - cos(theta)) / theta2, C = (1.0 - A) / theta2;
		for (int ii = 0; ii < 9; ii++)
			R[ii] += A*wx[ii] + B*wx2[ii];

		for (int ii = 0; ii < 9; ii++)
			V[ii] += B*wx[ii] + C*wx2[ii];
		mat_mul(V, t, T, 3, 3, 1);
	}

	return;
}
void getrFromR(double *R, double *r)
{
	//Project R to SO(3)
	Map < Matrix < double, 3, 3, RowMajor > > matR(R); //matR is referenced to R;
	JacobiSVD<MatrixXd> svd(matR, ComputeFullU | ComputeFullV);
	matR = svd.matrixU()*svd.matrixV().transpose();

	double rx = R[7] - R[5], ry = R[2] - R[6], rz = R[3] - R[1];
	double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
	double c = (R[0] + R[4] + R[8] - 1)*0.5;
	c = c > 1. ? 1. : c < -1. ? -1. : c;
	double theta = acos(c);

	if (s < 1e-5)
	{
		double t;
		if (c > 0)
			rx = ry = rz = 0.0;
		else
		{
			t = (R[0] + 1)*0.5, rx = sqrt(MAX(t, 0.));
			t = (R[4] + 1)*0.5, ry = sqrt(MAX(t, 0.))*(R[1] < 0 ? -1. : 1.);
			t = (R[8] + 1)*0.5, rz = sqrt(MAX(t, 0.))*(R[2] < 0 ? -1. : 1.);
			if (fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R[5] > 0) != (ry*rz > 0))
				rz = -rz;
			theta /= sqrt(rx*rx + ry*ry + rz*rz);
			rx *= theta, ry *= theta, rz *= theta;
		}
	}
	else
	{
		double vth = 1.0 / (2.0 * s);
		vth *= theta;
		rx *= vth; ry *= vth; rz *= vth;
	}
	r[0] = rx, r[1] = ry, r[2] = rz;

	return;
}
void getRfromr(double *r, double *R)
{
	/*Mat Rmat(3, 3, CV_64F), rvec(3, 1, CV_64F);
	for (int jj = 0; jj < 3; jj++)
	rvec.at<double>(jj) = r[jj];

	Rodrigues(rvec, Rmat);

	for (int jj = 0; jj < 9; jj++)
	R[jj] = Rmat.at<double>(jj);*/

	double theta = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]), theta2 = theta* theta;
	double rx[9] = { 0.0, -r[2], r[1], r[2], 0.0, -r[0], -r[1], r[0], 0.0 };
	double rx2[9]; mat_mul(rx, rx, rx2, 3, 3, 3);

	R[0] = 1.0, R[1] = 0.0, R[2] = 0.0;
	R[3] = 0.0, R[4] = 1.0, R[5] = 0.0;
	R[6] = 0.0, R[7] = 0.0, R[8] = 1.0;

	if (theta < 1.0e-9)
		return;
	else
	{
		double A = sin(theta) / theta, B = (1.0 - cos(theta)) / theta2, C = (1.0 - A) / theta2;
		for (int ii = 0; ii < 9; ii++)
			R[ii] += A*rx[ii] + B*rx2[ii];
	}

	return;
}

void GetrtFromRT(double *rt, double *R, double *T)
{
	Mat Rmat(3, 3, CV_64F), r(3, 1, CV_64F);

	for (int jj = 0; jj < 9; jj++)
		Rmat.at<double>(jj) = R[jj];

	Rodrigues(Rmat, r);

	for (int jj = 0; jj < 3; jj++)
		rt[jj] = r.at<double>(jj), rt[3 + jj] = T[jj];

	return;
}
void GetRTFromrt(double *rt, double *R, double *T)
{
	Mat Rmat(3, 3, CV_64F), rvec(3, 1, CV_64F);
	for (int jj = 0; jj < 3; jj++)
		rvec.at<double>(jj) = rt[jj];

	Rodrigues(rvec, Rmat);

	for (int jj = 0; jj < 9; jj++)
		R[jj] = Rmat.at<double>(jj);
	for (int jj = 0; jj < 3; jj++)
		T[jj] = rt[jj + 3];

	return;
}

void GetRTfromrt(double *rt, double *RT)
{
	double R[9], T[3];
	GetRTFromrt(rt, R, T);
	AssembleRT(R, T, RT, false);
	return;
}

void GetTfromC(double *R, double *C, double *T)
{
	mat_mul(R, C, T, 3, 3, 1);
	T[0] = -T[0], T[1] = -T[1], T[2] = -T[2];
	return;
}
void GetCfromT(double *R, double *T, double *C)
{
	//C = -R't;
	double iR[9];
	mat_transpose(R, iR, 3, 3);

	mat_mul(iR, T, C, 3, 3, 1);
	C[0] = -C[0], C[1] = -C[1], C[2] = -C[2];
	return;
}

void getRayDir(double *rayDir, double *iK, double *R, double *uv1)
{
	//rayDirection = iR*iK*[u,v,1]
	Map< const MatrixXdr >	eR(R, 3, 3);
	Map< const MatrixXdr >	eiK(iK, 3, 3);
	Map<Vector3d> euv1(uv1, 3);
	Map<Vector3d> eRayDir(rayDir, 3);
	eRayDir = eR.transpose()*eiK*euv1;
	//eRayDir = eRayDir / eRayDir.norm();

	return;
}
void AssembleRT(double *R, double *T, double *RT, bool GivenCenter)
{
	if (!GivenCenter)
	{
		RT[0] = R[0], RT[1] = R[1], RT[2] = R[2], RT[3] = T[0];
		RT[4] = R[3], RT[5] = R[4], RT[6] = R[5], RT[7] = T[1];
		RT[8] = R[6], RT[9] = R[7], RT[10] = R[8], RT[11] = T[2];
	}
	else//RT = [R, -R*C];
	{
		double mT[3];
		mat_mul(R, T, mT, 3, 3, 1);
		RT[0] = R[0], RT[1] = R[1], RT[2] = R[2], RT[3] = -mT[0];
		RT[4] = R[3], RT[5] = R[4], RT[6] = R[5], RT[7] = -mT[1];
		RT[8] = R[6], RT[9] = R[7], RT[10] = R[8], RT[11] = -mT[2];
	}
}
void DesembleRT(double *R, double *T, double *RT)
{
	R[0] = RT[0], R[1] = RT[1], R[2] = RT[2], T[0] = RT[3];
	R[3] = RT[4], R[4] = RT[5], R[5] = RT[6], T[1] = RT[7];
	R[6] = RT[8], R[7] = RT[9], R[8] = RT[10], T[2] = RT[11];
}
void AssembleP(double *K, double *R, double *T, double *P)
{
	double RT[12];
	Set_Sub_Mat(R, RT, 3, 3, 4, 0, 0);
	Set_Sub_Mat(T, RT, 1, 3, 4, 3, 0);
	mat_mul(K, RT, P, 3, 3, 4);
	return;
}
void AssembleP(double *K, double *RT, double *P)
{
	mat_mul(K, RT, P, 3, 3, 4);
	return;
}

void ConvertFrontoDepth2LineOfSightDepth(ImgData &Img, CameraData &Cam)
{
	int width = Img.width, height = Img.height;

	getRfromr(Cam.rt, Cam.R);
	double rayDir[3], opticalAxis[3], ij[3] = { Cam.intrinsic[3], Cam.intrinsic[4], 1 };
	getRayDir(opticalAxis, Cam.invK, Cam.R, ij);
	double normOptical = sqrt(pow(opticalAxis[0], 2) + pow(opticalAxis[1], 2) + pow(opticalAxis[2], 2));

	for (int jj = 0; jj < height; jj++)
	{
		for (int ii = 0; ii < width; ii++)
		{
			double ij[3] = { ii, jj, 1 };
			getRayDir(rayDir, Cam.invK, Cam.R, ij);
			double cos = (rayDir[0] * opticalAxis[0] + rayDir[1] * opticalAxis[1] + rayDir[2] * opticalAxis[2]) /
				sqrt(pow(rayDir[0], 2) + pow(rayDir[1], 2) + pow(rayDir[2], 2)) / normOptical;
			double fronto_d = Img.depth[ii + jj*width];
			if (fronto_d < 0)
				Img.depth[ii + jj*width] = 0;
			else
				Img.depth[ii + jj*width] = fronto_d / cos;
		}
	}
}
void ConvertLineOfSightDepth2FrontoDepth(ImgData &Img, CameraData &Cam)
{
	int width = Img.width, height = Img.height;

	getRfromr(Cam.rt, Cam.R);
	double rayDir[3], opticalAxis[3], ij[3] = { Cam.intrinsic[3], Cam.intrinsic[4], 1 };
	getRayDir(opticalAxis, Cam.invK, Cam.R, ij);
	double normOptical = sqrt(pow(opticalAxis[0], 2) + pow(opticalAxis[1], 2) + pow(opticalAxis[2], 2));

	for (int jj = 0; jj < height; jj++)
	{
		for (int ii = 0; ii < width; ii++)
		{
			double ij[3] = { ii, jj, 1 };
			getRayDir(rayDir, Cam.invK, Cam.R, ij);
			double cos = (rayDir[0] * opticalAxis[0] + rayDir[1] * opticalAxis[1] + rayDir[2] * opticalAxis[2]) /
				sqrt(pow(rayDir[0], 2) + pow(rayDir[1], 2) + pow(rayDir[2], 2)) / normOptical;
			double los_d = Img.depth[ii + jj*width];
			if (los_d < 0)
				Img.depth[ii + jj*width] = 0;
			else
				Img.depth[ii + jj*width] = los_d * cos;
		}
	}
}
void ConvertDisparirty2DepthMap(ImgData &Img, double f, double b, double doffs)
{
	for (int jj = 0; jj < Img.height; jj++)
		for (int ii = 0; ii < Img.width; ii++)
			Img.depth[ii + jj*Img.width] = (float)(b*f / (Img.depth[ii + jj*Img.width] + doffs));
	return;
}
void ConvertDepthMap2PointCloud(ImgData &Img, CameraData &Cam, vector<Point3f> &PCloud)
{
	PCloud.reserve(Img.height*Img.width);

	double uv1[3], rayDir[3], xyz[3];

	for (int jj = 0; jj < Img.height; jj++)
	{
		for (int ii = 0; ii < Img.width; ii++)
		{
			uv1[0] = ii, uv1[1] = jj, uv1[2] = 1.0;
			getRayDir(rayDir, Cam.invK, Cam.R, uv1);

			for (int kk = 0; kk < 3; kk++)
				xyz[kk] = Img.depth[ii + jj*Img.width] * rayDir[kk];

			PCloud.push_back(Point3f(xyz[0], xyz[1], xyz[2]));
		}
	}

	return;
}
