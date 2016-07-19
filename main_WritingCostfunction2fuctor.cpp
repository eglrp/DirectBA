#include "DataStructure.h"
#include "Ultility.h"
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

class BilinearInterp : public ceres::SizedCostFunction<1, 2>
{
public:
	BilinearInterp(uchar* Img, uchar *observed, int width, int height) :Img(Img), width(width), height(height)
	{
		for (int ii = 0; ii < 3; ii++)
			observation[ii] = observed[ii];
	}

	virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
	{
		double x = parameters[0][0], y = parameters[0][1];

		int xiD = (int)(x), yiD = (int)(y);
		int xiU = xiD + 1, yiU = yiD + 1;

		double f00 = (double)(int)Img[xiD + yiD*width];
		double f01 = (double)(int)Img[xiU + yiD*width];
		double f10 = (double)(int)Img[xiD + yiU*width];
		double  f11 = (double)(int)Img[xiU + yiU*width];
		double value = (f01 - f00)*(x - xiD) + (f10 - f00)*(y - yiD) + (f11 - f01 - f10 + f00)*(x - xiD)*(y - yiD) + f00;

		residuals[0] = value - (double)(int)observation[0];

		if (jacobians != NULL && jacobians[0] != NULL)
		{
			jacobians[0][0] = (f01 - f00) + (f11 - f01 - f10 + f00)*(y - yiD);
			jacobians[0][1] = (f10 - f00) + (f11 - f01 - f10 + f00)*(x - xiD);
		}
	}
	int width, height;
	uchar *Img, observation[3];
};

// Given samples from a function sampled at four equally spaced points,
//
//   p0 = f(-1)
//   p1 = f(0)
//   p2 = f(1)
//   p3 = f(2)
//
// Evaluate the cubic Hermite spline (also known as the Catmull-Rom
// spline) at a point x that lies in the interval [0, 1].
//
// This is also the interpolation kernel (for the case of a = 0.5) as
// proposed by R. Keys, in:
//
// "Cubic convolution interpolation for digital image processing".
// IEEE Transactions on Acoustics, Speech, and Signal Processing
// 29 (6): 1153–1160.
//
// For more details see
//
// http://en.wikipedia.org/wiki/Cubic_Hermite_spline
// http://en.wikipedia.org/wiki/Bicubic_interpolation
//
// f if not NULL will contain the interpolated function values.
// dfdx if not NULL will contain the interpolated derivative values.
template <int kDataDimension>	void CubicHermiteSpline(const Eigen::Matrix<double, kDataDimension, 1>& p0, const Eigen::Matrix<double, kDataDimension, 1>& p1, const Eigen::Matrix<double, kDataDimension, 1>& p2, const Eigen::Matrix<double, kDataDimension, 1>& p3, const double x, double* f, double* dfdx)
{
	DCHECK_GE(x, 0.0);
	DCHECK_LE(x, 1.0);
	typedef Eigen::Matrix<double, kDataDimension, 1> VType;
	const VType a = 0.5 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3);
	const VType b = 0.5 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3);
	const VType c = 0.5 * (-p0 + p2);
	const VType d = p1;

	// Use Horner's rule to evaluate the function value and its derivative.

	// f = ax^3 + bx^2 + cx + d
	if (f != NULL)
		Eigen::Map<VType>(f, kDataDimension) = d + x * (c + x * (b + x * a));

	// dfdx = 3ax^2 + 2bx + c
	if (dfdx != NULL)
		Eigen::Map<VType>(dfdx, kDataDimension) = c + x * (2.0 * b + 3.0 * a * x);
}

template<typename Grid>	class CERES_EXPORT BiCubicInterpolator
{
public:
	explicit BiCubicInterpolator(const Grid& grid) : grid_(grid){}

	// Evaluate the interpolated function value and/or its derivative. Returns false if r or c is out of bounds.
	void Evaluate(double r, double c, double* f, double* dfdr, double* dfdc) const
	{
		// BiCubic interpolation requires 16 values around the point being
		// evaluated.  We will use pij, to indicate the elements of the
		// 4x4 grid of values.
		//
		//          col
		//      p00 p01 p02 p03
		// row  p10 p11 p12 p13
		//      p20 p21 p22 p23
		//      p30 p31 p32 p33
		//
		// The point (r,c) being evaluated is assumed to lie in the square
		// defined by p11, p12, p22 and p21.
		const int row = std::floor(r);
		const int col = std::floor(c);

		Eigen::Matrix<double, Grid::DATA_DIMENSION, 1> p0, p1, p2, p3;

		// Interpolate along each of the four rows, evaluating the function
		// value and the horizontal derivative in each row.
		Eigen::Matrix<double, Grid::DATA_DIMENSION, 1> f0, f1, f2, f3;
		Eigen::Matrix<double, Grid::DATA_DIMENSION, 1> df0dc, df1dc, df2dc, df3dc;

		grid_.GetValue(row - 1, col - 1, p0.data());
		grid_.GetValue(row - 1, col, p1.data());
		grid_.GetValue(row - 1, col + 1, p2.data());
		grid_.GetValue(row - 1, col + 2, p3.data());
		CubicHermiteSpline<Grid::DATA_DIMENSION>(p0, p1, p2, p3, c - col, f0.data(), df0dc.data());

		grid_.GetValue(row, col - 1, p0.data());
		grid_.GetValue(row, col, p1.data());
		grid_.GetValue(row, col + 1, p2.data());
		grid_.GetValue(row, col + 2, p3.data());
		CubicHermiteSpline<Grid::DATA_DIMENSION>(p0, p1, p2, p3, c - col, f1.data(), df1dc.data());

		grid_.GetValue(row + 1, col - 1, p0.data());
		grid_.GetValue(row + 1, col, p1.data());
		grid_.GetValue(row + 1, col + 1, p2.data());
		grid_.GetValue(row + 1, col + 2, p3.data());
		CubicHermiteSpline<Grid::DATA_DIMENSION>(p0, p1, p2, p3, c - col, f2.data(), df2dc.data());

		grid_.GetValue(row + 2, col - 1, p0.data());
		grid_.GetValue(row + 2, col, p1.data());
		grid_.GetValue(row + 2, col + 1, p2.data());
		grid_.GetValue(row + 2, col + 2, p3.data());
		CubicHermiteSpline<Grid::DATA_DIMENSION>(p0, p1, p2, p3, c - col, f3.data(), df3dc.data());

		// Interpolate vertically the interpolated value from each row and
		// compute the derivative along the columns.
		CubicHermiteSpline<Grid::DATA_DIMENSION>(f0, f1, f2, f3, r - row, f, dfdr);
		if (dfdc != NULL)
			CubicHermiteSpline<Grid::DATA_DIMENSION>(df0dc, df1dc, df2dc, df3dc, r - row, dfdc, NULL);	// Interpolate vertically the derivative along the columns.
	}

	// The following two Evaluate overloads are needed for interfacing with automatic differentiation. The first is for when a scalar evaluation is done, and the second one is for when Jets are used.
	void Evaluate(const double& r, const double& c, double* f) const
	{
		Evaluate(r, c, f, NULL, NULL);
	}

	template<typename JetT> void Evaluate(const JetT& r, const JetT& c, JetT* f) const {
		double frc[Grid::DATA_DIMENSION];
		double dfdr[Grid::DATA_DIMENSION];
		double dfdc[Grid::DATA_DIMENSION];
		Evaluate(r.a, c.a, frc, dfdr, dfdc);
		for (int i = 0; i < Grid::DATA_DIMENSION; ++i) {
			f[i].a = frc[i];
			f[i].v = dfdr[i] * r.v + dfdc[i] * c.v;
		}
	}

private:
	const Grid& grid_;
};

// An object that implements an infinite two dimensional grid needed
// by the BiCubicInterpolator where the source of the function values
// is an grid of type T on the grid
//
//   [(row_start,   col_start), ..., (row_start,   col_end - 1)]
//   [                          ...                            ]
//   [(row_end - 1, col_start), ..., (row_end - 1, col_end - 1)]
//
// Since the input grid is finite and the grid is infinite, values
// outside this interval needs to be computed. Grid2D uses the value
// from the nearest edge.
//
// The function being provided can be vector valued, in which case
// kDataDimension > 1. The data maybe stored in row or column major
// format and the various dimensional slices of the function maybe
// interleaved, or they maybe stacked, i.e, if the function has
// kDataDimension = 2, is stored in row-major format and if
// kInterleaved = true, then it is stored as   f001, f002, f011, f012, ...
// A commonly occuring example are color images (RGB) where the three channels are stored interleaved.
// If kInterleaved = false, then it is stored as  f001, f011, ..., fnm1, f002, f012, ...
template <typename T, int kDataDimension = 1, bool kRowMajor = true, bool kInterleaved = true>	struct Grid2D
{
public:
	enum { DATA_DIMENSION = kDataDimension };

	Grid2D(const T* data, const int row_begin, const int row_end, const int col_begin, const int col_end) : data_(data), row_begin_(row_begin), row_end_(row_end), col_begin_(col_begin), col_end_(col_end), num_rows_(row_end - row_begin), num_cols_(col_end - col_begin), num_values_(num_rows_ * num_cols_)
	{
		CHECK_GE(kDataDimension, 1);
		CHECK_LT(row_begin, row_end);
		CHECK_LT(col_begin, col_end);
	}

	EIGEN_STRONG_INLINE void GetValue(const int r, const int c, double* f) const
	{
		const int row_idx = std::min(std::max(row_begin_, r), row_end_ - 1) - row_begin_;
		const int col_idx = std::min(std::max(col_begin_, c), col_end_ - 1) - col_begin_;
		const int n = (kRowMajor) ? num_cols_ * row_idx + col_idx : num_rows_ * col_idx + row_idx;


		if (kInterleaved) 
			for (int i = 0; i < kDataDimension; ++i) 
				f[i] = static_cast<double>(data_[kDataDimension * n + i]);
		else
			for (int i = 0; i < kDataDimension; ++i) 
				f[i] = static_cast<double>(data_[i * num_values_ + n]);
	}

private:
	const T* data_;
	const int row_begin_;
	const int row_end_;
	const int col_begin_;
	const int col_end_;
	const int num_rows_;
	const int num_cols_;
	const int num_values_;
};


struct DepthImgWarping2 {
	DepthImgWarping2(Point2i uv, double *rayDir, double *refCamCen, uchar *refImg, uchar *nonRefImgs, double *Intrinsic, Point2i &imgSize, int nchannels, int eleID, int imgID) :
		uv(uv), refCamCen(refCamCen), refImg(refImg), nonRefImgs(nonRefImgs), Intrinsic(Intrinsic), imgSize(imgSize), nchannels(nchannels), eleID(eleID), imgID(imgID)
	{
		for (int ii = 0; ii < 3; ii++)
			rayDirection[ii] = rayDir[ii];//ray direction changes for every data point. Pass by ref does not work
	}

	template <typename T>	bool operator()(const T* const invd, const T* const rt, T* residuals) const
	{
		//Parametes[0][0]: inverse depth for ref, Parameters[1..n][0..5]: poses for non-ref
		T XYZ[3];
		for (int ii = 0; ii < 3; ii++)
			XYZ[ii] = (T)rayDirection[ii] / invd[0] + (T)refCamCen[ii];

		//project to other views
		T tp[3], xcn, ycn, tu, tv, color[3];
		ceres::AngleAxisRotatePoint(rt, XYZ, tp);
		tp[0] += rt[3], tp[1] += rt[4], tp[2] += rt[5];
		xcn = tp[0] / tp[2], ycn = tp[1] / tp[2];

		tu = (T)Intrinsic[0] * xcn + (T)Intrinsic[2] * ycn + (T)Intrinsic[3];
		tv = (T)Intrinsic[1] * ycn + (T)Intrinsic[4];

		if (tu<(T)50 || tu>(T)(imgSize.x - 50) || tv<(T)50 || tv>(T)(imgSize.y - 50))
		{
			for (int jj = 0; jj < nchannels; jj++)
				residuals[jj] = (T)1000;
		}
		else
		{
			for (int jj = 0; jj < nchannels; jj++)
			{
				uchar refColor = refImg[uv.x + uv.y*imgSize.x];
				T dif = (T)color[jj] - (T)(double)(int)refColor;
				residuals[jj] = dif;
			}
		}

		return true;
	}

	static ceres::CostFunction* Create(Point2i uv, double *rayDir, double *refCamCen, uchar *refImg, uchar *nonRefImgs, double *Intrinsic, Point2i &imgSize, int nchannels, int eleID, int imgID)
	{
		return (new ceres::AutoDiffCostFunction<DepthImgWarping2, 1, 1, 6>(new DepthImgWarping2(uv, rayDir, refCamCen, refImg, nonRefImgs, Intrinsic, imgSize, nchannels, eleID, imgID)));
	}
	static ceres::CostFunction* CreateNumDif(Point2i uv, double *rayDir, double *refCamCen, uchar *refImg, uchar *nonRefImgs, double *Intrinsic, Point2i &imgSize, int nchannels, int eleID, int imgID)
	{
		return (new ceres::NumericDiffCostFunction<DepthImgWarping2, ceres::CENTRAL, 1, 1, 6>(new DepthImgWarping2(uv, rayDir, refCamCen, refImg, nonRefImgs, Intrinsic, imgSize, nchannels, eleID, imgID)));
	}

	uchar *refImg, *nonRefImgs;
	Point2i uv, imgSize;
	double rayDirection[3], *refCamCen, *Intrinsic;
	int  nchannels, eleID, imgID;
};

int DirectAlignment(vector<ImgData> &allImgs, vector<CameraData> &allCalibInfo)
{
	int width = allImgs[0].width, height = allImgs[0].height, nchannels = allImgs[0].nchannels;

	vector<int> u, v;
	vector<double> invd;
	//find texture region in the refImg and store in the vector
	for (int ii = 0; ii < (int)allImgs.size(); ii++)
		GaussianBlur(allImgs[ii].color, allImgs[ii].color, Size(5, 5), 0, 0);

	//calculate first order image derivatives
	float *dx = new float[width*height], *dy = new float[width*height];
	for (int jj = 1; jj < height - 1; jj++)
	{
		for (int ii = 1; ii < width - 1; ii++)
		{
			dx[ii + jj*width] = (float)(int)allImgs[0].color.data[ii + 1 + jj*width] - (float)(int)allImgs[0].color.data[ii - 1 + jj*width]; //-1, 0, 1
			dy[ii + jj*width] = (float)(int)allImgs[0].color.data[ii + (jj + 1)*width] - (float)(int)allImgs[0].color.data[ii + (jj - 1)*width]; //-1, 0, 1
		}
	}

	float mag2Thresh = 70;
	for (int jj = 0; jj < height; jj++)
	{
		for (int ii = 0; ii < width; ii++)
		{
			float x = allImgs[0].detph[0 + 0 * 1920];
			float mag2 = pow(dx[ii + jj*width], 2) + pow(dx[ii + jj*width], 2);
			if (mag2 > mag2Thresh  && allImgs[0].detph[ii + jj*width] > 0 && allImgs[0].detph[ii + jj*width] < 7 && allImgs[0].depthConf[ii + jj*width] >70)
				u.push_back(ii), v.push_back(jj), invd.push_back(1.0 / (allImgs[0].detph[ii + jj*width] + 1.0e-9));
		}
	}

	/*{
		//FILE *fpx = fopen("C:/temp/corres.txt", "w");
		//for (int ii = 0; ii < (int)invd.size(); ii += 10)
		int ii = 7300;
		{
		double rayDir[3], uv1[3] = { u[ii], v[ii], 1 };
		getRayDir(rayDir, allCalibInfo[0].invK, allCalibInfo[0].R, uv1);

		vector<double> colorProfile;
		colorProfile.push_back((double)(int)allImgs[0].color.data[u[ii] + v[ii] * 1920]);

		//back-project ref depth to 3D
		double XYZ[3], d = invd[ii];
		for (int jj = 0; jj < 3; jj++)
		XYZ[jj] = rayDir[jj] / d + allCalibInfo[0].camCenter[jj];

		//fprintf(fpx, "%d %d ", u[ii], v[ii]);
		for (int cid = 1; cid < (int)allImgs.size(); cid++)
		{
		Grid2D<uchar, 1>  img(allImgs[cid].color.data, 0, allCalibInfo[cid].height, 0, allCalibInfo[cid].width);
		BiCubicInterpolator<Grid2D < uchar, 1 > > imgInterp(img);

		double tp[3], xcn, ycn, tu, tv, color[3];
		ceres::AngleAxisRotatePoint(allCalibInfo[cid].rt, XYZ, tp);
		tp[0] += allCalibInfo[cid].rt[3], tp[1] += allCalibInfo[cid].rt[4], tp[2] += allCalibInfo[cid].rt[5];
		xcn = tp[0] / tp[2], ycn = tp[1] / tp[2];

		tu = allCalibInfo[cid].intrinsic[0] * xcn + allCalibInfo[cid].intrinsic[2] * ycn + allCalibInfo[cid].intrinsic[3];
		tv = allCalibInfo[cid].intrinsic[1] * ycn + allCalibInfo[cid].intrinsic[4];

		//imgInterp.Evaluate(tv, tu, color);//ceres takes row, column
		imgInterp.Evaluate(500, 500, color);//ceres takes row, column
		colorProfile.push_back(color[0]);
		}
		int a = 0;
		//fprintf(fpx, "\n");
		}
		//fclose(fpx);
		}*/

	//dump to Ceres
	ceres::Problem problem;
	ceres::LossFunction *loss_funcion = NULL;// new ceres::TukeyLoss(30);

	//for (int ii = 0; ii < (int)invd.size(); ii++)
	int ii = 7300;
	{
		double rayDir[3], uv1[3] = { u[ii], v[ii], 1 };
		getRayDir(rayDir, allCalibInfo[0].invK, allCalibInfo[0].R, uv1);

		//back-project ref depth to 3D
		double XYZ[3], d = invd[ii];
		for (int jj = 0; jj < 3; jj++)
			XYZ[jj] = rayDir[jj] / d + allCalibInfo[0].camCenter[jj];

		for (int cid = 1; cid < (int)allImgs.size(); cid++)
		{
			ceres::CostFunction* cost_function = DepthImgWarping2::Create(Point2i(u[ii], v[ii]), rayDir, allCalibInfo[0].camCenter,
				allImgs[0].color.data, allImgs[cid].color.data, allCalibInfo[cid].intrinsic, Point2i(allCalibInfo[cid].width, allCalibInfo[cid].height), nchannels, ii, cid);
			problem.AddResidualBlock(cost_function, loss_funcion, &invd[ii], allCalibInfo[cid].rt);
			problem.SetParameterBlockConstant(allCalibInfo[cid].rt);
		}
	}

	printf("...run BA...\n");
	ceres::Solver::Options options;
	options.minimizer_progress_to_stdout = true;
	options.num_threads = 1;// omp_get_max_threads();
	options.num_linear_solver_threads = 1;// omp_get_max_threads();
	options.trust_region_strategy_type = ceres::DOGLEG;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.use_inner_iterations = true;
	options.use_nonmonotonic_steps = false;
	options.max_num_iterations = 50;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";

	printf("Saving data...");
	FILE *fp = fopen("C:/temp/3d.xyz", "w+");
	for (int ii = 0; ii < (int)invd.size(); ii++)
	{
		double XYZ[3], rayDir[3], uv1[3] = { u[ii], v[ii], 1 };
		getRayDir(rayDir, allCalibInfo[0].invK, allCalibInfo[0].R, uv1);

		for (int kk = 0; kk < 3; kk++)
			XYZ[kk] = rayDir[kk] / invd[ii] + allCalibInfo[0].camCenter[kk];
		fprintf(fp, "%.4e %.4e %.4e\n", XYZ[0], XYZ[1], XYZ[2]);
	}
	fclose(fp);

	fp = fopen("C:/temp/pose.txt", "w+");
	for (int cid = 1; cid < (int)allImgs.size(); cid++)
		fprintf(fp, "%.16f %.16f %.16f %.16f %.16f %.16f\n", allCalibInfo[cid].rt[0], allCalibInfo[cid].rt[1], allCalibInfo[cid].rt[2], allCalibInfo[cid].rt[3], allCalibInfo[cid].rt[4], allCalibInfo[cid].rt[5]);
	fclose(fp);

	printf("Done\n");

	return 1;
}

int main(int argc, char** argv)
{
	char Fname[512], Path[] = "D:/Source/Repos/Users/sudipsin/recon3D/data/Y";

	vector<ImgData> allImgs;
	vector<CameraData> allCalibInfo;

	int syncId = 0;
	Corpus CorpusData;
	ReadSyncFile(Path, CorpusData, syncId);

	vector<int> camIDs; camIDs.push_back(26), camIDs.push_back(28), camIDs.push_back(30), camIDs.push_back(32), camIDs.push_back(22), camIDs.push_back(24), camIDs.push_back(20);
	for (int ii = 0; ii < camIDs.size(); ii++)
	{
		ImgData imgI;
		sprintf(Fname, "%s/Recon/urd-%03d.png", Path, camIDs[ii]);
		imgI.color = imread(Fname, 0);
		imgI.width = imgI.color.cols, imgI.height = imgI.color.rows, imgI.nchannels = 1;
		allImgs.push_back(imgI);
	}
	ReadDepthFile(Path, allImgs[0], camIDs[0]);

	for (int ii = 0; ii < camIDs.size(); ii++)
		allCalibInfo.push_back(CorpusData.camera[camIDs[ii]]);

	DirectAlignment(allImgs, allCalibInfo);

	for (int ii = 0; ii < camIDs.size(); ii++)
		for (int jj = 0; jj < 6; jj++)
			CorpusData.camera[camIDs[ii]].rt[jj] = allCalibInfo[ii].rt[jj];
	WriteSyncFile(Path, CorpusData, 1);
	return 0;
}
