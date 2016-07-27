#include "DataStructure.h"
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

struct DepthImgWarping {
	DepthImgWarping(Point2i uv, double *xycnRef_, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int eleID, int imgID) :
		uv(uv), refImg(refImg), nonRefImgs(nonRefImgs), intrinsic(intrinsic), imgSize(imgSize), nchannels(nchannels), isigma(isigma), eleID(eleID), imgID(imgID)
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

		int hb = 0;
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
				T ssd = (T)0;
				for (int jj = -hb; jj <= hb; jj++)
				{
					for (int ii = -hb; ii <= hb; ii++)
					{
						Imgs.Evaluate(tv + (T)jj, tu + (T)ii, color);//ceres takes row, column
						for (int kk = 0; kk < nchannels; kk++)
						{
							uchar refColor = refImg[(uv.x + ii + (uv.y + jj)*imgSize.x)*nchannels + kk];
							ssd += Parameters[3][0] * color[kk] + Parameters[3][1] - (T)(double)(int)refColor;
						}
					}
				}
				residuals[0] = (T)isigma*ssd / (T)((2 * hb + 1)*(2 * hb + 1) * 3);
			}
		}

		return true;
	}
private:
	uchar *refImg, *nonRefImgs;
	Point2i uv, imgSize;
	double xycnRef[3], *intrinsic, isigma;
	int  nchannels, eleID, imgID, boundary;
};
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
	DepthImgWarpingSmall(Point2i uv, double *xycnRef_, uchar *refImg, uchar *nonRefImgs, double *intrinsic, Point2i &imgSize, int nchannels, double isigma, int eleID, int imgID) :
		uv(uv), refImg(refImg), nonRefImgs(nonRefImgs), intrinsic(intrinsic), imgSize(imgSize), nchannels(nchannels), isigma(isigma), eleID(eleID), imgID(imgID)
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

		if (nchannels == 1)
		{
			Grid2D<uchar, 1>  img(nonRefImgs, 0, imgSize.y, 0, imgSize.x);
			BiCubicInterpolator<Grid2D < uchar, 1 > > Imgs(img);

			if (tu<(T)boundary || tu>(T)(imgSize.x - boundary) || tv<(T)boundary || tv>(T)(imgSize.y - boundary))
				residuals[0] = (T)1000;
			else
			{
				Imgs.Evaluate(tv, tu, color);//ceres takes row, column
				uchar refColor = refImg[uv.x + uv.y*imgSize.x];
				T dif = Parameters[3][0] * color[0] + Parameters[3][1] - (T)(double)(int)refColor;
				residuals[0] = (T)isigma*dif;
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
				Imgs.Evaluate(tv, tu, color);//ceres takes row, column
				for (int jj = 0; jj < nchannels; jj++)
				{
					uchar refColor = refImg[(uv.x + uv.y*imgSize.x)*nchannels + jj];
					T dif = Parameters[3][0] * color[jj] + Parameters[3][1] - (T)(double)(int)refColor;
					residuals[jj] = (T)isigma*dif;
				}
			}
		}

		return true;
	}
private:
	uchar *refImg, *nonRefImgs;
	Point2i uv, imgSize;
	double xycnRef[3], *intrinsic, isigma;
	int  nchannels, eleID, imgID, boundary;
};
void DirectAlignmentToFile(char *Path, DirectAlignPara &alignmentParas, vector<ImgData> &allImgs, vector<CameraData> &allCalibInfo, int pyrID)
{
	char Fname[512];

	double dataWeight = alignmentParas.dataWeight, regIntraWeight = alignmentParas.regIntraWeight, regInterWeight = alignmentParas.regInterWeight;
	double colorSigma = alignmentParas.colorSigma, depthSigma = alignmentParas.depthSigma; //expected std of variables (grayscale, mm);
	int nchannels = allImgs[0].nchannels;

	//find texture region in the refImg and store in the vector
	vector<float *> dxAll, dyAll;
	for (int cid = 0; cid < (int)allImgs.size(); cid++)
	{
		int width = allImgs[cid].imgPyr[pyrID].cols, height = allImgs[cid].imgPyr[pyrID].rows, boundary = width / 50;
		float *dx = new float[width*height], *dy = new float[width*height];
		for (int jj = boundary; jj < height - boundary; jj++)
		{
			for (int ii = boundary; ii < width - boundary; ii++)
			{
				//calculate first order image derivatives: using 1 channel should be enough
				dx[ii + jj*width] = (float)(int)allImgs[cid].imgPyr[pyrID].data[(ii + 1)*nchannels + jj*nchannels*width] - (float)(int)allImgs[cid].imgPyr[pyrID].data[(ii - 1)*nchannels + jj*nchannels*width]; //-1, 0, 1
				dy[ii + jj*width] = (float)(int)allImgs[cid].imgPyr[pyrID].data[ii *nchannels + (jj + 1)*nchannels*width] - (float)(int)allImgs[cid].imgPyr[pyrID].data[ii *nchannels + (jj - 1)*nchannels*width]; //-1, 0, 1
			}
		}
		dxAll.push_back(dx), dyAll.push_back(dy);
	}

	//Getting valid pixels (high texture && has depth &&good conf)
	float mag2Thresh = alignmentParas.gradientThresh2; //suprisingly, using more relaxed thresholding better convergence than aggressive one. 
	vector<int> *indIAll = new vector<int>[allImgs.size()], *indJAll = new vector<int>[allImgs.size()];
	vector<double> *invDAll = new vector<double>[allImgs.size()];
	for (int cid = 0; cid < (int)allImgs.size(); cid++)
	{
		int width = allImgs[cid].imgPyr[pyrID].cols, height = allImgs[cid].imgPyr[pyrID].rows, boundary = width / 50;
		invDAll[cid].reserve(width*height);
		indIAll[cid].reserve(width*height), indJAll[cid].reserve(width*height);

		for (int jj = boundary; jj < height - boundary; jj++)
		{
			for (int ii = boundary; ii < width - boundary; ii++)
			{
				float mag2 = pow(dxAll[cid][ii + jj*width], 2) + pow(dyAll[cid][ii + jj*width], 2);
				if (IsNumber(allImgs[cid].depthPyr[pyrID][ii + jj*width]) == 1 && mag2 > mag2Thresh && abs(allImgs[cid].depthPyr[pyrID][ii + jj*width]) >0.0)
					indIAll[cid].push_back(ii), indJAll[cid].push_back(jj), invDAll[cid].push_back(1.0 / (allImgs[cid].depthPyr[pyrID][ii + jj*width]));
			}
		}
	}

	for (int cid = 0; cid < (int)allImgs.size(); cid++)
	{
		for (int jj = 0; jj < (int)allImgs.size(); jj++)
			allCalibInfo[cid].photometric.push_back(1.0), allCalibInfo[cid].photometric.push_back(0);

		GetIntrinsicScaled(allCalibInfo[cid].intrinsic, allCalibInfo[cid].activeIntrinsic, allImgs[cid].scaleFactor[pyrID]);
		GetKFromIntrinsic(allCalibInfo[cid].activeK, allCalibInfo[cid].activeIntrinsic);
		GetiK(allCalibInfo[cid].activeinvK, allCalibInfo[cid].activeK);
	}

	printf("Saving data...");

	sprintf(Fname, "%s/pose.txt", Path); FILE *fp = fopen(Fname, "a+");
	for (int cid = 0; cid < (int)allImgs.size(); cid++)
		fprintf(fp, "Cid: %d  PyrID: %d %.16f %.16f %.16f %.16f %.16f %.16f %\n", cid, pyrID,
		allCalibInfo[cid].rt[0], allCalibInfo[cid].rt[1], allCalibInfo[cid].rt[2], allCalibInfo[cid].rt[3], allCalibInfo[cid].rt[4], allCalibInfo[cid].rt[5]);
	fclose(fp);

	//for (int cid = 0; cid < (int)allImgs.size(); cid++)
	{
		int cid = 0;
		int width = allImgs[cid].imgPyr[pyrID].cols, height = allImgs[cid].imgPyr[pyrID].rows, lengthJ = width*height;
		float *depthMap = new float[width*height];
		for (int ii = 0; ii < width*height; ii++)
			depthMap[ii] = 0.f;
		for (int ii = 0; ii < (int)invDAll[cid].size(); ii++)
		{
			double ij[3] = { indIAll[cid][ii], indJAll[cid][ii], 1 };
			depthMap[indIAll[cid][ii] + indJAll[cid][ii] * width] = 1.0 / invDAll[cid][ii];
		}
		sprintf(Fname, "%s/D_%d_%d.dat", Path, cid, pyrID), WriteGridBinary(Fname, depthMap, width, height, 1);
		sprintf(Fname, "%s/invD_%d_%d.png", Path, cid, pyrID), WriteInvDepthToImage(Fname, depthMap, width, height);

		if (nchannels == 1)
			sprintf(Fname, "%s/3d_%d_%d.xyz", Path, cid, pyrID);
		else
			sprintf(Fname, "%s/3d_%d_%d.txt", Path, cid, pyrID);
		FILE *fp = fopen(Fname, "w+");
		for (int ii = 0; ii < (int)invDAll[cid].size(); ii++)
		{
			int i = indIAll[cid][ii], j = indJAll[cid][ii];
			double XYZ[3], rayDir[3], ij[3] = { i, j, 1 };
			getRfromr(allCalibInfo[cid].rt, allCalibInfo[cid].R);
			GetCfromT(allCalibInfo[cid].R, allCalibInfo[cid].rt + 3, allCalibInfo[cid].C);

			getRayDir(rayDir, allCalibInfo[cid].activeinvK, allCalibInfo[cid].R, ij);
			for (int kk = 0; kk < 3; kk++)
				XYZ[kk] = rayDir[kk] / invDAll[cid][ii] + allCalibInfo[cid].C[kk];
			if (nchannels == 1)
				fprintf(fp, "%.8e %.8e %.8e\n", XYZ[0], XYZ[1], XYZ[2]);
			else
				fprintf(fp, "%.8e %.8e %.8e %d %d %d\n", XYZ[0], XYZ[1], XYZ[2],
				(int)allImgs[cid].imgPyr[pyrID].data[(i + j*width)*nchannels], (int)allImgs[cid].imgPyr[pyrID].data[(i + j*width)*nchannels + 1], (int)allImgs[cid].imgPyr[pyrID].data[(i + j*width)*nchannels + 2]);
		}
		fclose(fp);
	}
	printf("Done\n");

	delete[]indIAll, delete[]invDAll;
	for (int ii = 0; ii < (int)dxAll.size(); ii++)
		delete[]dxAll[ii], delete[]dyAll[ii];

	return;
}
double DirectAlignmentPyr(char *Path, DirectAlignPara &alignmentParas, vector<ImgData> &allImgs, vector<CameraData> &allCalibInfo, int fixPose, int fixDepth, int pyrID, int SmallAngle = 0, int verbose = 0)
{
	char Fname[512];

	double dataWeight = alignmentParas.dataWeight, regIntraWeight = alignmentParas.regIntraWeight, regInterWeight = alignmentParas.regInterWeight;
	double colorSigma = alignmentParas.colorSigma, depthSigma = alignmentParas.depthSigma; //expected std of variables (grayscale, mm);
	int nchannels = allImgs[0].nchannels;

	//find texture region in the refImg and store in the vector
	vector<float *> dxAll, dyAll;
	for (int cid = 0; cid < (int)allImgs.size(); cid++)
	{
		int width = allImgs[cid].imgPyr[pyrID].cols, height = allImgs[cid].imgPyr[pyrID].rows, boundary = width / 50;
		float *dx = new float[width*height], *dy = new float[width*height];
		for (int jj = boundary; jj < height - boundary; jj++)
		{
			for (int ii = boundary; ii < width - boundary; ii++)
			{
				//calculate first order image derivatives: using 1 channel should be enough
				dx[ii + jj*width] = (float)(int)allImgs[cid].imgPyr[pyrID].data[(ii + 1)*nchannels + jj*nchannels*width] - (float)(int)allImgs[cid].imgPyr[pyrID].data[(ii - 1)*nchannels + jj*nchannels*width]; //1, 0, -1
				dy[ii + jj*width] = (float)(int)allImgs[cid].imgPyr[pyrID].data[ii *nchannels + (jj + 1)*nchannels*width] - (float)(int)allImgs[cid].imgPyr[pyrID].data[ii *nchannels + (jj - 1)*nchannels*width]; //1, 0, -1
			}
		}
		dxAll.push_back(dx), dyAll.push_back(dy);
	}

	//Getting valid pixels (high texture && has depth &&good conf)
	float mag2Thresh = alignmentParas.gradientThresh2; //suprisingly, using more relaxed thresholding better convergence than aggressive one. 
	vector<bool *> validPixelsAll;
	vector<int*> sub2indAll;
	vector<int> *indIAll = new vector<int>[allImgs.size()], *indJAll = new vector<int>[allImgs.size()];
	vector<double> *invDAll = new vector<double>[allImgs.size()];
	for (int cid = 0; cid < (int)allImgs.size(); cid++)
	{
		int width = allImgs[cid].imgPyr[pyrID].cols, height = allImgs[cid].imgPyr[pyrID].rows, boundary = width / 50;
		int *sub2ind = new int[width*height];
		bool *validPixels = new bool[width*height];
		for (int ii = 0; ii < width*height; ii++)
			validPixels[ii] = false, sub2ind[ii] = -1;

		invDAll[cid].reserve(width*height);
		indIAll[cid].reserve(width*height), indJAll[cid].reserve(width*height);

		//int ii = 134, jj = 179;
		int ii = (int)(1196*allImgs[0].scaleFactor[pyrID]), jj = (int)(921*allImgs[0].scaleFactor[pyrID]);
		//for (int jj = boundary; jj < height - boundary; jj++)
		{
			//for (int ii = boundary; ii < width - boundary; ii++)
			{
				float mag2 = pow(dxAll[cid][ii + jj*width], 2) + pow(dyAll[cid][ii + jj*width], 2);
				if (IsNumber(allImgs[cid].depthPyr[pyrID][ii + jj*width]) == 1 && mag2 > mag2Thresh && abs(allImgs[cid].depthPyr[pyrID][ii + jj*width]) >0.0)
				{
					indIAll[cid].push_back(ii), indJAll[cid].push_back(jj), invDAll[cid].push_back(1.0 / (allImgs[cid].depthPyr[pyrID][ii + jj*width])),
						validPixels[ii + jj*width] = true, sub2ind[ii + jj*width] = (int)indIAll[cid].size() - 1;
				}
			}
		}
		validPixelsAll.push_back(validPixels), sub2indAll.push_back(sub2ind);
	}

	//detecting nearby pixels for regularization
	vector<vector<int> > nNNAll;
	vector<vector<int*> > NNAll;
	for (int cid = 0; cid < (int)allImgs.size(); cid++)
	{
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

	{
		//Sliding the parameters
		double orgx = allCalibInfo[1].rt[0], orgy = allCalibInfo[1].rt[1], orgz = allCalibInfo[1].rt[2];
		

		Grid2D<uchar, 1>  img1(allImgs[1].imgPyr[pyrID].data, 0, allImgs[1].height, 0, allImgs[1].width);
		BiCubicInterpolator<Grid2D < uchar, 1 > > imgInterp1(img1);

		Grid2D<uchar, 3>  img3(allImgs[1].imgPyr[pyrID].data, 0, allImgs[1].height, 0, allImgs[1].width);
		BiCubicInterpolator<Grid2D < uchar, 3 > > imgInterp3(img3);

		omp_set_num_threads(omp_get_max_threads());

		for (int x = 0; x <= 0; x++)
		{
			for (int y = -0; y <= 0; y++)
			{
				for (int z = -0; z <= 0; z++)
				{
					//printf("(%d, %d, %d)... ", x, y, z);
					int cidJ = 0, cidI = 1, npts = 0;
					double residuals = 0.0, R[9], rt[6] = { orgx + 0.005*x, orgy + 0.005*y, orgz + 0.005*z, allCalibInfo[1].rt[3], allCalibInfo[1].rt[4], allCalibInfo[1].rt[5] };

					//for (int cidJ = 0; cidJ < (int)allImgs.size(); cidJ++)
					{
						int cidJ = 0;
						bool once = true;
						int nsteps = 1000, count = 0, per = 5;

						for (int ii = 0; ii < (int)invDAll[cidJ].size(); ii++)
						{
#pragma omp critical
							{
								count++;
								if (100.0*count / (int)invDAll[cidJ].size() > per)
								{									
									printf("%d%%.. ", per);
									per += 5;
								}
							}

							sprintf(Fname, "C:/temp/costD_%d.txt", pyrID);  FILE *fp = fopen(Fname, "w+");
							double minCost = 9e9, bestInvD = 0;
//#pragma omp parallel for schedule(dynamic,1)
							for (int kk = 1; kk < nsteps; kk++)
							{
								residuals = 0;
								double invD = 0.001*kk;
								for (int cidI = 0; cidI < (int)allImgs.size(); cidI++)
								{
									if (cidI == cidJ)
										continue;

									int width = allImgs[cidI].imgPyr[pyrID].cols, height = allImgs[cidI].imgPyr[pyrID].rows;
									int i = indIAll[cidJ][ii], j = indJAll[cidJ][ii];
									double xycnRef[3] = { 0, 0, 0 }, ij[3] = { i, j, 1 };
									mat_mul(allCalibInfo[cidJ].activeinvK, ij, xycnRef, 3, 3, 1);

									ceres::DynamicAutoDiffCostFunction<DepthImgWarping, 4> *cost_function = new ceres::DynamicAutoDiffCostFunction < DepthImgWarping, 4 >
										(new DepthImgWarping(Point2i(i, j), xycnRef, allImgs[cidJ].imgPyr[pyrID].data, allImgs[cidI].imgPyr[pyrID].data, allCalibInfo[cidI].activeIntrinsic, Point2i(width, height), allImgs[cidI].nchannels, 1.0 / colorSigma, ii, cidI));

									cost_function->SetNumResiduals(1);

									vector<double*> parameter_blocks;
									parameter_blocks.push_back(&invD), cost_function->AddParameterBlock(1);
									parameter_blocks.push_back(allCalibInfo[cidJ].rt), cost_function->AddParameterBlock(6);
									parameter_blocks.push_back(rt), cost_function->AddParameterBlock(6);
									parameter_blocks.push_back(&allCalibInfo[cidI].photometric[2 * cidJ]), cost_function->AddParameterBlock(2);

									double resi[3];
									cost_function->Evaluate(&parameter_blocks[0], resi, NULL);
									//for (int jj = 0; jj < nchannels; jj++)
									//	residuals += resi[jj] * resi[jj];
									residuals += resi[0] * resi[0];
									delete[]cost_function;
									//npts += nchannels;
								
								}
								if (minCost > residuals)
									minCost = residuals, bestInvD = invD;
								fprintf(fp, "%.4f %.16e\n", invD, residuals);
							}		fclose(fp);
							invDAll[cidJ][ii] = bestInvD;
						}
					}
					//#pragma omp critical
				}
			}
		}

		return 0;
	}

	//dump to Ceres
	ceres::Problem problem;

	//Data term
	ceres::LossFunction *ColorLoss = new ceres::HuberLoss(5);
	ceres::LossFunction* ScaleColorLoss = new ceres::ScaledLoss(ColorLoss, dataWeight, ceres::TAKE_OWNERSHIP);
	//for (int cidJ = 0; cidJ < (int)allImgs.size(); cidJ++)
	{
		int cidJ = 0;
		bool once = true;
		for (int cidI = 0; cidI < (int)allImgs.size(); cidI++)
		{
			if (cidI == cidJ)
				continue;
			int width = allImgs[cidI].imgPyr[pyrID].cols, height = allImgs[cidI].imgPyr[pyrID].rows;

			for (int ii = 0; ii < (int)invDAll[cidJ].size(); ii++)
			{
				int i = indIAll[cidJ][ii], j = indJAll[cidJ][ii];
				double xycnRef[3] = { 0, 0, 0 }, ij[3] = { i, j, 1 };
				mat_mul(allCalibInfo[cidJ].activeinvK, ij, xycnRef, 3, 3, 1);

				if (SmallAngle == 1)
				{
					ceres::DynamicAutoDiffCostFunction<DepthImgWarpingSmall, 4> *cost_function = new ceres::DynamicAutoDiffCostFunction < DepthImgWarpingSmall, 4 >
						(new DepthImgWarpingSmall(Point2i(i, j), xycnRef, allImgs[cidJ].imgPyr[pyrID].data, allImgs[cidI].imgPyr[pyrID].data, allCalibInfo[cidI].activeIntrinsic, Point2i(width, height), allImgs[cidI].nchannels, 1.0 / colorSigma, ii, cidI));

					cost_function->SetNumResiduals(1);

					vector<double*> parameter_blocks;
					parameter_blocks.push_back(&invDAll[cidJ][ii]), cost_function->AddParameterBlock(1);
					parameter_blocks.push_back(allCalibInfo[cidJ].rt), cost_function->AddParameterBlock(6);
					parameter_blocks.push_back(allCalibInfo[cidI].rt), cost_function->AddParameterBlock(6);
					parameter_blocks.push_back(&allCalibInfo[cidI].photometric[2 * cidJ]), cost_function->AddParameterBlock(2);
					problem.AddResidualBlock(cost_function, ScaleColorLoss, parameter_blocks);

					if (fixDepth == 1)
						problem.SetParameterBlockConstant(parameter_blocks[0]); //depth
					if (cidJ == 0)
						problem.SetParameterBlockConstant(parameter_blocks[1]); //pose Ref
					if (fixPose == 1)
						problem.SetParameterBlockConstant(parameter_blocks[2]); //pose nonRef
					problem.SetParameterBlockConstant(parameter_blocks[3]); //photometric
				}
				else
				{
					//ceres::DynamicNumericDiffCostFunction<DepthImgWarping, ceres::CENTRAL> *cost_function = new ceres::DynamicNumericDiffCostFunction<DepthImgWarping, ceres::CENTRAL>
					//	(new DepthImgWarping(Point2i(i, j), xycnRef, allImgs[cidJ].imgPyr[pyrID].data, allImgs[cidI].imgPyr[pyrID].data, allCalibInfo[cidI].intrinsic, Point2i(allImgs[cidI].width, allImgs[cidI].height), allImgs[cidI].nchannels, 1.0 / colorSigma, ii, cidI));
					ceres::DynamicAutoDiffCostFunction<DepthImgWarping, 4> *cost_function = new ceres::DynamicAutoDiffCostFunction < DepthImgWarping, 4 >
						(new DepthImgWarping(Point2i(i, j), xycnRef, allImgs[cidJ].imgPyr[pyrID].data, allImgs[cidI].imgPyr[pyrID].data, allCalibInfo[cidI].activeIntrinsic, Point2i(width, height), allImgs[cidI].nchannels, 1.0 / colorSigma, ii, cidI));

					cost_function->SetNumResiduals(1);

					vector<double*> parameter_blocks;
					parameter_blocks.push_back(&invDAll[cidJ][ii]), cost_function->AddParameterBlock(1);
					parameter_blocks.push_back(allCalibInfo[cidJ].rt), cost_function->AddParameterBlock(6);
					parameter_blocks.push_back(allCalibInfo[cidI].rt), cost_function->AddParameterBlock(6);
					parameter_blocks.push_back(&allCalibInfo[cidI].photometric[2 * cidJ]), cost_function->AddParameterBlock(2);
					problem.AddResidualBlock(cost_function, ScaleColorLoss, parameter_blocks);

					if (fixDepth == 1)
						problem.SetParameterBlockConstant(parameter_blocks[0]); //depth
					if (cidJ == 0)
						problem.SetParameterBlockConstant(parameter_blocks[1]); //pose Ref
					if (fixPose == 1)
						problem.SetParameterBlockConstant(parameter_blocks[2]); //pose nonRef
					problem.SetParameterBlockConstant(parameter_blocks[3]); //photometric
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
			for (int ii = 0; ii < (int)NNAll[cid].size(); ii++)
			{
				for (int jj = 1; jj < (int)nNNAll[cid][ii]; jj++)
				{
					int iref = indIAll[cid][NNAll[cid][ii][0]], jref = indJAll[cid][NNAll[cid][ii][0]],
						i = indIAll[cid][NNAll[cid][ii][jj]], j = indJAll[cid][NNAll[cid][ii][jj]], width = allImgs[cid].imgPyr[pyrID].cols;

					double colorDif = 0;
					for (int kk = 0; kk < nchannels; kk++)
						colorDif += (double)((int)allImgs[cid].imgPyr[pyrID].data[kk + (iref + jref* width)*nchannels] - (int)allImgs[cid].imgPyr[pyrID].data[kk + (i + j* width)*nchannels]);
					colorDif /= nchannels;
					double edgePreservingWeight = std::exp(-abs(colorDif) / alignmentParas.colorSigma);

					ceres::LossFunction *RegIntraLoss = NULL;
					ceres::LossFunction *ScaleRegIntraLoss = new ceres::ScaledLoss(RegIntraLoss, regIntraWeight*edgePreservingWeight, ceres::TAKE_OWNERSHIP);

					ceres::CostFunction* cost_function = IntraDepthRegularize::Create(1.0 / depthSigma);
					problem.AddResidualBlock(cost_function, ScaleRegIntraLoss, &invDAll[cid][NNAll[cid][ii][0]], &invDAll[cid][NNAll[cid][ii][jj]]);
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
					resImg[ii] = (uchar)255, synthImg[ii] = (uchar)255;

				int widthI = allImgs[cidI].imgPyr[pyrID].cols, heightI = allImgs[cidI].imgPyr[pyrID].rows;
				for (int ii = 0; ii < (int)invDAll[cidJ].size(); ii++)
				{
					//back-project ref depth to 3D
					int i = indIAll[cidJ][ii], j = indJAll[cidJ][ii];
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

						resImg[indIAll[cidJ][ii] + indJAll[cidJ][ii] * widthJ] = (uchar)(int)(10.0 * abs(dif + 0.5)); //magnify 10x
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

							resImg[indIAll[cidJ][ii] + indJAll[cidJ][ii] * widthJ + kk*lengthJ] = (uchar)(int)(10.0 * abs(dif + 0.5)); //magnify 10x
							synthImg[indIAll[cidJ][ii] + indJAll[cidJ][ii] * widthJ + kk*lengthJ] = (uchar)(int)(fcolor + 0.5);
							residuals += dif*dif;
						}
						npts += 3;
					}
				}
				sprintf(Fname, "%s/R_%d_%d_%d_%d.png", Path, cidJ, cidI, pyrID, iter), WriteGridToImage(Fname, resImg, widthJ, heightJ, nchannels);
				sprintf(Fname, "%s/S_%d_%d_%d_%d.png", Path, cidJ, cidI, pyrID, iter), WriteGridToImage(Fname, synthImg, widthJ, heightJ, nchannels);

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
	ceres::Solve(options, &problem, &summary);
	if (verbose == 1)
		std::cout << "\n" << summary.BriefReport() << "\n";

	//Saving data
	for (int cid = 0; cid < (int)allImgs.size(); cid++)
	{
		int width = allImgs[cid].imgPyr[pyrID].cols;
		for (int ii = 0; ii < (int)invDAll[cid].size(); ii++)
			allImgs[cid].depthPyr[pyrID][indIAll[cid][ii] + indJAll[cid][ii] * width] = 1.0 / invDAll[cid][ii];
	}

	for (int ii = 0; ii < (int)validPixelsAll.size(); ii++)
		delete[] validPixelsAll[ii];
	for (int ii = 0; ii < (int)sub2indAll.size(); ii++)
		delete[] sub2indAll[ii];
	delete[]indIAll, delete[]indJAll, delete[]invDAll;
	for (int ii = 0; ii < (int)dxAll.size(); ii++)
		delete[]dxAll[ii], delete[]dyAll[ii];

	return summary.final_cost;
}
int DirectAlignment(char *Path, vector<ImgData> &allImgs, vector<CameraData> &allCalibInfo, int smallAngle = 0)
{
	printf("Current Path: %s\n", Path);

	int nscales = 5, //actually 6 scales = 5 down sampled images + org image
		innerIter = 20;
	double dataWeight = 0.9, regIntraWeight = 0.1, regInterWeight = 1.0 - dataWeight - regIntraWeight;
	double colorSigma = 1.0, depthSigma = 0.01; //expected std of variables (grayscale, system unit);
	double ImGradientThesh2 = 1;
	DirectAlignPara alignmentParas(dataWeight, regIntraWeight, regInterWeight, colorSigma, depthSigma, ImGradientThesh2);

	for (int cid = 0; cid < (int)allImgs.size(); cid++)
	{
		for (int pyrID = 0; pyrID < nscales + 1; pyrID++)
			allImgs[cid].scaleFactor.push_back(1.0 / pow(2, pyrID));

		GaussianBlur(allImgs[cid].color, allImgs[cid].color, Size(5, 5), 0.707);
		buildPyramid(allImgs[cid].color, allImgs[cid].imgPyr, nscales);

		BuildDepthPyramid(allImgs[cid].depth, allImgs[cid].depthPyr, allImgs[cid].width, allImgs[cid].height, nscales);
	}

	//DirectAlignmentToFile(Path, alignmentParas, allImgs, allCalibInfo, 0);
	//exit(0);
	/*ReadGridBinary("D:/Micro/D_0_2.dat", allImgs[0].depthPyr[2], allImgs[0].imgPyr[2].cols, allImgs[0].imgPyr[2].rows);
	for (int cid = 0; cid < (int)allImgs.size(); cid++)
	UpsamleDepth(allImgs[cid].depthPyr[2], allImgs[cid].depthPyr[2 - 1], allImgs[cid].imgPyr[2].cols, allImgs[cid].imgPyr[2].rows, allImgs[cid].imgPyr[2 - 1].cols, allImgs[cid].imgPyr[2 - 1].rows);
	FILE *fp = fopen("D:/Micro/p2.txt", "r");
	for (int cid = 0; cid < (int)allImgs.size(); cid++)
	for (int ii = 0; ii < 6; ii++)
	fscanf(fp, "%lf ", &allCalibInfo[cid].rt[ii]);
	fclose(fp); */

	double startTime = omp_get_wtime();
	for (int pyrID = nscales; pyrID >=0; pyrID--)
	{
		printf("\n\n@ level: %d\n", pyrID);
		/*for (int cid = 0; cid < (int)allImgs.size(); cid++)
		{
			char Fname[512]; sprintf(Fname, "%s/%d_%d.png", Path, cid, pyrID);
			imwrite(Fname, allImgs[cid].imgPyr[pyrID]);
		}*/
		/*double startTimeI = omp_get_wtime();
		double percentage, innerCost1, innerCost2, pinnerCost1 = 1e16, pinnerCost2 = 1e16;
		for (int innerID = 0; innerID < innerIter; innerID++) //Pose is mostly well estimated even with nosiy depth. Depth is improved with good pose. -->Alternating optim, which is extremely helpful in practice
		{
		printf("@(Depth) Iter %d: ", innerID); //at low res, calibration error is not servere, let's optim depth first
		innerCost1 = DirectAlignmentPyr(Path, alignmentParas, allImgs, allCalibInfo, 1, 0, pyrID, smallAngle);

		printf("\n@(Pose) Iter %d: ", innerID);
		innerCost2 = DirectAlignmentPyr(Path, alignmentParas, allImgs, allCalibInfo, 0, 1, pyrID, smallAngle);

		percentage = abs(innerCost1 + innerCost2 - pinnerCost1 - pinnerCost2) / (pinnerCost1 + pinnerCost2);
		printf("Change: %.3e\n", percentage);
		if (percentage < 1.0e-3)
		break;
		pinnerCost1 = innerCost1, pinnerCost2 = innerCost2;
		}
		printf("Total time: %.2f\n", omp_get_wtime() - startTimeI);

		if (pyrID == 0)*/
		DirectAlignmentPyr(Path, alignmentParas, allImgs, allCalibInfo, 1, 0, pyrID, smallAngle, 0);
		//DirectAlignmentToFile(Path, alignmentParas, allImgs, allCalibInfo, pyrID);

		if (pyrID != 0)
			for (int cid = 0; cid < (int)allImgs.size(); cid++)
				UpsamleDepth(allImgs[cid].depthPyr[pyrID], allImgs[cid].depthPyr[pyrID - 1], allImgs[cid].imgPyr[pyrID].cols, allImgs[cid].imgPyr[pyrID].rows, allImgs[cid].imgPyr[pyrID - 1].cols, allImgs[cid].imgPyr[pyrID - 1].rows);
	}
	printf("\n\nTotal time: %.2f\n", omp_get_wtime() - startTime);

	for (int cid = 0; cid < (int)allImgs.size(); cid++)
		for (int pyrID = 0; pyrID < nscales + 1; pyrID++)
			delete[]allImgs[cid].depthPyr[pyrID];

	return 1;
}

struct FlowBASmall {
	FlowBASmall(Point2i uv, double *RefRayDir_, double *RefCC, Point2f *Flow, double *intrinsic, Point2i imgSize, double isigma, int eleID, int imgID) :
		uv(uv), RefCC(RefCC), Flow(Flow), intrinsic(intrinsic), imgSize(imgSize), isigma(isigma), eleID(eleID), imgID(imgID)
	{
		boundary = max(imgSize.x, imgSize.y) / 50;
		for (int ii = 0; ii < 3; ii++)
			RefRayDir[ii] = RefRayDir_[ii];//xycn changes for every pixel. Pass by ref does not work
	}

	template <typename T>	bool operator()(const T* const rt, const T* const idepth, T* residuals) 	const
	{
		T XYZ[3], d = idepth[0];
		for (int ii = 0; ii < 3; ii++)
			XYZ[ii] = RefRayDir[ii] / d + RefCC[ii]; //X = r*d+c

		//project to other views
		T tp[3], xcn, ycn, tu, tv;
		tp[0] = XYZ[0] - rt[2] * XYZ[1] + rt[1] * XYZ[2] + rt[3];
		tp[1] = rt[2] * XYZ[0] + XYZ[1] - rt[0] * XYZ[2] + rt[4];
		tp[2] = -rt[1] * XYZ[0] + rt[0] * XYZ[1] + XYZ[2] + rt[5];
		xcn = tp[0] / tp[2], ycn = tp[1] / tp[2];

		tu = (T)intrinsic[0] * xcn + (T)intrinsic[2] * ycn + (T)intrinsic[3];
		tv = (T)intrinsic[1] * ycn + (T)intrinsic[4];

		residuals[0] = (T)isigma*((T)Flow[uv.x + uv.y*imgSize.x].x - tu);
		residuals[1] = (T)isigma*((T)Flow[uv.x + uv.y*imgSize.x].y - tv);

		return true;
	}
	static ceres::CostFunction* Create(Point2i uv, double *RefRayDir, double *RefCC, Point2f *Flow, double *intrinsic, Point2i imgSize, double isigma, int eleID, int imgID)
	{
		return (new ceres::AutoDiffCostFunction<FlowBASmall, 2, 6, 1>(new FlowBASmall(uv, RefRayDir, RefCC, Flow, intrinsic, imgSize, isigma, eleID, imgID)));
	}

	Point2f *Flow;
	Point2i uv, imgSize;
	double RefRayDir[3], *RefCC, *intrinsic, isigma;
	int  eleID, imgID, boundary;
};
double FlowBasedBundleAdjustment(char *Path, DirectAlignPara &alignmentParas, vector<ImgData> &allImgs, vector<Point2f*> &allFlows, vector<CameraData> &allCalibInfo, int fixPose, int fixDepth, int pyrID, int SmallAngle = 0, int verbose = 0)
{
	char Fname[512];

	double dataWeight = alignmentParas.dataWeight, regIntraWeight = alignmentParas.regIntraWeight, regInterWeight = alignmentParas.regInterWeight;
	double reProjectionSigma = alignmentParas.reProjectionSigma, depthSigma = alignmentParas.depthSigma; //expected std of variables (grayscale, mm);
	int nchannels = allImgs[0].nchannels;

	//find texture region in the refImg and store in the vector
	vector<float *> dxAll, dyAll;
	//for (int cid = 0; cid < (int)allImgs.size(); cid++)
	{
		int cid = 0;
		int width = allImgs[cid].imgPyr[pyrID].cols, height = allImgs[cid].imgPyr[pyrID].rows, boundary = width / 50;
		float *dx = new float[width*height], *dy = new float[width*height];
		for (int jj = boundary; jj < height - boundary; jj++)
		{
			for (int ii = boundary; ii < width - boundary; ii++)
			{
				//calculate first order image derivatives: using 1 channel should be enough
				dx[ii + jj*width] = (float)(int)allImgs[cid].imgPyr[pyrID].data[(ii + 1)*nchannels + jj*nchannels*width] - (float)(int)allImgs[cid].imgPyr[pyrID].data[(ii - 1)*nchannels + jj*nchannels*width]; //1, 0, -1
				dy[ii + jj*width] = (float)(int)allImgs[cid].imgPyr[pyrID].data[ii *nchannels + (jj + 1)*nchannels*width] - (float)(int)allImgs[cid].imgPyr[pyrID].data[ii *nchannels + (jj - 1)*nchannels*width]; //1, 0, -1
			}
		}
		dxAll.push_back(dx), dyAll.push_back(dy);
	}

	//Getting valid pixels (high texture && has depth &&good conf)
	float mag2Thresh = alignmentParas.gradientThresh2; //suprisingly, using more relaxed thresholding better convergence than aggressive one. 
	vector<bool *> validPixelsAll;
	vector<int*> sub2indAll;
	vector<int> *indIAll = new vector<int>[allImgs.size()], *indJAll = new vector<int>[allImgs.size()];
	vector<double> *invDAll = new vector<double>[allImgs.size()];
	//for (int cid = 0; cid < (int)allImgs.size(); cid++)
	{
		int cid = 0;
		int width = allImgs[cid].imgPyr[pyrID].cols, height = allImgs[cid].imgPyr[pyrID].rows, boundary = width / 50;
		int *sub2ind = new int[width*height];
		bool *validPixels = new bool[width*height];
		for (int ii = 0; ii < width*height; ii++)
			validPixels[ii] = false, sub2ind[ii] = -1;

		invDAll[cid].reserve(width*height);
		indIAll[cid].reserve(width*height), indJAll[cid].reserve(width*height);

		for (int jj = boundary; jj < height - boundary; jj++)
		{
			for (int ii = boundary; ii < width - boundary; ii++)
			{
				float mag2 = pow(dxAll[cid][ii + jj*width], 2) + pow(dyAll[cid][ii + jj*width], 2);
				if (IsNumber(allImgs[cid].depthPyr[pyrID][ii + jj*width]) == 1 && mag2 > mag2Thresh && abs(allImgs[cid].depthPyr[pyrID][ii + jj*width]) >0.0)
				{
					/*bool notgood = false;
					for (int kk = 0; !notgood && kk < (int)allFlows.size(); kk++)
					if (abs(allFlows[kk][ii + jj*width].x) < 1.e-6 || abs(allFlows[kk][ii + jj*width].y) < 1.e-6)
					notgood = true;
					if (notgood)
					continue;*/

					indIAll[cid].push_back(ii), indJAll[cid].push_back(jj), invDAll[cid].push_back(1.0 / (allImgs[cid].depthPyr[pyrID][ii + jj*width] + 1.0e-16)),
						validPixels[ii + jj*width] = true, sub2ind[ii + jj*width] = (int)indIAll[cid].size() - 1;
				}
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

	//dump to Ceres
	ceres::Problem problem;

	//Data term
	ceres::LossFunction *ColorLoss = new ceres::TukeyLoss(10.0);
	ceres::LossFunction* ScaleReProjectionLoss = new ceres::ScaledLoss(ColorLoss, dataWeight, ceres::TAKE_OWNERSHIP);
	bool once = true;
	GetKFromIntrinsic(allCalibInfo[0].K, allCalibInfo[0].intrinsic);
	GetiK(allCalibInfo[0].invK, allCalibInfo[0].K);
	getRfromr(allCalibInfo[0].rt, allCalibInfo[0].R);
	GetCfromT(allCalibInfo[0].R, allCalibInfo[0].rt + 3, allCalibInfo[0].C);
	for (int cid = 1; cid < (int)allFlows.size() + 1; cid++)
	{
		int width = allImgs[cid].imgPyr[pyrID].cols, height = allImgs[cid].imgPyr[pyrID].rows;
		for (int ii = 0; ii < (int)invDAll[0].size(); ii++)
		{
			int i = indIAll[0][ii], j = indJAll[0][ii];
			double rayDirRef[3] = { 0, 0, 0 }, ij[3] = { i, j, 1 };
			getRayDir(rayDirRef, allCalibInfo[0].invK, allCalibInfo[0].R, ij);

			ceres::CostFunction* cost_function = FlowBASmall::Create(Point2i(i, j), rayDirRef, allCalibInfo[0].C, allFlows[cid - 1], allCalibInfo[cid].activeIntrinsic, Point2i(width, height), 1.0 / reProjectionSigma, ii, cid);
			problem.AddResidualBlock(cost_function, ScaleReProjectionLoss, allCalibInfo[cid].rt, &invDAll[0][ii]);
		}
	}

	if (fixDepth == 0)
	{
		//Intra depth regularization term
		int cid = 0;
		for (int ii = 0; ii < (int)NNAll[cid].size(); ii++)
		{
			for (int jj = 1; jj < (int)nNNAll[cid][ii]; jj++)
			{
				int iref = indIAll[cid][NNAll[cid][ii][0]], jref = indJAll[cid][NNAll[cid][ii][0]],
					i = indIAll[cid][NNAll[cid][ii][jj]], j = indJAll[cid][NNAll[cid][ii][jj]], width = allImgs[cid].imgPyr[pyrID].cols;

				double colorDif = 0;
				for (int kk = 0; kk < nchannels; kk++)
					colorDif += (double)((int)allImgs[cid].imgPyr[pyrID].data[kk + (iref + jref* width)*nchannels] - (int)allImgs[cid].imgPyr[pyrID].data[kk + (i + j* width)*nchannels]);
				colorDif /= nchannels;
				double edgePreservingWeight = std::exp(-abs(colorDif) / alignmentParas.colorSigma);

				ceres::LossFunction *RegIntraLoss = NULL;
				ceres::LossFunction *ScaleRegIntraLoss = new ceres::ScaledLoss(RegIntraLoss, regIntraWeight*edgePreservingWeight, ceres::TAKE_OWNERSHIP);

				ceres::CostFunction* cost_function = IntraDepthRegularize::Create(1.0 / depthSigma);
				problem.AddResidualBlock(cost_function, ScaleRegIntraLoss, &invDAll[0][NNAll[cid][ii][0]], &invDAll[0][NNAll[cid][ii][jj]]);
			}
		}
	}


	ceres::Solver::Options options;
	options.num_threads = omp_get_max_threads(); //jacobian eval
	options.num_linear_solver_threads = omp_get_max_threads(); //linear solver
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.linear_solver_type = ceres::CGNR;
	options.preconditioner_type = ceres::JACOBI;
	options.use_inner_iterations = true;
	options.use_nonmonotonic_steps = false;
	options.max_num_iterations = pyrID > 1 ? 50 : 200;
	options.parameter_tolerance = 1.0e-9;
	options.minimizer_progress_to_stdout = true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << "\n" << summary.FullReport() << "\n";

	//Saving data
	//for (int cid = 0; cid < (int)allImgs.size(); cid++)
	{
		int cid = 0;
		int width = allImgs[cid].imgPyr[pyrID].cols;
		for (int ii = 0; ii < (int)invDAll[cid].size(); ii++)
			allImgs[cid].depthPyr[pyrID][indIAll[cid][ii] + indJAll[cid][ii] * width] = 1.0 / invDAll[cid][ii];
	}

	sprintf(Fname, "%s/pose.txt", Path); FILE *fp = fopen(Fname, "a+");
	for (int cid = 0; cid < (int)allImgs.size(); cid++)
		fprintf(fp, "Cid: %d  PyrID: %d %.16f %.16f %.16f %.16f %.16f %.16f %\n", cid, pyrID,
		allCalibInfo[cid].rt[0], allCalibInfo[cid].rt[1], allCalibInfo[cid].rt[2], allCalibInfo[cid].rt[3], allCalibInfo[cid].rt[4], allCalibInfo[cid].rt[5]);
	fclose(fp);

	//for (int cid = 0; cid < (int)allImgs.size(); cid++)
	{
		int cid = 0;
		int width = allImgs[cid].imgPyr[pyrID].cols, height = allImgs[cid].imgPyr[pyrID].rows, lengthJ = width*height;
		float *depthMap = new float[width*height];
		for (int ii = 0; ii < width*height; ii++)
			depthMap[ii] = 0.f;
		for (int ii = 0; ii < (int)invDAll[cid].size(); ii++)
		{
			double ij[3] = { indIAll[cid][ii], indJAll[cid][ii], 1 };
			depthMap[indIAll[cid][ii] + indJAll[cid][ii] * width] = 1.0 / invDAll[cid][ii];
		}
		sprintf(Fname, "%s/D_%d_%d.dat", Path, cid, pyrID), WriteGridBinary(Fname, depthMap, width, height, 1);

		if (nchannels == 1)
			sprintf(Fname, "%s/3d_%d_%d.xyz", Path, cid, pyrID);
		else
			sprintf(Fname, "%s/3d_%d_%d.txt", Path, cid, pyrID);
		FILE *fp = fopen(Fname, "w+");
		for (int ii = 0; ii < (int)invDAll[cid].size(); ii++)
		{
			int i = indIAll[cid][ii], j = indJAll[cid][ii];
			double XYZ[3], rayDir[3], ij[3] = { i, j, 1 };
			getRfromr(allCalibInfo[cid].rt, allCalibInfo[cid].R);
			GetCfromT(allCalibInfo[cid].R, allCalibInfo[cid].rt + 3, allCalibInfo[cid].C);

			getRayDir(rayDir, allCalibInfo[cid].activeinvK, allCalibInfo[cid].R, ij);
			for (int kk = 0; kk < 3; kk++)
				XYZ[kk] = rayDir[kk] / invDAll[cid][ii] + allCalibInfo[cid].C[kk];
			if (nchannels == 1)
				fprintf(fp, "%.8e %.8e %.8e\n", XYZ[0], XYZ[1], XYZ[2]);
			else
				fprintf(fp, "%.8e %.8e %.8e %d %d %d\n", XYZ[0], XYZ[1], XYZ[2],
				(int)allImgs[cid].imgPyr[pyrID].data[(i + j*width)*nchannels], (int)allImgs[cid].imgPyr[pyrID].data[(i + j*width)*nchannels + 1], (int)allImgs[cid].imgPyr[pyrID].data[(i + j*width)*nchannels + 2]);
		}
		fclose(fp);
	}
	printf("Done\n");

	for (int ii = 0; ii < (int)validPixelsAll.size(); ii++)
		delete[] validPixelsAll[ii];
	for (int ii = 0; ii < (int)sub2indAll.size(); ii++)
		delete[] sub2indAll[ii];
	delete[]indIAll, delete[]indJAll, delete[]invDAll;
	for (int ii = 0; ii < (int)dxAll.size(); ii++)
		delete[]dxAll[ii], delete[]dyAll[ii];

	return summary.final_cost;
}
int main(int argc, char** argv)
{
	//srand(time(NULL));
	srand(1);
	char Fname[512];
	int mode = 3;
	//4: init with spline from other key frames; 
	//3:good init but add noise on Sudipta's format; 
	//2: synth; 
	//1: low based; 
	//0: middlebury

	if (mode == 4) //no init
	{
		int nchannels = 3;
		char Path[] = "C:/Data/Micro";
		vector<ImgData> allImgs;
		vector<CameraData> allCalibInfo;

		vector<int> camIDs; camIDs.push_back(1765);
		for (int ii = 1; ii <= 15; ii += 1)
			camIDs.push_back(1765 + ii), camIDs.push_back(1765 - ii);

		for (int ii = 0; ii < camIDs.size(); ii++)
		{
			ImgData imgI;
			sprintf(Fname, "%s/0/%d.png", Path, camIDs[ii]);
			imgI.color = imread(Fname, nchannels == 1 ? 0 : 1);
			imgI.width = imgI.color.cols, imgI.height = imgI.color.rows, imgI.nchannels = nchannels;
			allImgs.push_back(imgI);
		}

		VideoData vInfo;
		ReadVideoDataI(Path, vInfo, 0, 0, 1800);
		for (int ii = 0; ii < (int)camIDs.size(); ii++)
			allCalibInfo.push_back(vInfo.VideoInfo[camIDs[ii]]);

		for (int cid = 0; cid < (int)allImgs.size(); cid++)
			allImgs[cid].depth = new float[allImgs[0].width*allImgs[0].height];

		for (int cid = 0; cid < (int)allImgs.size(); cid++)
			for (int jj = 0; jj < allImgs[cid].height; jj++)
				for (int ii = 0; ii < allImgs[cid].width; ii++)
					allImgs[cid].depth[ii + jj*allImgs[cid].width] = 1.0 / (0.9 * rand() / RAND_MAX + 0.1); //invd: [0.1, 1.0]

		DirectAlignment(Path, allImgs, allCalibInfo, 0);
	}
	if (mode == 3) //good init but add noise on Sudipta's format
	{
		int nchannels = 3;
		char Path[] = "D:/Data/Micro3";
		vector<ImgData> allImgs;
		vector<CameraData> allCalibInfo;

		int SynthID = 0;
		Corpus CorpusData;
		ReadSynthFile(Path, CorpusData, SynthID);

		vector<int> camIDs; camIDs.push_back(45);
		for (int ii = 1; ii <= 15; ii++)
			camIDs.push_back(45 + ii), camIDs.push_back(45 - ii);

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

		ReadSudiptaDepth(Path, allImgs[0], camIDs[0]);
		ConvertFrontoDepth2LineOfSightDepth(allImgs[0], allCalibInfo[0]);
		for (int cid = 1; cid < (int)allImgs.size(); cid++)
			allImgs[cid].depth = new float[allImgs[0].width*allImgs[0].height];

		//for (int cid = 0; cid < (int)allImgs.size(); cid++)
		{
			int cid = 0;
			for (int jj = 0; jj < allImgs[cid].height; jj++)
				for (int ii = 0; ii < allImgs[cid].width; ii++)
					if (abs(allImgs[cid].depth[ii + jj*allImgs[cid].width]) >0.001)
						allImgs[cid].depth[ii + jj*allImgs[cid].width] += 9.0*rand() / RAND_MAX - 4.5;
		}
		DirectAlignment(Path, allImgs, allCalibInfo);
	}
	if (mode == 2) //synth
	{
		char Path[] = "D:/Data/DirectBA";

		int nchannels = 3;
		vector<int> selectedCamID; selectedCamID.push_back(8), selectedCamID.push_back(4), selectedCamID.push_back(12), selectedCamID.push_back(0), selectedCamID.push_back(16);

		Corpus CorpusData;
		sprintf(Fname, "%s/Corpus/calibInfo.txt", Path);
		readCalibInfo(Fname, CorpusData);

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

			sprintf(Fname, "%s/GT/CD_%d_1.ijz", Path, selectedCamID[ii]);
			imgI.depth = new float[imgI.width * imgI.height];
			ReadGridBinary(Fname, imgI.depth, imgI.width, imgI.height);

			allImgs.push_back(imgI);
		}

		for (int cid = 1; cid < (int)allImgs.size(); cid++)
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
					if (allImgs[cid].depth[ii + jj*allImgs[cid].width] < -0.001)
						;// allImgs[cid].depth[ii + jj*allImgs[cid].width] += min(max(gaussian_noise(0, 0.5), -1.5), 1.5);
				}
		}

		DirectAlignment(Path, allImgs, allCalibInfo);
	}
	if (mode == 1) //flow based
	{
		int nchannels = 3;
		char Path[] = "D:/Data/Micro3";
		vector<ImgData> allImgs;
		vector<Point2f*> allFlows;
		vector<CameraData> allCalibInfo;

		vector<int> camIDs; camIDs.push_back(45);
		for (int ii = 1; ii < 16; ii++)
			camIDs.push_back(45 + ii), camIDs.push_back(45 - ii);

		for (int ii = 0; ii < camIDs.size(); ii++)
		{
			ImgData imgI;
			sprintf(Fname, "%s/Recon/urd-%03d.png", Path, camIDs[ii]);
			imgI.color = imread(Fname, nchannels == 1 ? 0 : 1);
			imgI.width = imgI.color.cols, imgI.height = imgI.color.rows, imgI.nchannels = nchannels;
			allImgs.push_back(imgI);
		}

		for (int ii = 1; ii < camIDs.size(); ii++)
		{
			char Fname1[512], Fname2[512];
			sprintf(Fname1, "%s/Flow/X_%d_%d.dat", Path, camIDs[0], camIDs[ii]);
			sprintf(Fname2, "%s/Flow/Y_%d_%d.dat", Path, camIDs[0], camIDs[ii]);
			Point2f *Flow = new Point2f[allImgs[0].width* allImgs[0].height];
			ReadFlowDataBinary(Fname1, Fname2, Flow, allImgs[0].width, allImgs[0].height);
			allFlows.push_back(Flow);
		}

		/*for (int ii = 1; ii < camIDs.size(); ii++)
		{
		Point2f *Flow = new Point2f[allImgs[0].width* allImgs[0].height];
		for (int jj = 0; jj < 1920 * 1080; jj++)
		Flow[jj] = Point2f(0, 0);
		allFlows.push_back(Flow);
		}

		int nimages, i, j;  float du, dv;
		FILE *fp = fopen("C:/temp/x.txt", "r");
		while (fscanf(fp, "%d %d %d ", &nimages, &i, &j) != EOF)
		{
		for (int ii = 1; ii < nimages; ii++)
		{
		fscanf(fp, "%f %f ", &du, &dv);
		allFlows[ii - 1][i + j * 1920].x = du;
		allFlows[ii - 1][i + j * 1920].y = dv;
		}
		}
		fclose(fp);*/

		for (int ii = 0; ii < camIDs.size(); ii++)
		{
			CameraData cam; cam.intrinsic[0] = 1958.0, cam.intrinsic[1] = 1958.0, cam.intrinsic[2] = 0, cam.intrinsic[3] = 1920 / 2, cam.intrinsic[4] = 1080 / 2;
			for (int jj = 0; jj < 6; jj++)
				cam.rt[jj] = 0;

			allCalibInfo.push_back(cam);
		}

		for (int cid = 0; cid < (int)allImgs.size(); cid++)
		{
			allImgs[cid].depth = new float[allImgs[0].width*allImgs[0].height];
			for (int jj = 0; jj < allImgs[cid].height; jj++)
				for (int ii = 0; ii < allImgs[cid].width; ii++)
					allImgs[cid].depth[ii + jj*allImgs[cid].width] = 1.0 / (0.9 * rand() / RAND_MAX + 0.1); //invd: [0.1, 1.0]
		}

		double dataWeight = 0.6, regIntraWeight = 0.4, regInterWeight = 1.0 - dataWeight - regIntraWeight;
		double colorSigma = 5.0, depthSigma = 0.02, reProjectionSigma = 1.0; //expected std of variables (grayscale, system unit);
		double ImGradientThesh2 = 500;
		DirectAlignPara alignmentParas(dataWeight, regIntraWeight, regInterWeight, colorSigma, depthSigma, ImGradientThesh2, reProjectionSigma);

		for (int cid = 0; cid < (int)allImgs.size(); cid++)
		{
			for (int pyrID = 0; pyrID < 1; pyrID++)
				allImgs[cid].scaleFactor.push_back(1.0 / pow(2, pyrID));

			GaussianBlur(allImgs[cid].color, allImgs[cid].color, Size(5, 5), 0.707);
			buildPyramid(allImgs[cid].color, allImgs[cid].imgPyr, 0);
			BuildDepthPyramid(allImgs[cid].depth, allImgs[cid].depthPyr, allImgs[cid].width, allImgs[cid].height, 0);
		}

		FlowBasedBundleAdjustment(Path, alignmentParas, allImgs, allFlows, allCalibInfo, 0, 0, 0, 1, 1);

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

		DirectAlignment("C:/temp/Pipes-perfect", allImgs, allCalibInfo);

		return 0;
	}


	return 0;
}

