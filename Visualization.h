#include <cstdlib>
#include <stdio.h>
#include <algorithm>
#include <math.h>
#include <omp.h>

#include <opencv2/opencv.hpp>
#include "DataStructure.h"
#include "MathUlti.h"
#include "DataIO.h"
#include "GL/glut.h"


using namespace cv;
using namespace std;

#if !defined( VISUALIZATION_H )
#define VISUALIZATION_H

#define VIEWING_DISTANCE_MIN  1.0

void Draw_Axes();
void DrawCamera(bool highlight = false);
void RenderObjects();
void display(void);
void InitGraphics(void);
void Keyboard(unsigned char key, int x, int y);
void MouseButton(int button, int state, int x, int y);
void MouseMotion(int x, int y);
void ReshapeGL(int width, int height);

void visualization();
int visualizationDriver(char *inPath, int startF, int stopF, bool hasColor, int CurrentFrame);

void GetRCGL(CameraData &camInfo);
void GetRCGL(double *R, double *T, double *Rgl, double *C);

void ReadCurrentSfmGL(char *path, bool hasColor);
int ReadVideoPoseGL(char *path, int StartTime, int StopTime);
int ReadCurrent3DGL(char *path, bool hasColor, int timeID, bool setCoordinate);


int screenShot(char *Fname, int width, int height, bool color);
#endif


