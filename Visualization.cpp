#include"Visualization.h"

#pragma comment(lib, "glut32.lib")

using namespace std;
using namespace cv;

#define RADPERDEG 0.0174533

char *Path;

int renderTime = 0;
bool showImg = true;
GLfloat UnitScale = 1.0f; //1 unit corresponds to 1 mm
GLfloat g_ratio, g_coordAxisLength, g_fViewDistance, g_nearPlane, g_farPlane;
int g_Width = 1600, g_Height = 900, org_Width = g_Width, org_Height = g_Height, g_xClick = 0, g_yClick = 0, g_mouseYRotate = 0, g_mouseXRotate = 0;

GLfloat CameraSize, pointSize, normalSize, arrowThickness;
int nCorpusCams = 0, nNonCorpusCams, CurrentFrameID = 0, oldFrameID = 0;

enum cam_mode { CAM_DEFAULT, CAM_ROTATE, CAM_ZOOM, CAM_PAN };
static cam_mode g_camMode = CAM_DEFAULT;
GLfloat Red[3] = { 1, 0, 0 }, Green[3] = { 0, 1, 0 }, Blue[3] = { 0, 0, 1 }, White[3] = { 1, 1, 1 }, Yellow[3] = { 1.0f, 1.0f, 0 }, Magneta[3] = { 1.f, 0.f, 1.f }, Cycan[3] = { 0.f, 1.f, 1.f };

bool g_bButton1Down = false, drawPointColor = false; int colorCoded = 1;
bool ReCenterNeeded = false, PickingMode = false, bFullsreen = false, changeBackgroundColor = false, showAxis = false;
bool SaveScreen = false, ImmediateSnap = false, SaveStaticViewingParameters = false, SetStaticViewingParameters = true, RenderedReady = false;

bool drawCorpusPoints = 1, drawCorpusCameras = false, drawNonCorpusPoints = true, drawNonCorpusCameras = true;
double DisplaystartFrame = 0.0, DisplayTimeStep = 0.01; //60fps

GLfloat PointsCentroid[3], PointVar[3];
vector<int> PickedCorpusPoints, PickedNonCorpusPoints, PickedCorpusCams, PickedNonCorpusCams;
vector<int> selectedCams;

int startFrame, stopFrame;
typedef struct { GLfloat  viewDistance, CentroidX, CentroidY, CentroidZ; int CurrentFrameID, mouseYRotate, mouseXRotate; } ViewingParas;
VisualizationManager g_vis;

//float fps = 10.0, Tscale = 1000.0, radius = 2.5e3;
float fps = 1, Tscale = 1, radius = 2.5e3;
int Pick(int x, int y)
{
	GLuint buff[64];
	GLint hits, view[4];

	//selection data
	glSelectBuffer(64, buff);
	glGetIntegerv(GL_VIEWPORT, view);
	glRenderMode(GL_SELECT);

	//Push stack for picking
	glInitNames();
	glPushName(0);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluPickMatrix(x, view[3] - y, 10.0, 10.0, view);
	gluPerspective(65.0, g_ratio, g_nearPlane, g_farPlane);

	glMatrixMode(GL_MODELVIEW);
	RenderObjects();
	glutSwapBuffers();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	hits = glRenderMode(GL_RENDER);
	glMatrixMode(GL_MODELVIEW);

	if (hits > 0)
		return buff[3];
	else
		return -1;
}
void SelectionFunction(int x, int y, bool append_rightClick = false)
{
	if (PickingMode)
	{
		int pickedID = Pick(x, y);
		if (pickedID < 0)
			return;

		//do something
	}

	return;
}
void Keyboard(unsigned char key, int x, int y)
{
	char Fname[200];
	switch (key)
	{
	case 27:             // ESCAPE key
		exit(0);
		break;
	case 'F':
		bFullsreen = !bFullsreen;
		if (bFullsreen)
			glutFullScreen();
		else
		{
			glutReshapeWindow(org_Width, org_Height);
			glutInitWindowPosition(0, 0);
		}
		break;
	case 'c':
		printf("Current cameraSize: %f. Please enter the new size: ", CameraSize);
		cin >> CameraSize;
		printf("New cameraSize: %f\n", CameraSize);
		break;
	case 'p':
		printf("Current pointSize: %f. Please enter the new size: ", pointSize);
		cin >> pointSize;
		printf("New pointSize: %f\n", pointSize);
		break;
	case 'u':
		printf("Current UnitScale: %f. Please enter the size: ", UnitScale);
		cin >> UnitScale;
		printf("New UnitScale: %f\n", UnitScale);
		break;
	case 'b':
		changeBackgroundColor = !changeBackgroundColor;
		break;
	case 'P':
		PickingMode = !PickingMode;
		if (PickingMode)
			printf("Picking Mode: ON\n");
		else
			printf("Picking Mode: OFF\n");
		break;
	case '1':
		printf("Toggle corpus points display\n");
		drawCorpusPoints = !drawCorpusPoints;
		break;
	case '2':
		drawCorpusCameras = !drawCorpusCameras;
		if (drawCorpusCameras)
			printf("Corpus cameras display: ON\n");
		else
			printf("Corpus cameras display: OFF\n");
		break;
	case '3':
		drawNonCorpusCameras = !drawNonCorpusCameras;
		if (drawNonCorpusCameras)
			printf("Corpus moving cameras display: ON\n");
		else
			printf("Corpus moving cameras display: OFF\n");
		break;
	case '4':
		printf("Save OpenGL viewing parameters\n");
		SaveStaticViewingParameters = true;
		break;
	case '5':
		printf("Read OpenGL viewing parameters\n");
		SetStaticViewingParameters = true;
		break;
	case 'A':
		printf("Toggle axis display\n");
		showAxis = !showAxis;
		break;
	case 's':
		SaveScreen = !SaveScreen;
		if (SaveScreen)
			printf("Save screen: ON\n");
		else
			printf("Save screen: OFF\n");
		break;
	case 'S':
		ImmediateSnap = !ImmediateSnap;
		if (ImmediateSnap)
			printf("Snap screen: ON\n");
		else
			printf("Snap screen: OFF\n");
		break;
	case 'a':
		g_mouseXRotate += 5;
		break;
	case 'd':
		g_mouseXRotate -= 5;
		break;
	case 'w':
		g_fViewDistance -= 10.0f*UnitScale;
		break;
	case 'x':
		g_fViewDistance += 10.0f*UnitScale;
		break;
	}
	glutPostRedisplay();
}
void SpecialInput(int key, int x, int y)
{
	switch (key)
	{
	case GLUT_KEY_PAGE_UP:
		break;
	case GLUT_KEY_PAGE_DOWN:
		break;
	case GLUT_KEY_UP:
		break;
	case GLUT_KEY_DOWN:
		break;
	case GLUT_KEY_LEFT:
		CurrentFrameID--;
		if (CurrentFrameID < startFrame)
			CurrentFrameID = startFrame;
		printf("Current frameID: %)\n", CurrentFrameID);
		showImg = true;
		break;
	case GLUT_KEY_RIGHT:
		CurrentFrameID++;
		if (CurrentFrameID > stopFrame)
			CurrentFrameID = stopFrame;
		printf("frameID time: %d\n", CurrentFrameID);
		showImg = true;
		break;
	case GLUT_KEY_HOME:
		CurrentFrameID = startFrame;
		printf("Current frameID: %)\n", CurrentFrameID);
		break;
	case GLUT_KEY_END:
		CurrentFrameID = stopFrame;
		printf("Current frameID: %)\n", CurrentFrameID);
		break;
	}

	glutPostRedisplay();
}
void MouseButton(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON)
	{
		g_bButton1Down = (state == GLUT_DOWN) ? true : false;
		g_xClick = x;
		g_yClick = y;

		if (glutGetModifiers() == GLUT_ACTIVE_CTRL)
			g_camMode = CAM_ROTATE;
		else if (glutGetModifiers() == GLUT_ACTIVE_SHIFT)
			g_camMode = CAM_PAN;
		else if (glutGetModifiers() == GLUT_ACTIVE_ALT)
			g_camMode = CAM_ZOOM;
		else
		{
			g_camMode = CAM_DEFAULT;

			if (state == GLUT_DOWN) //picking single point
				SelectionFunction(x, y, false);
		}
	}
	else if (button == GLUT_RIGHT_BUTTON)
	{
		if (glutGetModifiers() == GLUT_ACTIVE_CTRL)
		{
			g_camMode = CAM_DEFAULT;

			ReCenterNeeded = true;
			SelectionFunction(x, y, false);
		}
		if (glutGetModifiers() == GLUT_ACTIVE_ALT)//Deselect
		{
			printf("Deselect all picked objects\n");
			PickedCorpusCams.clear(), PickedNonCorpusCams.clear(), PickedCorpusPoints.clear();
		}
	}
	else if (button == GLUT_MIDDLE_BUTTON)
	{
		g_xClick = x;
		g_yClick = y;
		g_bButton1Down = true;
		g_camMode = CAM_ZOOM;
	}
}
void MouseMotion(int x, int y)
{
	if (g_bButton1Down)
	{
		if (g_camMode == CAM_ZOOM)
			g_fViewDistance += 5.0f*(y - g_yClick) *UnitScale;
		else if (g_camMode == CAM_ROTATE)
		{
			showAxis = true;
			g_mouseXRotate += (x - g_xClick);
			g_mouseYRotate -= (y - g_yClick);
			g_mouseXRotate = g_mouseXRotate % 360;
			g_mouseYRotate = g_mouseYRotate % 360;
		}
		else if (g_camMode == CAM_PAN)
		{
			showAxis = true;
			float dX = -(x - g_xClick)*UnitScale, dY = (y - g_yClick)*UnitScale;

			float cphi = cos(-Pi*g_mouseYRotate / 180), sphi = sin(-Pi*g_mouseYRotate / 180);
			float Rx[9] = { 1, 0, 0, 0, cphi, -sphi, 0, sphi, cphi };

			cphi = cos(-Pi*g_mouseXRotate / 180), sphi = sin(-Pi*g_mouseXRotate / 180);
			float Ry[9] = { cphi, 0, sphi, 0, 1, 0, -sphi, 0, cphi };

			float R[9];  mat_mul(Rx, Ry, R, 3, 3, 3);
			float incre[3], orgD[3] = { dX, dY, 0 }; mat_mul(R, orgD, incre, 3, 3, 1);

			PointsCentroid[0] += incre[0], PointsCentroid[1] += incre[1], PointsCentroid[2] += incre[2];
		}

		g_xClick = x, g_yClick = y;

		glutPostRedisplay();
	}
}
void ReshapeGL(int width, int height)
{
	g_Width = width;
	g_Height = height;
	glViewport(0, 0, g_Width, g_Height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	g_ratio = (float)g_Width / g_Height;
	gluPerspective(65.0, g_ratio, g_nearPlane, g_farPlane);
	glMatrixMode(GL_MODELVIEW);
}
void GetRGL(CameraData &camInfo, GLfloat *Rgl)
{

	Rgl[0] = camInfo.R[0], Rgl[1] = camInfo.R[1], Rgl[2] = camInfo.R[2], Rgl[3] = 0.0;
	Rgl[4] = camInfo.R[3], Rgl[5] = camInfo.R[4], Rgl[6] = camInfo.R[5], Rgl[7] = 0.0;
	Rgl[8] = camInfo.R[6], Rgl[9] = camInfo.R[7], Rgl[10] = camInfo.R[8], Rgl[11] = 0.0;
	Rgl[12] = 0, Rgl[13] = 0, Rgl[14] = 0, Rgl[15] = 1.0;

	return;
}

void DrawCube(Point3f &length)
{
	glBegin(GL_QUADS);
	// Top face (z)
	glColor4f(0.0f, 1.0f, 0.0f, 0.3f);     // Green
	glVertex3f(-length.x / 2, -length.y / 2, length.z / 2);
	glVertex3f(-length.x / 2, length.y / 2, length.z / 2);
	glVertex3f(length.x / 2, length.y / 2, length.z / 2);
	glVertex3f(length.x / 2, -length.y / 2, length.z / 2);

	// Bottom face (-z)
	glColor4f(1.0f, 0.5f, 0.0f, 0.3f);     // Orange
	glVertex3f(-length.x / 2, -length.y / 2, -length.z / 2);
	glVertex3f(-length.x / 2, length.y / 2, -length.z / 2);
	glVertex3f(length.x / 2, length.y / 2, -length.z / 2);
	glVertex3f(length.x / 2, -length.y / 2, -length.z / 2);

	// Front face  (x)
	glColor4f(1.0f, 0.0f, 0.0f, 0.3f);     // Red
	glVertex3f(length.x / 2, -length.y / 2, -length.z / 2);
	glVertex3f(length.x / 2, -length.y / 2, length.z / 2);
	glVertex3f(length.x / 2, length.y / 2, length.z / 2);
	glVertex3f(length.x / 2, length.y / 2, -length.z / 2);

	// Back face (-x)
	glColor4f(1.0f, 1.0f, 0.0f, 0.3f);     // Yellow
	glVertex3f(-length.x / 2, -length.y / 2, -length.z / 2);
	glVertex3f(-length.x / 2, -length.y / 2, length.z / 2);
	glVertex3f(-length.x / 2, length.y / 2, length.z / 2);
	glVertex3f(-length.x / 2, length.y / 2, -length.z / 2);

	// Left face (y)
	glColor4f(0.0f, 0.0f, 1.0f, 0.3f);     // Blue
	glVertex3f(-length.x / 2, length.y / 2, -length.z / 2);
	glVertex3f(-length.x / 2, length.y / 2, length.z / 2);
	glVertex3f(length.x / 2, length.y / 2, length.z / 2);
	glVertex3f(length.x / 2, length.y / 2, -length.z / 2);

	// Right face (-y)
	glColor4f(1.0f, 0.0f, 1.0f, 0.3f);     // Magenta
	glVertex3f(-length.x / 2, -length.y / 2, -length.z / 2);
	glVertex3f(-length.x / 2, -length.y / 2, length.z / 2);
	glVertex3f(length.x / 2, -length.y / 2, length.z / 2);
	glVertex3f(length.x / 2, -length.y / 2, -length.z / 2);
	glEnd();  // End of drawing color-cube

}
void Draw_Axes(void)
{
	glPushMatrix();

	glBegin(GL_LINES);
	glColor3f(1, 0, 0); // X axis is red.
	glVertex3f(0, 0, 0);
	glVertex3f(g_coordAxisLength, 0, 0);
	glColor3f(0, 1, 0); // Y axis is green.
	glVertex3f(0, 0, 0);
	glVertex3f(0, g_coordAxisLength, 0);
	glColor3f(0, 0, 1); // Z axis is blue.
	glVertex3f(0, 0, 0);
	glVertex3f(0, 0, g_coordAxisLength);
	glEnd();

	glPopMatrix();
}
void DrawCamera(bool highlight)
{
	glColorMaterial(GL_FRONT, GL_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	if (!highlight)
		glColor3fv(Red);
	else
		glColor3fv(Blue);

	glBegin(GL_LINES);
	glVertex3f(0, 0, 0); //
	glVertex3f(0.5*CameraSize, 0.5*CameraSize, 1 * CameraSize); //
	glVertex3f(0, 0, 0); //
	glVertex3f(0.5*CameraSize, -0.5*CameraSize, 1 * CameraSize); //
	glVertex3f(0, 0, 0); //
	glVertex3f(-0.5*CameraSize, 0.5*CameraSize, 1 * CameraSize); //
	glVertex3f(0, 0, 0); //
	glVertex3f(-0.5*CameraSize, -0.5*CameraSize, 1 * CameraSize); //
	glEnd();

	if (!highlight)
		glColor3fv(Red);
	else
		glColor3fv(Blue);

	// we also has to draw a square for the bottom of the pyramid so that as it rotates we wont be able see inside of it but all a square is is two triangle put together
	glBegin(GL_LINE_STRIP);
	glVertex3f(0.5*CameraSize, 0.5*CameraSize, 1 * CameraSize);
	glVertex3f(-0.5*CameraSize, 0.5*CameraSize, 1 * CameraSize);
	glVertex3f(-0.5*CameraSize, -0.5*CameraSize, 1 * CameraSize);
	glVertex3f(0.5*CameraSize, -0.5*CameraSize, 1 * CameraSize);
	glVertex3f(0.5*CameraSize, 0.5*CameraSize, 1 * CameraSize);
	glEnd();
	glDisable(GL_COLOR_MATERIAL);
}
void Arrow(GLdouble x1, GLdouble y1, GLdouble z1, GLdouble x2, GLdouble y2, GLdouble z2, GLdouble D)
{
	double x = x2 - x1;
	double y = y2 - y1;
	double z = z2 - z1;
	double L = sqrt(x*x + y*y + z*z);

	GLUquadricObj *quadObj;

	glPushMatrix();

	glTranslated(x1, y1, z1);

	if (x != 0.f || y != 0.f)
	{
		glRotated(atan2(y, x) / RADPERDEG, 0., 0., 1.);
		glRotated(atan2(sqrt(x*x + y*y), z) / RADPERDEG, 0., 1., 0.);
	}
	else if (z < 0)
		glRotated(180, 1., 0., 0.);

	glTranslatef(0, 0, L - 4 * D);

	quadObj = gluNewQuadric();
	gluQuadricDrawStyle(quadObj, GLU_FILL);
	gluQuadricNormals(quadObj, GLU_SMOOTH);
	gluCylinder(quadObj, 2 * D, 0.0, 4 * D, 32, 1);
	gluDeleteQuadric(quadObj);

	quadObj = gluNewQuadric();
	gluQuadricDrawStyle(quadObj, GLU_FILL);
	gluQuadricNormals(quadObj, GLU_SMOOTH);
	gluDisk(quadObj, 0.0, 2 * D, 32, 1);
	gluDeleteQuadric(quadObj);

	glTranslatef(0, 0, -L + 4 * D);

	quadObj = gluNewQuadric();
	gluQuadricDrawStyle(quadObj, GLU_FILL);
	gluQuadricNormals(quadObj, GLU_SMOOTH);
	gluCylinder(quadObj, D, D, L - 4 * D, 32, 1);
	gluDeleteQuadric(quadObj);

	quadObj = gluNewQuadric();
	gluQuadricDrawStyle(quadObj, GLU_FILL);
	gluQuadricNormals(quadObj, GLU_SMOOTH);
	gluDisk(quadObj, 0.0, D, 32, 1);
	gluDeleteQuadric(quadObj);

	glPopMatrix();

}

void RenderObjects()
{
	//Draw not picked corpus points
	if (drawCorpusPoints)
	{
		for (int ii = 0; ii < (int)g_vis.CorpusPointPosition.size(); ii++)
		{
			glLoadName(ii + nCorpusCams);//for picking purpose

			bool picked = false;
			for (unsigned int jj = 0; jj < PickedCorpusPoints.size(); jj++)
				if (ii == PickedCorpusPoints[jj])
					picked = true;

			glPushMatrix();
			glTranslatef(g_vis.CorpusPointPosition[ii].x - PointsCentroid[0], g_vis.CorpusPointPosition[ii].y - PointsCentroid[1], g_vis.CorpusPointPosition[ii].z - PointsCentroid[2]);
			if (!picked&&drawPointColor)
				glColor3f(g_vis.CorpusPointColor[ii].x, g_vis.CorpusPointColor[ii].y, g_vis.CorpusPointColor[ii].z);
			else
				glColor3fv(Red);
			glutSolidSphere(pointSize, 4, 4);
			glPopMatrix();
		}
	}

	//draw Corpus camera 
	if (drawCorpusCameras)
	{
		for (int ii = 0; ii < g_vis.glCorpusCameraInfo.size(); ii++)
		{
			double* centerPt = g_vis.glCorpusCameraInfo[ii].C;
			GLfloat Rgl[16]; GetRGL(g_vis.glCorpusCameraInfo[ii], Rgl);

			glLoadName(ii);//for picking purpose

			bool picked = false;
			for (int jj = 0; jj < !picked && PickedCorpusCams.size(); jj++)
				if (ii == PickedCorpusCams[jj])
					picked = true;

			glPushMatrix();
			glTranslatef(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
			glMultMatrixf(Rgl);
			if (picked)
				DrawCamera(true);
			else
				DrawCamera();
			glPopMatrix();
		}
	}

	//Draw not picked corpus points
	if (drawNonCorpusPoints)
	{
		for (int ii = 0; ii < (int)g_vis.PointPosition.size(); ii++)
		{
			glLoadName(ii + nCorpusCams + (int)g_vis.CorpusPointPosition.size());//for picking purpose

			bool picked = false;
			for (unsigned int jj = 0; jj < PickedNonCorpusPoints.size(); jj++)
				if (ii == PickedNonCorpusPoints[jj])
					picked = true;

			glPushMatrix();
			glTranslatef(g_vis.PointPosition[ii].x - PointsCentroid[0], g_vis.PointPosition[ii].y - PointsCentroid[1], g_vis.PointPosition[ii].z - PointsCentroid[2]);
			if (!picked&&drawPointColor)
				glColor3f(g_vis.PointColor[ii].x, g_vis.PointColor[ii].y, g_vis.PointColor[ii].z);
			else
				glColor3fv(Red);
			glutSolidSphere(pointSize, 4, 4);
			glPopMatrix();
		}
	}

	//draw 3d moving cameras 
	if (drawNonCorpusCameras)
	{
		int cumCamID = 0;
		for (unsigned int i = 0; i < g_vis.glCameraPoseInfo.size(); i++)
		{
			cumCamID++;
			int id = g_vis.glCameraPoseInfo[i].frameID;
			if (id > CurrentFrameID)
				break;

			double* centerPt = g_vis.glCameraPoseInfo[i].C;
			if (abs(centerPt[0]) + abs(centerPt[1]) + abs(centerPt[2]) < 0.01)
				continue;
			GLfloat Rgl[16]; GetRGL(g_vis.glCameraPoseInfo[i], Rgl);

			glLoadName(cumCamID + nCorpusCams + (int)g_vis.CorpusPointPosition.size() + (int)g_vis.PointPosition.size());//for picking purpose

			glPushMatrix();
			glTranslatef(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
			glMultMatrixf(Rgl);
			DrawCamera();
			glPopMatrix();
		}
	}

	return;
}

void display(void)
{
	if (changeBackgroundColor)
		glClearColor(1.0, 1.0, 1.0, 0.0);
	else
		glClearColor(0.0, 0.0, 0.0, 0.0);

	if (SaveStaticViewingParameters)
	{
		char Fname[200]; sprintf(Fname, "%s/OpenGLViewingPara.txt", Path);	FILE *fp = fopen(Fname, "w+");
		fprintf(fp, "%.8f %d %d %.8f %.8f %.8f ", g_fViewDistance, g_mouseXRotate, g_mouseYRotate, PointsCentroid[0], PointsCentroid[1], PointsCentroid[2]);
		fclose(fp);
		SaveStaticViewingParameters = false;
	}
	if (SetStaticViewingParameters)
	{
		char Fname[200]; sprintf(Fname, "%s/OpenGLViewingPara.txt", Path); FILE *fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			fscanf(fp, "%f %d %d %f %f %f", &g_fViewDistance, &g_mouseXRotate, &g_mouseYRotate, &PointsCentroid[0], &PointsCentroid[1], &PointsCentroid[2]);
			fclose(fp);
		}
		SetStaticViewingParameters = false;
	}


	// Clear frame buffer and depth buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glLoadIdentity();
	glTranslatef(0, 0, -g_fViewDistance);
	//glRotated(180,  0, 0, 1), glRotated(180, 0, 1, 0);
	glRotated(-g_mouseYRotate, 1, 0, 0);
	glRotated(-g_mouseXRotate, 0, 1, 0);

	RenderObjects();
	if (showAxis)
		Draw_Axes(), showAxis = false;

	glutSwapBuffers();

	if (SaveScreen && oldFrameID != CurrentFrameID)
	{
		char Fname[200];	sprintf(Fname, "%s/ScreenShot", Path); makeDir(Fname);
		sprintf(Fname, "%s/ScreenShot/%d.png", Path, CurrentFrameID);
		screenShot(Fname, g_Width, g_Height, true);
		oldFrameID = CurrentFrameID;
	}
}
void IdleFunction(void)
{
	return;
}
void Visualization()
{
	char *myargv[1];
	int myargc = 1;
	myargv[0] = "SfM";
	glutInit(&myargc, myargv);

	glutInitWindowSize(g_Width > 1920 ? 1920 : g_Width, g_Height > 1080 ? 1080 : g_Height);
	glutInitWindowPosition(0, 0);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

	glutCreateWindow("SfM!");


	glShadeModel(GL_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_DEPTH_TEST);
	//glEnable(GL_CULL_FACE);

	//select clearing (background) color
	//glClearColor(1.0, 1.0, 1.0, 0.0);
	glClearColor(1.0, 1.0, 1.0, 0.0);

	glutDisplayFunc(display);
	glutKeyboardFunc(Keyboard);
	glutSpecialFunc(SpecialInput);
	glutReshapeFunc(ReshapeGL);
	glutMouseFunc(MouseButton);
	glutIdleFunc(IdleFunction);
	glutMotionFunc(MouseMotion);

	glutMainLoop();

}
int visualizationDriver(char *inPath, int startF, int stopF, bool hasColor, int CurrentFrame)
{
	Path = inPath;
	drawPointColor = hasColor;
	startFrame = startF, stopFrame = stopF;

	UnitScale = 0.004;// sqrt(pow(PointVar[0], 2) + pow(PointVar[1], 2) + pow(PointVar[2], 2)) / 100.0;
	g_coordAxisLength = 20.f*UnitScale, g_fViewDistance = 1000 * UnitScale* VIEWING_DISTANCE_MIN;
	g_nearPlane = 1.0*UnitScale, g_farPlane = 30000.f * UnitScale;
	CameraSize = 20.0f*UnitScale, pointSize = 1.0f*UnitScale, normalSize = 5.f*UnitScale, arrowThickness = .1f*UnitScale;

	ReadCurrentSfmGL(Path, drawPointColor);
	ReadVideoPoseGL(Path, startFrame, stopFrame);
	ReadCurrent3DGL(Path, drawPointColor, 0, 1);

	oldFrameID = CurrentFrame, CurrentFrameID = CurrentFrame;

	Visualization();
	destroyAllWindows();

	return 0;
}

void ReadCurrentSfmGL(char *path, bool drawPointColor)
{
	char Fname[200];
	int viewID, id;

	CameraData camI;
	Corpus CorpusData;
	sprintf(Fname, "%s/Corpus/BA_Camera_AllParams_after.txt", path);
	if (ReadCalibInfo(Fname, CorpusData) == 0)
		for (int ii = 0; ii < CorpusData.nCameras; ii++)
			g_vis.glCorpusCameraInfo.push_back(CorpusData.camera[ii]);

	g_vis.CorpusPointPosition.clear(); g_vis.CorpusPointPosition.reserve(10e5);
	if (drawPointColor)
		g_vis.CorpusPointColor.clear(), g_vis.CorpusPointColor.reserve(10e5);

	Point3i iColor; Point3f fColor; Point3f t3d;
	bool filenotvalid = false;
	sprintf(Fname, "%s/Corpus/3dGL.xyz", path);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return;
	}
	while (fscanf(fp, "%d %f %f %f ", &id, &t3d.x, &t3d.y, &t3d.z) != EOF)
	{
		if (drawPointColor)
		{
			fscanf(fp, "%d %d %d ", &iColor.x, &iColor.y, &iColor.z);
			fColor.x = 1.0*iColor.x / 255;
			fColor.y = 1.0*iColor.y / 255;
			fColor.z = 1.0*iColor.z / 255;
			g_vis.CorpusPointColor.push_back(fColor);
		}
		PointsCentroid[0] += t3d.x, PointsCentroid[1] += t3d.y, PointsCentroid[2] += t3d.z;
		g_vis.CorpusPointPosition.push_back(t3d);
	}
	fclose(fp);

	if (g_vis.CorpusPointPosition.size() > 0)
	{
		PointsCentroid[0] /= g_vis.CorpusPointPosition.size();
		PointsCentroid[1] /= g_vis.CorpusPointPosition.size();
		PointsCentroid[2] /= g_vis.CorpusPointPosition.size();

		PointVar[0] = 0.0, PointVar[1] = 0.0, PointVar[2] = 0.0;
		for (int ii = 0; ii < g_vis.CorpusPointPosition.size(); ii++)
		{
			PointVar[0] += pow(g_vis.CorpusPointPosition[ii].x - PointsCentroid[0], 2);
			PointVar[1] += pow(g_vis.CorpusPointPosition[ii].y - PointsCentroid[1], 2);
			PointVar[2] += pow(g_vis.CorpusPointPosition[ii].z - PointsCentroid[2], 2);
		}
		PointVar[0] = sqrt(PointVar[0] / g_vis.CorpusPointPosition.size());
		PointVar[1] = sqrt(PointVar[1] / g_vis.CorpusPointPosition.size());
		PointVar[2] = sqrt(PointVar[2] / g_vis.CorpusPointPosition.size());
	}
	else
		PointsCentroid[0] = PointsCentroid[1] = PointsCentroid[2] = 0;

	return;
}
int ReadVideoPoseGL(char *path, int startFrame, int stopFrame)
{
	char Fname[200];
	CameraData *PoseInfo = new CameraData[stopFrame + 1];

	int frameID;
	double rt[6], R[9], T[3], Rgl[16], Cgl[3], dummy[6];
	sprintf(Fname, "%s/CamPose_0.txt", Path);
	if (IsFileExist(Fname) == 0)
			return 1;
	FILE *fp = fopen(Fname, "r");

	for (int jj = 0; jj < stopFrame + 1; jj++)
		PoseInfo[jj].valid = false;
	while (fscanf(fp, "%d ", &frameID) != EOF)
	{
		PoseInfo[frameID].valid = true;
		PoseInfo[frameID].frameID = frameID;
		for (int jj = 0; jj < 6; jj++)
			fscanf(fp, "%lf ", &PoseInfo[frameID].rt[jj]);
		getRfromr(PoseInfo[frameID].rt, PoseInfo[frameID].R);
		GetCfromT(PoseInfo[frameID].R, PoseInfo[frameID].rt + 3, PoseInfo[frameID].C);
	}
	fclose(fp);

	for (int jj = startFrame; jj < stopFrame; jj++)
		if (PoseInfo[jj].valid)
			g_vis.glCameraPoseInfo.push_back(PoseInfo[jj]);
		else
		{
			CameraData dummyCam;
			dummyCam.frameID = jj;
			dummyCam.valid = false;
			g_vis.glCameraPoseInfo.push_back(dummyCam);
		}

	nNonCorpusCams = stopFrame - startFrame + 1;

	return 0;
}
int ReadCurrent3DGL(char *path, bool drawPointColor, int CurrentFrameID, bool setCoordinate)
{
	char Fname[200];
	g_vis.PointPosition.clear(); g_vis.PointPosition.reserve(10e5);
	if (drawPointColor)
		g_vis.PointColor.clear(), g_vis.PointColor.reserve(10e5);

	if (setCoordinate)
		PointsCentroid[0] = 0.0f, PointsCentroid[1] = 0.0f, PointsCentroid[2] = 0.f;

	Point3i iColor; Point3f fColor; Point3f t3d, n3d;
	sprintf(Fname, "%s/3d_0_0.txt", path, CurrentFrameID); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return 1;
	}
	while (fscanf(fp, "%f %f %f ", &t3d.x, &t3d.y, &t3d.z) != EOF)
	{
		if (drawPointColor)
		{
			fscanf(fp, "%d %d %d ", &iColor.x, &iColor.y, &iColor.z);
			fColor.x = 1.0*iColor.x / 255;
			fColor.y = 1.0*iColor.y / 255;
			fColor.z = 1.0*iColor.z / 255;
			g_vis.PointColor.push_back(fColor);
		}
		else
			g_vis.PointColor.push_back(Point3f(255, 0, 0));

		if (setCoordinate)
			PointsCentroid[0] += t3d.x, PointsCentroid[1] += t3d.y, PointsCentroid[2] += t3d.z;
		g_vis.PointPosition.push_back(t3d);
	}
	fclose(fp);



	if (setCoordinate)
		PointsCentroid[0] /= g_vis.PointPosition.size(), PointsCentroid[1] /= g_vis.PointPosition.size(), PointsCentroid[2] /= g_vis.PointPosition.size();

	return 0;
}

int screenShot(char *Fname, int width, int height, bool color)
{
	int ii, jj;

	unsigned char *data = new unsigned char[width*height * 4];
	IplImage *cvImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);

	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, data);

	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			cvImg->imageData[3 * ii + 3 * jj*width] = data[3 * ii + 3 * (height - 1 - jj)*width + 2],
			cvImg->imageData[3 * ii + 3 * jj*width + 1] = data[3 * ii + 3 * (height - 1 - jj)*width + 1],
			cvImg->imageData[3 * ii + 3 * jj*width + 2] = data[3 * ii + 3 * (height - 1 - jj)*width];

	if (color)
		cvSaveImage(Fname, cvImg);
	else
	{
		IplImage *cvImgGray = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
		cvCvtColor(cvImg, cvImgGray, CV_BGR2GRAY);
		cvSaveImage(Fname, cvImgGray);
		cvReleaseImage(&cvImgGray);
	}

	cvReleaseImage(&cvImg);

	delete[]data;
	return 0;
}
