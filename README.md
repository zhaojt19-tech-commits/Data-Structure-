#pragma warning(disable:4819)
#pragma warning(disable:4244)
#pragma warning(disable:4267)

#include <time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <limits>

using namespace std;

/*
名称：Projection(一张视图)
功能：判断投影是否在visual hull内部
变量：
	m_projMat;//3 * 4转换矩阵
	m_imag//cv::Mat类，表示一张图像
	m_threshold = 125//界限，无符号整型常量
函数：
	bool outOfRange(int x, int max)//判断点在某一维度是否出界(出界返回true)
	bool checkRange(double x, double y, double z)//判断是否落在投影内部(出界返回false)
*/
struct Projection
{
	Eigen::Matrix<float, 3, 4> m_projMat;//转换矩阵
	cv::Mat m_image;
	const uint m_threshold = 125;//阈值

	bool outOfRange(int x, int max)
	{
		return x < 0 || x >= max;
	}//判断点在某一维度是否出界(出界返回true)

	bool checkRange(double x, double y, double z)
	{
		Eigen::Vector3f vec3 = m_projMat * Eigen::Vector4f(x, y, z, 1);//获取点在投影上的坐标(u,v,1)
		int indX = vec3[1] / vec3[2];
		int indY = vec3[0] / vec3[2];//得到投影点实际坐标(x,y)

		if (outOfRange(indX, m_image.size().height) || outOfRange(indY, m_image.size().width))
			return false;//判断是否落在投影内部(出界返回false)
		return m_image.at<uchar>((uint)(vec3[1] / vec3[2]), (uint)(vec3[0] / vec3[2])) > m_threshold;
	}
};

/*
名称：CoordinateInfo
功能：用于index和实际坐标之间的转换
 变量：
	m_resolution//某一维度点数目
	m_min//某一维度最小值
	m_max//某一维度最大值
函数：
	double index2coor(int index)//获取某一维度实际坐标
*/
struct CoordinateInfo
{
	int m_resolution;//某一维度点数目
	double m_min;//某一维度最小值
	double m_max;//某一维度最大值

	double index2coor(int index)
	{
		return m_min + index * (m_max - m_min) / m_resolution;
	}

	CoordinateInfo(int resolution = 10, double min = 0.0, double max = 100.0)
		: m_resolution(resolution)
		, m_min(min)
		, m_max(max)
	{
	}
};
/*
名称：Model
功能：主体
类型：
	Pixel//二维bool型数组
	Voxel//三维bool型数组
变量：
    m_corrX,m_corrY,m_corrZ//CoordinateInfo结构体
	m_neiborSize//整型
	m_projectionList//Projection一维数组
	m_voxel,m_surface//三维bool型数组
函数：
	Model(int resX = 100, int resY = 100, int resZ = 100);
	void loadMatrix(const char* pFileName);
	void loadImage(const char* pDir, const char* pPrefix, const char* pSuffix);
	void getModel();
	void getSurface();
	void saveModel(const char* pFileName);
	void saveModelWithNormal(const char* pFileName);
	Eigen::Vector3f getNormal(int indX, int indY, int indZ);//仅被其它函数调用
*/
class Model
{
public:
	typedef std::vector<std::vector<bool>> Pixel;//像素矩阵(二维数组,bool类型)
	typedef std::vector<Pixel> Voxel;//体素矩阵(三维数组,bool类型)

	Model(int resX = 100, int resY = 100, int resZ = 100);
	~Model();

	void saveModel(const char* pFileName);
	void saveModelWithNormal(const char* pFileName);
	void loadMatrix(const char* pFileName);
	void loadImage(const char* pDir, const char* pPrefix, const char* pSuffix);
	void getModel();
	void getSurface();
	Eigen::Vector3f getNormal(int indX, int indY, int indZ);

private:
	CoordinateInfo m_corrX;
	CoordinateInfo m_corrY;
	CoordinateInfo m_corrZ;

	int m_neiborSize;

	std::vector<Projection> m_projectionList;

	Voxel m_voxel;
	Voxel m_surface;
};

// 分别设置xyz方向的Voxel分辨率(x、y、z方向点的数目)
Model::Model(int resX, int resY, int resZ)//main中初始化为(300, 300, 300)
	: m_corrX(resX, -5, 5)
	, m_corrY(resY, -10, 10)
	, m_corrZ(resZ, 15, 30)//使用index2coor接口即可得到实际坐标
{
	if (resX > 100)
		m_neiborSize = resX / 100;//这里值为3，不再改变
	else
		m_neiborSize = 1;
	m_voxel = Voxel(m_corrX.m_resolution, Pixel(m_corrY.m_resolution, std::vector<bool>(m_corrZ.m_resolution, true)));//300 * 300 * 300 初始化为true
	m_surface = m_voxel;
}

Model::~Model()
{
}



// 读取相机的内外参数(保留)
void Model::loadMatrix(const char* pFileName)
{
	std::ifstream fin(pFileName);//calibParamsI.txt

	int num;
	Eigen::Matrix<float, 3, 3> matInt;
	Eigen::Matrix<float, 3, 4> matExt;
	Projection projection;
	while (fin >> num)//依次读入20个摄像头的参数
	{
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				fin >> matInt(i, j);

		double temp;
		fin >> temp;
		fin >> temp;//吃掉无效字符
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 4; j++)
				fin >> matExt(i, j);

		projection.m_projMat = matInt * matExt;
		m_projectionList.push_back(projection);//最后构成包含20个元素的一维数组
	}
}

// 读取投影图片(保留)
void Model::loadImage(const char* pDir, const char* pPrefix, const char* pSuffix)
{
	int fileCount = m_projectionList.size();//图像个数
	std::string fileName(pDir);//wd_segmented
	fileName += '/';
	fileName += pPrefix;//WD2_
	for (int i = 0; i < fileCount; i++)
	{
		std::cout << fileName + std::to_string(i) + pSuffix << std::endl;//_00020_segmented.png
		m_projectionList[i].m_image = cv::imread(fileName + std::to_string(i) + pSuffix, CV_8UC1);//读入一张图像
	}
}

// 得到Voxel模型(优化)
void Model::getModel()
{
	int prejectionCount = m_projectionList.size();//20张图片

	for (int indexX = 0; indexX < m_corrX.m_resolution; indexX++)
		for (int indexY = 0; indexY < m_corrY.m_resolution; indexY++)
			for (int indexZ = 0; indexZ < m_corrZ.m_resolution; indexZ++)
				for (int i = 0; i < prejectionCount; i++)
				{
					double coorX = m_corrX.index2coor(indexX);//实际立体模型上点的x坐标
					double coorY = m_corrY.index2coor(indexY);//实际立体模型上点的y坐标
					double coorZ = m_corrZ.index2coor(indexZ);//实际立体模型上点的z坐标
					m_voxel[indexX][indexY][indexZ] = m_voxel[indexX][indexY][indexZ] && m_projectionList[i].checkRange(coorX, coorY, coorZ);//去掉外部点，置为false(只要在一张图片上出界，即可判定为外部点)
                    if (!m_voxel[indexX][indexY][indexZ]) break;
				}//此时已经为稀疏矩阵，仅true部分为体素点
}

// 获得Voxel模型的表面(优化)
void Model::getSurface()
{
	// 邻域：上、下、左、右、前、后。
	int dx[6] = { -1, 0, 0, 0, 0, 1 };
	int dy[6] = { 0, 1, -1, 0, 0, 0 };
	int dz[6] = { 0, 0, 0, 1, -1, 0 };

	// lambda表达式，用于判断某个点是否在Voxel的范围内
	auto outOfRange = [&](int indexX, int indexY, int indexZ){  
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= m_corrX.m_resolution
			|| indexY >= m_corrY.m_resolution
			|| indexZ >= m_corrZ.m_resolution;
	};//出界返回true

	for (int indexX = 0; indexX < m_corrX.m_resolution; indexX++)
		for (int indexY = 0; indexY < m_corrY.m_resolution; indexY++)
			for (int indexZ = 0; indexZ < m_corrZ.m_resolution; indexZ++)
			{
				if (!m_voxel[indexX][indexY][indexZ])
				{
					m_surface[indexX][indexY][indexZ] = false;
					continue;
				}
				bool ans = false;
				for (int i = 0; i < 6; i++)
				{
					ans = ans || outOfRange(indexX + dx[i], indexY + dy[i], indexZ + dz[i])
						|| !m_voxel[indexX + dx[i]][indexY + dy[i]][indexZ + dz[i]];
				}
				m_surface[indexX][indexY][indexZ] = ans;//只要有相邻点在外部，则该点为边界点
			}//相比m_vixel更加为稀疏，仅少部分为true，代表表面点
}

// 将模型导出为xyz格式(优化)，存入无法向模型点阵(优化)
void Model::saveModel(const char* pFileName)//saveModelWithoutNormal
{
	std::ofstream fout(pFileName);//WithoutNormal.xyz

	for (int indexX = 0; indexX < m_corrX.m_resolution; indexX++)
		for (int indexY = 0; indexY < m_corrY.m_resolution; indexY++)
			for (int indexZ = 0; indexZ < m_corrZ.m_resolution; indexZ++)
				if (m_surface[indexX][indexY][indexZ])//极稀疏矩阵全遍历
				{
					double coorX = m_corrX.index2coor(indexX);
					double coorY = m_corrY.index2coor(indexY);
					double coorZ = m_corrZ.index2coor(indexZ);
					fout << coorX << ' ' << coorY << ' ' << coorZ << std::endl;
				}
}
//存入带法向模型点阵(优化)
void Model::saveModelWithNormal(const char* pFileName)
{
	std::ofstream fout(pFileName);//WithNormal.xyz

	double midX = m_corrX.index2coor(m_corrX.m_resolution / 2);
	double midY = m_corrY.index2coor(m_corrY.m_resolution / 2);
	double midZ = m_corrZ.index2coor(m_corrZ.m_resolution / 2);//局部变量仅出现一次，不知道干什么

	for (int indexX = 0; indexX < m_corrX.m_resolution; indexX++)
		for (int indexY = 0; indexY < m_corrY.m_resolution; indexY++)
			for (int indexZ = 0; indexZ < m_corrZ.m_resolution; indexZ++)
				if (m_surface[indexX][indexY][indexZ])
				{
					double coorX = m_corrX.index2coor(indexX);
					double coorY = m_corrY.index2coor(indexY);
					double coorZ = m_corrZ.index2coor(indexZ);
					fout << coorX << ' ' << coorY << ' ' << coorZ << ' ';//输出表面点坐标

					Eigen::Vector3f nor = getNormal(indexX, indexY, indexZ);//调用getNormal函数，得到法向三维向量
					fout << nor(0) << ' ' << nor(1) << ' ' << nor(2) << std::endl;//输出法向方向(三维向量)
				}
}

//获取点的法向(三维向量表示)
Eigen::Vector3f Model::getNormal(int indX, int indY, int indZ)//对应indexX,indexY,indexZ
{
	auto outOfRange = [&](int indexX, int indexY, int indexZ){
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= m_corrX.m_resolution
			|| indexY >= m_corrY.m_resolution
			|| indexZ >= m_corrZ.m_resolution;
	};

	std::vector<Eigen::Vector3f> neiborList;//周围的表面点数组(三维向量数组)
	std::vector<Eigen::Vector3f> innerList;//周围的内部点数组(三维向量数组)

	for (int dX = -m_neiborSize; dX <= m_neiborSize; dX++)//m_neiborSize值为3
		for (int dY = -m_neiborSize; dY <= m_neiborSize; dY++)
			for (int dZ = -m_neiborSize; dZ <= m_neiborSize; dZ++)
			{
				if (!dX && !dY && !dZ)//dX,dY,dZ全为零跳过(跳过自身)
					continue;
				int neiborX = indX + dX;
				int neiborY = indY + dY;
				int neiborZ = indZ + dZ;
				if (!outOfRange(neiborX, neiborY, neiborZ))
				{
					float coorX = m_corrX.index2coor(neiborX);
					float coorY = m_corrY.index2coor(neiborY);
					float coorZ = m_corrZ.index2coor(neiborZ);
					if (m_surface[neiborX][neiborY][neiborZ])
						neiborList.push_back(Eigen::Vector3f(coorX, coorY, coorZ));
					else if (m_voxel[neiborX][neiborY][neiborZ])
						innerList.push_back(Eigen::Vector3f(coorX, coorY, coorZ));
				}
			}

	Eigen::Vector3f point(m_corrX.index2coor(indX), m_corrY.index2coor(indY), m_corrZ.index2coor(indZ));//待求法向的表面点向量表示

	Eigen::MatrixXf matA(3, neiborList.size());
	for (int i = 0; i < neiborList.size(); i++)
		matA.col(i) = neiborList[i] - point;
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(matA * matA.transpose());
	Eigen::Vector3f eigenValues = eigenSolver.eigenvalues();
	int indexEigen = 0;
	if (abs(eigenValues[1]) < abs(eigenValues[indexEigen]))
		indexEigen = 1;
	if (abs(eigenValues[2]) < abs(eigenValues[indexEigen]))
		indexEigen = 2;
	Eigen::Vector3f normalVector = eigenSolver.eigenvectors().col(indexEigen);
	
	Eigen::Vector3f innerCenter = Eigen::Vector3f::Zero();
	for (auto const& vec : innerList)
		innerCenter += vec;
	innerCenter /= innerList.size();

	if (normalVector.dot(point - innerCenter) < 0)
		normalVector *= -1;
	return normalVector;//这段应该不用优化，没细看，就是求法向向量
}

int main(int argc, char** argv)
{
	clock_t t = clock();

	// 分别设置xyz方向的Voxel分辨率
	Model model(300, 300, 300);

	// 读取相机的内外参数
	model.loadMatrix("../../calibParamsI.txt");

	// 读取投影图片
	model.loadImage("../../wd_segmented", "WD2_", "_00020_segmented.png");

	// 得到Voxel模型
	model.getModel();
	std::cout << "get model done\n";

	// 获得Voxel模型的表面
	model.getSurface();
	std::cout << "get surface done\n";

	// 将模型导出为xyz格式
	model.saveModel("../../WithoutNormal.xyz");
	std::cout << "save without normal done\n";

	model.saveModelWithNormal("../../WithNormal.xyz");
	std::cout << "save with normal done\n";

	system("PoissonRecon.x64 --in ../../WithNormal.xyz --out ../../mesh.ply");
	std::cout << "save mesh.ply done\n";

	t = clock() - t;
	std::cout << "time: " << (float(t) / CLOCKS_PER_SEC) << "seconds\n";

	return (0);
}
