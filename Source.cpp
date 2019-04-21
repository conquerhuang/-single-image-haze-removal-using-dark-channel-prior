#include<opencv2/opencv.hpp>
#include<string>
#include "Source.h"
#include "get_dark_channel.h"
#include "get_atmosphere.h"
#include "get_est_transmap.h"
#include "guided_filter.h"
#include "get_radiance.h"
using namespace std;

constexpr int win_size{ 15 };
const double omega{ 0.9 };//决定复原图像中雾保留程度,在一些情况下，该参数的设置会影响去雾效果，需要根据图像自适应

int main(void)
{
	cv::Mat input_image;
	input_image = cv::imread("2.jpg");
	if (input_image.empty())//判断图像是否读取成功
	{
		cout << "error in get image" << endl;
		return -1;
	}

	clock_t start{ 0 }, finish{ 0 };
	start = clock();

	cv::Mat image_double;//获得double类型的图像，防止后续计算溢出
	input_image.convertTo(image_double, CV_32F, 1.0 / 255, 0);//转换为浮点图，方便后续计算而不溢出

	/*************************************获取暗通道***********************************/
	cv::Mat dark_channel;
	get_darkchannel(image_double, dark_channel, win_size);
	/*********************************************************************************/

	/**************************由暗通道和原图计算大气光*********************************/
	cv::Vec3f atmosphere{ 0,0,0 };//用于存储大气光值的向量
	get_atmosphere(image_double, dark_channel, atmosphere);//通过函数获取大气光
	/*********************************************************************************/

	/*********************************计算光照传递评估图*******************************/
	cv::Mat trans_est;
	get_esttransmap(image_double,atmosphere,omega,win_size,trans_est);
	/*********************************************************************************/
	
	/***************************采用导向滤波改良传递评估图******************************/
	cv::Mat trans_map;//光照传输图矩阵
	cv::Mat image_gray;//原图对应的灰度图
	cv::cvtColor(image_double, image_gray, CV_BGR2GRAY);
	//采用导向滤波计算传递矩阵
	trans_map = fastGuidedFilter(image_gray, trans_est, 15, 0.001, 1);;
	/*********************************************************************************/

	/********************************获得无雾图像**************************************/
	cv::Mat radiance;
	get_radiance(image_double, trans_map, atmosphere, radiance);
	/*********************************************************************************/
	finish = clock();
	cout << "totally used time is " << static_cast<double>(finish - start)/CLOCKS_PER_SEC <<" S"<< endl;
	cv::Mat out_file;
	radiance.convertTo(out_file, CV_8U, 250.0);
	cv::imwrite("output_image.bmp", out_file);
	cv::namedWindow("input image", cv::WINDOW_NORMAL);
	cv::namedWindow("dehazed image", cv::WINDOW_NORMAL);
	cv::imshow("input image", image_double);
	cv::imshow("dehazed image", radiance);
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}