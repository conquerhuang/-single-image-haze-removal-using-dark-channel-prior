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
const double omega{ 0.9 };//������ԭͼ���������̶�,��һЩ����£��ò��������û�Ӱ��ȥ��Ч������Ҫ����ͼ������Ӧ

int main(void)
{
	cv::Mat input_image;
	input_image = cv::imread("2.jpg");
	if (input_image.empty())//�ж�ͼ���Ƿ��ȡ�ɹ�
	{
		cout << "error in get image" << endl;
		return -1;
	}

	clock_t start{ 0 }, finish{ 0 };
	start = clock();

	cv::Mat image_double;//���double���͵�ͼ�񣬷�ֹ�����������
	input_image.convertTo(image_double, CV_32F, 1.0 / 255, 0);//ת��Ϊ����ͼ�������������������

	/*************************************��ȡ��ͨ��***********************************/
	cv::Mat dark_channel;
	get_darkchannel(image_double, dark_channel, win_size);
	/*********************************************************************************/

	/**************************�ɰ�ͨ����ԭͼ���������*********************************/
	cv::Vec3f atmosphere{ 0,0,0 };//���ڴ洢������ֵ������
	get_atmosphere(image_double, dark_channel, atmosphere);//ͨ��������ȡ������
	/*********************************************************************************/

	/*********************************������մ�������ͼ*******************************/
	cv::Mat trans_est;
	get_esttransmap(image_double,atmosphere,omega,win_size,trans_est);
	/*********************************************************************************/
	
	/***************************���õ����˲�������������ͼ******************************/
	cv::Mat trans_map;//���մ���ͼ����
	cv::Mat image_gray;//ԭͼ��Ӧ�ĻҶ�ͼ
	cv::cvtColor(image_double, image_gray, CV_BGR2GRAY);
	//���õ����˲����㴫�ݾ���
	trans_map = fastGuidedFilter(image_gray, trans_est, 15, 0.001, 1);;
	/*********************************************************************************/

	/********************************�������ͼ��**************************************/
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