#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

                                                                             /*a demo by liuchao*/

struct pics
{
     //cv::Point2f size[2];
    float x,y,area;
};


int length(float x,float y)
{
    int l;
    l = sqrt(abs(x*x) + abs(y*y));
    return l;

}

class picture_process
{
    public:
        Mat picbase_process(Mat src);
        Mat hough_detect(Mat pic,Mat src);
        Mat rec_detect(Mat mask,Mat src);
        Mat change_size(string path);
};


Mat picture_process::change_size(string path)
{
    Mat src = cv::imread(path);
    cv::resize(src,src,Size(0,0),0.5,0.5,INTER_AREA);
    return src;
}


Mat picture_process:: picbase_process(Mat src)
{
    int iterations = 3;    //膨胀次数3次
    int g_nStructElementSize = 3; //结构元素(内核矩阵)的尺寸
    Mat kernel = getStructuringElement(MORPH_RECT,Size(g_nStructElementSize,g_nStructElementSize));//3*3的卷积核
    Mat hsv,mask;

    medianBlur(src,src,5);
    cvtColor(src,hsv,COLOR_BGR2HSV);
    inRange(hsv,Scalar(35,45,45),Scalar(80,255,255),mask);//提取绿色
    medianBlur(mask,mask,5);
    dilate(mask,mask,kernel,Point(1,1), 1);//白色区域膨胀1次
    GaussianBlur(mask,mask,Size(1,1),1.5);//高斯滤波
    mask = 255 -mask;//黑白互换
    dilate(mask,mask,kernel,Point(1,1), iterations);//白色区域膨胀3次
    erode(mask,mask,kernel);//白色区域腐蚀

    return mask;

}

Mat picture_process::hough_detect(Mat pic,Mat src)
{
     std::cout << "hough process open" << std::endl;
     //float x[20];
     vector<Vec3f> circles;
     HoughCircles(pic, circles, HOUGH_GRADIENT,  1, 10, 50, 14, 0, 85);
     float x[circles.size()];
     float y[circles.size()];

     for (size_t i = 0; i < circles.size(); i++)
         {
             Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));   //圆心
             int radius = cvRound(circles[i][2]);                            //半径
             //绘制圆心
             circle(src, center, 3, Scalar(0, 0, 255), -1); //red
             //绘制圆轮廓
             circle(src, center, radius, Scalar(0, 0, 255), 10); //red
             x[i] = float(center.x);
             y[i] = float(center.y);
             std::cout << "圆的坐标是 " << "x " << center.x << " y: " << center.y << std::endl;
         }
        std::cout << "the number of circle is " << circles.size() << std::endl;
        float l1,l2,l3;
        l1 =length(x[3]-x[1],y[3]-y[1]);
        l2 =length(x[1]-x[0],y[1]-y[0]);
        l3 =length(x[0]-x[2],y[0]-y[2]);
        std::cout << "第1个圆和第2个圆球的距离是 " << l1 << std::endl;
        std::cout << "第2个圆和第3个圆球的距离是 " << l2 << std::endl;
        std::cout << "第3个圆和第4个圆球的距离是 " << l3 << std::endl;
        //cv::line(src,Point(x[3],y[3]),Point(x[1],y[1]),Scalar(0,255,255),5);
        //cv::line(src,Point(x[1],y[1]),Point(x[0],y[0]),Scalar(255,255,0),5);
        //cv::line(src,Point(x[3],y[3]),Point(x[2],y[2]),Scalar(255,0,255),5);


        cv::imshow("Hough", src);
        //cv::waitKey(0);
        return src;
}

Mat picture_process::rec_detect(Mat mask,Mat src)
{
    int g_nStructElementSize = 7; //结构元素(内核矩阵)的尺寸
    Mat kernel = getStructuringElement(MORPH_RECT,Size(g_nStructElementSize,g_nStructElementSize));//7*7的卷积核
    int inter_number = 0;

    erode(mask,mask,kernel);//白色区域腐蚀
    erode(mask,mask,kernel);//白色区域腐蚀
    std:: cout <<  "the rec detect open " << std::endl;
    vector<vector<Point>> contours;
    vector<Vec4i> hierachy;
    cv::findContours(mask, contours, hierachy, RETR_TREE, cv::CHAIN_APPROX_NONE, Point(-1,-1));
    for(int i = 0;i<contours.size();i++)
    {
        float area = contourArea(contours[i]);
        if(area > 200 | area <=50)
            continue;
        else
          {
            cv::RotatedRect rect = cv::minAreaRect(contours[i]);
            cv::Point2f points[4];
            rect.points(points);
            cv::Point2f center = rect.center;
            for(int n = 0;n < 4;n++)
            {
                if(n == 3)
                {
                    cv::line(src,points[n],points[0],Scalar(255,0,0),2,8,0);
                    break;
                }
                cv::line(src,points[n],points[n+1],Scalar(255,0,0),2,8,0);
            }
            cv::circle(src,center,2,Scalar(0,0,255),2,8,0);
            inter_number ++;

          }
    }
    std::cout << " 内部矩形的数量是 " << inter_number <<std::endl;

    return src;

}

int main()
{
  std::cout << "This is a cvdemo" << std::endl;
  picture_process pic_procession;
  pics area_picture;
  string path ="/home/next/picture1.png";
  string path_2 = "/home/next/picture2.png";

  /*一号图片*/
  Mat picture,hough,circle;
  circle = pic_procession.change_size(path);
  picture = pic_procession.picbase_process(circle);
  hough = pic_procession.hough_detect(picture,circle);

  /*二号图片*/
  Mat pic,rectang_,rec,area1;
  area1 = cv::imread(path_2);
  area_picture.x = area1.cols; //width
  area_picture.y = area1.rows; //height
  std::cout << "the size of picture2 is " << "width " << area_picture.x << " height " << area_picture.y << "  and area is " << area1.cols * area1.rows << std::endl;

  rec = pic_procession.change_size(path_2);
  pic = pic_procession.picbase_process(rec);
  rectang_ = pic_procession.rec_detect(pic,rec);
  cv::imshow("rec",rectang_);
  cv::waitKey(0);

  return 0;
}
