#define _USE_MATH_DEFINES
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <ctime>
#include <opencv2/opencv.hpp>


int Clamp(int value, int min, int max)
{
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

cv::Mat MedianFilter(cv::Mat img) {
  cv::Mat res = img.clone();
  int width = img.cols;
  int height = img.rows;

  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++) {
      std::vector<int> medR = std::vector<int>(9);
      std::vector<int> medG = std::vector<int>(9);
      std::vector<int> medB = std::vector<int>(9);

      int count = 0;
      for (int l = -1; l <= 1; l++) {
        for (int k = -1; k <= 1; k++)
        {
          if (i == 225)
            std::sort(medR.begin(), medR.end());
          int idX = Clamp(i + k, 0, height - 1);
          int idY = Clamp(j + l, 0, width - 1);

          medR[count] = (int)img.at<cv::Vec3b>(idX, idY)[2];
          medG[count] = (int)img.at<cv::Vec3b>(idX, idY)[1];
          medB[count] = (int)img.at<cv::Vec3b>(idX, idY)[0];
          count++;
        }        
      }

      std::sort(medR.begin(), medR.end());
      std::sort(medG.begin(), medG.end());
      std::sort(medB.begin(), medB.end());
      
      res.at<cv::Vec3b>(i, j)[2] = Clamp(medR[4], 0, 255);
      res.at<cv::Vec3b>(i, j)[1] = Clamp(medG[4], 0, 255);
      res.at<cv::Vec3b>(i, j)[0] = Clamp(medB[4], 0, 255);
    }

  return res;
}

cv::Mat SobelFilter(cv::Mat img, cv::Mat angles) {
  std::vector<std::vector<int>> kernel = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
  std::vector<std::vector<int>> kernelY = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };

  int width = img.cols;
  int height = img.rows;
  cv::Mat res = cv::Mat(height - 2, width - 2, CV_8UC1);
  angles = cv::Mat(height - 2, width - 2, CV_32FC1);

  for (int i = 1; i < height - 1; i++)
    for (int j = 1; j < width - 1; j++) {
      double sumX = 0;
      double sumY = 0;
      
      for (int l = 0; l < 3; l++)
        for (int k = 0; k < 3; k++)
        {
          int x = i + l - 1;
          int y = j + k - 1;
          double temp = 0.299 * (int)img.at<cv::Vec3b>(x, y)[2] + 0.587 * (int)img.at<cv::Vec3b>(x, y)[1] + 0.114 * (int)img.at<cv::Vec3b>(x, y)[0];
          sumX += kernel[l][k] * temp;
          sumY += kernelY[l][k] * temp;
        }

      double sum = Clamp(sqrt(sumX*sumX + sumY*sumY), 0, 255);
      res.at<uchar>(i - 1, j - 1) = sum;

      if (sumX == 0)
        angles.at<float>(i - 1, j - 1) = 90;
      else
        angles.at<float>(i - 1, j - 1) = atan(sumY / sumX);
    }

    return res;
}

cv::Mat NonMax(cv::Mat& img, cv::Mat& angles) {
  int width = img.cols;
  int height = img.rows;
  cv::Mat res = cv::Mat(height, width, CV_8UC1);

  for (int i = 1; i < height - 1; i++) 
    for (int j = 1; j < width - 1; j++) {
      if (j == 226)
        res.at<uchar>(i - 1, j - 1);
      float grad_dir = angles.at<float>(i, j);
      
      res.at<uchar>(i - 1, j - 1) = img.at<uchar>(i, j);

      if ((-22.5 < grad_dir && grad_dir <= 22.5) || (157.5 < grad_dir && grad_dir <= -157.5))
        if ((img.at<uchar>(i, j) < img.at<uchar>(i, j + 1)) || (img.at<uchar>(i, j) < img.at<uchar>(i, j - 1)))
          res.at<uchar>(i - 1, j - 1) = 0;
      
      if ((-112.5 < grad_dir && grad_dir <= -67.5) || (67.5 < grad_dir && grad_dir <= 112.5))
        if ((img.at<uchar>(i, j) < img.at<uchar>(i + 1, j)) || (img.at<uchar>(i, j) < img.at<uchar>(i - 1, j)))
          res.at<uchar>(i - 1, j - 1) = 0;

      if ((-67.5 < grad_dir && grad_dir <= -22.5) || (112.5 < grad_dir && grad_dir <= 157.5))
        if ((img.at<uchar>(i, j) < img.at<uchar>(i - 1, j + 1)) || (img.at<uchar>(i, j) < img.at<uchar>(i + 1, j - 1)))
          res.at<uchar>(i - 1, j - 1) = 0;

      if ((-157.5 < grad_dir && grad_dir <= -112.5) || (22.5 < grad_dir && grad_dir <= 67.5))
        if ((img.at<uchar>(i, j) < img.at<uchar>(i + 1, j + 1)) || (img.at<uchar>(i, j) < img.at<uchar>(i - 1, j - 1)))
          res.at<uchar>(i - 1, j - 1) = 0;
    }

  return res;
}

cv::Mat DoubleThreshold(const cv::Mat& img, int min, int max) {
  min = Clamp(min, 0, 255);
  max = Clamp(max, 0, 255);
  cv::Mat res = img.clone();

  for (int i = 0; i < img.rows; i++)
  {
    for (int j = 0; j < img.cols; j++)
    {
      res.at<uchar>(i, j) = img.at<uchar>(i, j);

      if (res.at<uchar>(i, j) > max)
        res.at<uchar>(i, j) = 255;
      else if (res.at<uchar>(i, j) < min)
        res.at<uchar>(i, j) = 0;
      else 
        res.at<uchar>(i, j) = 100;
    }
  }

  return res;
}

cv::Mat Trace(const cv::Mat& img) {
  cv::Mat res = img.clone();

  for (int i = 0; i < img.rows; i++)
    for (int j = 0; j < img.cols; j++)
    {
      if (res.at<uchar>(i, j) != 100)
        continue;

      bool gotNeighbours = false;
      for (int l = i - 1; l < i + 1; l++)
        for (int k = j - 1; k < j + 1; k++)
        {
          int x = Clamp(l, 0, img.cols);
          int y = Clamp(k, 0, img.rows);
          if (res.at<uchar>(x, y) == 255) {
            res.at<uchar>(i, j) = 255;
            gotNeighbours = true;
            break;
          }
          if (gotNeighbours)
            break;
        }
      if (!gotNeighbours)
        res.at<uchar>(i, j) = 0;
    }

  return res;
}

cv::Mat Canny(cv::Mat img, int thresholdMin, int thresholdMax) {
  cv::Mat res = img.clone();
  cv::Mat angles = cv::Mat(img.rows, img.cols, CV_32FC1);

  res = MedianFilter(res);
  res = SobelFilter(res, angles);
  res = NonMax(res, angles);
  res = DoubleThreshold(res, thresholdMin, thresholdMax);
  res = Trace(res);

  return res;
}

std::vector<std::vector<int>> HoughCircles(cv::Mat img)
{
  int width = img.cols;
  int height = img.rows;
  std::vector<std::vector<int>> circles;

  for (int radius = 10; radius < 70; radius++) {
    std::vector<std::vector<int>> acc = std::vector<std::vector<int>>(height);
    for (int i = 0; i < height; i++) 
      acc[i] = std::vector<int>(width);

    for (int i = 0; i < height; i++) 
      for (int j = 0; j < width; j++) {

        if (img.at<uchar>(i, j) == 255) {
          for (int angle = 0; angle < 360; angle++) {
            int a = i - round(radius * cos(angle * M_PI / 180));
            int b = j - round(radius * sin(angle * M_PI / 180));            
            if ((0 <= a) && (a < height) && (0 <= b) && (b < width))
              acc[a][b]++;
          }
        }
      }

    int max_acc = 0;
    for (int i = 0; i < height; i++)
      for (int j = 0; j < width; j++)
        if (acc[i][j] > max_acc)
          max_acc = acc[i][j];

    if (max_acc > 140) {
      for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
          if (acc[i][j] < 140)
            acc[i][j] = 0;

      for (int i = 1; i < height - 1; i++)
        for (int j = 1; j < width - 1; j++) {
          if (acc[i][j] >= 140) {
            double sum = 0;

            for (int l = i - 1; l < i + 1; l++)
              for (int k = j - 1; k < j + 1; k++)
                sum += acc[l][k];

            sum /= 9;

            if (sum >= 90) {
              circles.push_back({ i, j, radius });
              for (int l = 0; l < 5; l++)
                for (int k = 0; k < 7; k++)
                  acc[i + l][j + k] = 0;
            }
          }
        }
    }
  }

  return circles;
}

void main() {
  cv::Mat img = cv::imread("img.jpg");
  clock_t t1, t2;
  cv::Mat res_canny;

  t1 = clock();
  res_canny = Canny(img, 90, 120);
  t2 = clock();
  std::cout << "My time: " << difftime(t2, t1) << std::endl;
  cv::imshow("res1: ", res_canny);

  t1 = clock();
  cv::Canny(img, res_canny, 90, 120, 3);
  t2 = clock();
  std::cout << "OpenCV time: " << difftime(t2, t1) << std::endl;

  //cv::imshow("res1: ", res_canny);

  img = cv::imread("balls.jpg");
  std::vector<std::vector<int>> circles;

  res_canny = Canny(img, 90, 120);
  t1 = clock();
  circles = HoughCircles(res_canny);
  for (std::vector<int> c : circles) {
    cv::circle(img, cv::Point(c[1], c[0]), c[2], { 40, 255, 80 }, 2);
  }
  t2 = clock();
  std::cout << "My time: " << difftime(t2, t1) << std::endl;
  cv::imshow("res2: ", img);

  std::vector<cv::Vec3f> circ;
  t1 = clock();
  cv::HoughCircles(res_canny, circ, CV_HOUGH_GRADIENT, 1, 10, 140, 90, 10, 70);
  for (std::vector<int> c : circles) {
    cv::circle(img, cv::Point(c[1], c[0]), c[2], { 40, 255, 80 }, 2);
  }
  t2 = clock();
  std::cout << "OpenCV time: " << difftime(t2, t1) << std::endl;

  //cv::imshow("res2: ", img);
  cv::waitKey(0);
}