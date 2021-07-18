
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/tracking.hpp>
#include <cmath>
#include <stdlib.h>  
#include <unordered_map>
#include <chrono>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudafilters.hpp>
#include <npp.h>
#include <nppi.h>
#include <npps.h>
#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <nppi_morphological_operations.h>

using namespace cv;
using namespace std::chrono;
using namespace cv::cuda;

cv::Mat frame;
cv::cuda::GpuMat gpu_frame;
cv::Mat img;


int main() {

	int erosion_size = 3;

	// CPU ERODE

	Mat src, erosion_dst;

	src = imread("img1.jpeg", IMREAD_COLOR);

	auto start_cpu = high_resolution_clock::now();

	Mat element = getStructuringElement( MORPH_RECT,
	               Size( 2*erosion_size + 1, 2*erosion_size+1 ));

	erode(src, erosion_dst, element);

	auto end_cpu = high_resolution_clock::now();

	imwrite("cpuresult.jpeg", erosion_dst);

	printf("CPU dilate time: %f\n", duration_cast<milliseconds>(end_cpu - start_cpu).count() / 1000.0);


	// NPP + OPENCV Erode

	Mat img = imread("img1.jpeg", IMREAD_COLOR); //CV_LOAD_IMAGE_GRAYSCALE

	cv::cvtColor(img, img, COLOR_BGR2GRAY);

	size_t imgsize = img.step[0] * img.rows;

	unsigned char *dSrc, *dDst;

	cudaMalloc((void**)&dSrc,imgsize);
	cudaMalloc((void**)&dDst,(img.cols-2)*(img.rows-2));

	cudaMemset(dDst, 0, (img.cols-2)*(img.rows-2));

	//Copy Data From img to Device Pointer
	cudaMemcpy(dSrc, img.data, imgsize,cudaMemcpyHostToDevice);

	auto start_npp = high_resolution_clock::now();

	cv::Mat hostKernel = getStructuringElement(MORPH_RECT,  Size( 2*erosion_size + 1, 2*erosion_size+1 ));

	size_t kernelsize = hostKernel.step[0] * hostKernel.rows;

	unsigned char *pKernel;

	cudaMalloc((void**)&pKernel, kernelsize);
	cudaMemcpy(pKernel, hostKernel.data, kernelsize, cudaMemcpyHostToDevice);

	NppiSize oMaskSize = {hostKernel.cols, hostKernel.rows};

	NppiPoint oAnchor;
	oAnchor.x = 1; //(oMaskSize.width-1)/2;
	oAnchor.y = 1; //(oMaskSize.height-1)/2;//0;

	//NppiSize oSizeROI = {img.cols - oMaskSize.width + (oMaskSize.width-1)/2, img.rows - oMaskSize.height + (oMaskSize.height-1)/2};
	NppiSize oSizeROI;
	oSizeROI.width = img.cols - 2;
	oSizeROI.height = img.rows - 2;

   NppStatus eStatusNPP;
	eStatusNPP = nppiErode_8u_C1R(dSrc+img.cols+1, img.cols, dDst, oSizeROI.width, oSizeROI, pKernel, oMaskSize, oAnchor);

	auto end_npp = high_resolution_clock::now();
	
	//Negative return codes indicate errors, positive return codes indicate warnings, a return code of 0 indicates success. 
	printf("Erode error: %i \n", eStatusNPP);

	cv::Mat imgout(img.rows - 2, img.cols - 2,CV_8UC1);

	for (int i = 0; i < (img.rows - 2)*(img.cols - 2); i++) {
   	imgout.data[i] = 0; }
   
    //Copy back the result from device to IplImage
   cudaError_t code = cudaMemcpy(imgout.data,dDst,(img.cols-2)*(img.rows-2),cudaMemcpyDeviceToHost);

   if (code != cudaSuccess) 
   {
      fprintf(stderr,"error: %s\n", cudaGetErrorString(code));
      if (abort) exit(code);
   }

   imwrite("nppresult.jpeg", imgout);

   cudaFree(dSrc);
   cudaFree(dDst);
   cudaFree(pKernel);

	printf("NPP dilate time: %f\n", duration_cast<milliseconds>(end_npp - start_npp).count() / 1000.0);


	///////////////////////////////////////////////////////////////////////////////////////////////////////

	// CUDA OPENCV EROSION

   //auto start_cuda = high_resolution_clock::now();

   Mat image1 = imread("img1.jpeg", IMREAD_COLOR);

	// upload frame to GPU

    cuda::GpuMat image1_gpu, image2_gpu;

	image1_gpu.upload(image1);

	cv::cuda::cvtColor(image1_gpu, image1_gpu, COLOR_BGR2GRAY);

	auto start_cuda = high_resolution_clock::now();

	Mat element_4_gpu = getStructuringElement(MORPH_RECT,  Size( 2*erosion_size + 1, 2*erosion_size+1 ));

	Ptr<cuda::Filter> erodeFilter = cuda::createMorphologyFilter(MORPH_ERODE, image1_gpu.type(), element_4_gpu);

	erodeFilter->apply(image1_gpu, image2_gpu);

	auto end_cuda = high_resolution_clock::now();

	image2_gpu.download(image1);
	
   // auto end_cuda = high_resolution_clock::now();

   printf("OPENCV CUDA dilate time: %f\n", duration_cast<milliseconds>(end_cuda - start_cuda).count() / 1000.0);

   imwrite("cudaresult.jpeg", image1);

}


