
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

// CUDA LIBS
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaoptflow.hpp>

using namespace cv;
using namespace cv::cuda;


cv::Mat frame, frame_prev;
cv::cuda::GpuMat gpu_frame, gpu_frame_prev;
int detect_interval = 8;
int frame_idx = 0;

cv::cuda::GpuMat d_p0;
cv::cuda::GpuMat d_p1;
cv::cuda::GpuMat d_p0r;
cv::cuda::GpuMat d_status;
cv::cuda::GpuMat d_err;

int main() {

   std::vector<cv::Vec4i> hierarchy;

	// vector of vectors that stores the tracks (all positions) of each feature point

	std::vector<std::vector<cv::Point2f> > tracks;

	// feature points

	std::vector<cv::Point2f> p0, p1, p0r, p;

	std::vector<KeyPoint> cpuKeys; 

	TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);

	std::vector<uchar> status;
	std::vector<float> err;
	
 	std::vector<Scalar> colors;
 	std::vector<Scalar> new_colors;
	RNG rng;

	GpuMat keypoints;

	cv::VideoCapture video("my_video.mp4");

	if (!video.isOpened()) {
		
        std::cout << "Error: Unable to open file" << std::endl;
        return false;
    }

	cv::Ptr<cv::cuda::ORB> detector = cv::cuda::ORB::create(500, 1.2f, 8, 31, 0, 2, 0, 31, 20, true); 

	while (1) {


		if (!video.read(frame)) {
			break;
		}
	   
		// Crop the image
		
		Rect Rec(430, 530, 1280, 780);     //x,y,width,height
		rectangle(frame, Rec, Scalar(255), 1, 8, 0);

		frame = frame(Rec);

		Mat vis = frame.clone();  // for visualization 

		cv::Rect roi(440,  530, 360, 280);

		rectangle(frame, roi, Scalar(100,100,100),-1);

		// upload frame to GPU

		gpu_frame.upload(frame);

	   // convert to gray

		cv::cuda::cvtColor(gpu_frame, gpu_frame, COLOR_BGR2GRAY);


		if (!tracks.empty()) {

			//track last position of points, in forward and backward direction
			// in order to find points that were predicted correctly

			// start optical flow timer

			p0.clear();

			for (auto i = 0; i < tracks.size(); i++ ) {

				p0.push_back(tracks[i].back());
				
			}

			Mat mat(1, p0.size(), CV_32FC2, (void*)&p0[0]);

			d_p0.upload(mat);


			// create optical flow instance
			Ptr< cuda::SparsePyrLKOpticalFlow > d_pyrLK = cuda::SparsePyrLKOpticalFlow::create(Size(31,31), 2, 30);

			d_pyrLK->calc(gpu_frame_prev, gpu_frame, d_p0, d_p1, d_status, d_err);

			d_pyrLK->calc(gpu_frame, gpu_frame_prev, d_p1, d_p0r, d_status, d_err);

			p1.resize(d_p1.size().width);
			if (!p1.empty())
				d_p1.row(0).download(cv::Mat(p1).reshape(2, 1));


			p0r.resize(d_p0r.size().width);
			if (!p0r.empty())
				d_p0r.row(0).download(cv::Mat(p0r).reshape(2, 1));

			d_status.row(0).download(status);


			//  keep only good matches

			// The absolute difference is calculated between initial points (p0) and the backwards predicted points (p0r). 
			// if the value is below one then the points were predicted correctly, if not it is a "bad" point.

			std::vector<int> good;

			good.clear();

			for (auto i = 0; i < p0.size(); i++) {

				double dist = max(abs(p0[i].x-p0r[i].x), abs(p0[i].y-p0r[i].y));

				if ((dist < 1) && (status[i])) {
			
					good.push_back(1);
				}
				else {
					good.push_back(0);
				}

			}
			
			// append only new good locations to existing tracked points

			std::vector<std::vector<cv::Point2f> > new_tracks;

			new_tracks.clear();
			new_colors.clear();

			for (auto i = 0; i < tracks.size(); i++) {

				if (!good[i]) {
					continue;
				}

				new_tracks.push_back(tracks[i]);
				new_colors.push_back(colors[i]);
			}


			int k=0;

			for (int j = 0; j < p1.size(); j++) {

				if (!good[j]) {
					continue;
				}

				new_tracks[k].push_back(p1[j]);

				k++;
			}

			tracks = new_tracks;

			colors = new_colors;

				// draw latest points and their tracks 

			for (auto i = 0; i < tracks.size(); i++) {

				circle(vis, tracks[i].back(), 3, Scalar(0,0,0), -1);

					for (auto j = 0; j < tracks[i].size()-1; j++) {

					line(vis, tracks[i][j], tracks[i][j+1], colors[i], 1.5);
					}
			}

			auto end_opt_flow_time2 = high_resolution_clock::now();

		}

			// display number of tracked points

		putText(vis, format("track count: %ld", tracks.size()), cv::Point(10, vis.rows / 2), cv::FONT_HERSHEY_DUPLEX,
            1.0,
            CV_RGB(0, 255, 0), //font color
           2);

		// Every 'detect_interval' frames, the contours get updated

		if ((frame_idx % detect_interval == 0) || (tracks.empty())) {

			detector->detectAsync(gpu_frame, keypoints);

			p.clear();

			detector->convert(keypoints, cpuKeys); 

			cv::KeyPoint::convert(cpuKeys, p);

			std::vector<cv::Point2f> v1;
			std::vector<cv::Point2f> neighbor_p;
			std::vector<cv::Point2f> p_new;
			std::vector<cv::Point2f> p_unique;
			cv::Point2f mo_point;
			double mo_x, mo_y;
			double dist1;
			int match;

			p_unique.clear();

			// Keep only unique new points

			if (!p.empty()) {

				if (!tracks.empty()) {

					for (auto i = 0; i<p.size(); i++) {

						match = 0;

						for (auto j = 0; j<tracks.size(); j++) {

							dist1 = cv::norm(p[i]-tracks[j].back());
       
							if (dist1 < 13) {

								match = 1;

								break;
							}
						}

						if (match == 0) {

							p_unique.push_back(p[i]);

						}
					}
				}
				else {

					p_unique = p;
				}
			}

			neighbor_p.clear();
			p_new.clear();

			int a=0;
			float dist2;

			// Group points that form clusters

			if (!p_unique.empty()) {

				for (auto i = p_unique.begin(); i != p_unique.end(); i++) {


					neighbor_p.clear();

					if (((*i).x != 0) && ((*i).y != 0)) {


						for (auto j = p_unique.begin(); j != p_unique.end(); j++) {

							if ((j != i) && (((*j).x != 0) && ((*j).y != 0)))	{

								dist2 = cv::norm(*i-*j);
							}
							else {
								
								continue;
							}
	                 
							//osa exoun mikrh apostasi me to shmeio, apothikeyontai kai diagrafontai apo th lista

							if (dist2 < 50) {

								neighbor_p.push_back(*i);

								neighbor_p.push_back(*j);

								(*j).x = 0;
								(*j).y = 0;
							}
						}

						if (neighbor_p.size() > 0) {

							Rect box = boundingRect(neighbor_p); 
							rectangle(imageROI, box, Scalar(0,0,255),1,8,0);

							mo_x = 0;
							mo_y = 0;
							
							for (auto k = 0; k<neighbor_p.size(); k++) {


								mo_x = mo_x + neighbor_p[k].x;
								mo_y = mo_y + neighbor_p[k].y;
							}

							mo_point.x = mo_x / neighbor_p.size();
							mo_point.y = mo_y / neighbor_p.size();


							p_new.push_back(mo_point);
						}
						else {

							p_new.push_back(*i);

							(*i).x = 0;
							(*i).y = 0;
						}
					}
				}
			}

			// Append new set of points 

			if (!p_new.empty()) {

				for (auto i = 0; i<p_new.size(); i++) {

					v1.push_back(p_new[i]);
					tracks.push_back(v1);

					// Create a random color

					int r = rng.uniform(0, 256);
			        int g = rng.uniform(0, 256);
			        int b = rng.uniform(0, 256);
			        colors.push_back(Scalar(r,g,b));

					v1.clear();
				}
			}

		}

		frame_idx++;
		gpu_frame_prev = gpu_frame;
		
    	//imshow("Optical flow tracks", vis);

		int key = waitKey(1);

      if (key == 'q' || key == 27)
          break;

      if (key == ' ')    //pause video
        	waitKey(-1);

	}


}


