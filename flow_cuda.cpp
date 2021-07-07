
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
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaoptflow.hpp>

using namespace cv;
using namespace std::chrono;
using namespace cv::cuda;


cv::Mat frame, frame_prev;
cv::cuda::GpuMat gpu_frame, gpu_frame_prev;
cv::Mat img;
cv::Mat imageROI;
int applyCLAHE = 0;
int roi_top = 1560;
int roi_bottom = 1560;
int roi_left = 2104;
int roi_right = 2104;
int opening_elem = 0;
int closing_elem = 0;
int opening_size = 0;
int closing_size = 0;
int detect_interval = 8;
int frame_idx = 0;
int track_len = 15;

cv::cuda::GpuMat d_p0;
cv::cuda::GpuMat d_p1;
cv::cuda::GpuMat d_p0r;
cv::cuda::GpuMat d_status;
cv::cuda::GpuMat d_err;

 // to track time for every stage at each iteration
 std::vector<std::vector<double> > timers;


int main() {

	cv::VideoCapture input("NIR_1.mp4");
		
   	std::vector<cv::Vec4i> hierarchy;
	
	int desiredWidth=640, desiredheight=480; //window sizes

	// vector of vectors that stores the tracks (all positions) of each feature point

	std::vector<std::vector<cv::Point2f> > tracks;

	// feature points

	std::vector<cv::Point2f> p0, p1, p0r, p;

    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);

	std::vector<uchar> status;
	std::vector<float> err;
	
	cv::VideoCapture video("NIR_1.mp4");
	
	//namedWindow("Contours", WINDOW_NORMAL);
	//resizeWindow("Contours", desiredWidth, desiredheight);

	namedWindow("Optical flow tracks", WINDOW_NORMAL);
	resizeWindow("Optical flow tracks", desiredWidth, desiredheight);
	
	// Pre-processing values 

	applyCLAHE = 1;
	opening_elem = 1;
	opening_size = 7;
	closing_elem = 1;
	closing_size = 5;
	
   	std::vector<Scalar> colors;
   	std::vector<Scalar> new_colors;
    RNG rng;

    std::vector<double> v2;

    v2.push_back(0);

    timers.push_back(v2);
    timers.push_back(v2);
    timers.push_back(v2);
    timers.push_back(v2);
    timers.push_back(v2);

    // start full pipeline timer

	auto start_full_time = high_resolution_clock::now();
	
	for (;;) {

		// start reading timer

		auto start_read_time = high_resolution_clock::now();
	
		if (!video.read(img))
			break;

		// end reading timer

	    auto end_read_time = high_resolution_clock::now();

	    // add elapsed iteration time

	    timers[0].push_back(duration_cast<milliseconds>(end_read_time - start_read_time).count() / 1000.0);


	    // start pre-process timer

	    auto start_pre_time = high_resolution_clock::now();
		
	
		// Crop the image
		
		Rect Rec(430, 530, 1280, 800);     //x,y,width,height
		rectangle(img, Rec, Scalar(255), 1, 8, 0);

		imageROI = img(Rec);

		frame = imageROI.clone();

		// upload frame to GPU

        cv::cuda::GpuMat gpu_frame;
        gpu_frame.upload(frame);

        // convert to gray

		cv::cuda::cvtColor(gpu_frame, gpu_frame, COLOR_BGR2GRAY);
		
		// Perform the operations
		
		if (applyCLAHE) {

			Ptr<cuda::CLAHE> clahe = cv::cuda::createCLAHE();

			clahe->setClipLimit(5);

			clahe->apply(gpu_frame, gpu_frame);
		}
	
		// Otsu Threshold
		
		cv::cuda::threshold(gpu_frame, gpu_frame, 0, 255, THRESH_BINARY | THRESH_OTSU);
		
		// Morphological Opening
		
		Mat element3 = getStructuringElement( opening_elem, Size( 2*opening_size + 1, 2*opening_size+1 ), Point( opening_size, opening_size ) );

		Ptr<cuda::Filter> openFilter = cuda::createMorphologyFilter(MORPH_OPEN, gpu_frame.type(), element3);
   		openFilter->apply(gpu_frame, gpu_frame);
		
		
		// Morphological Closing
		
		Mat element4 = getStructuringElement( closing_elem, Size( 2*closing_size + 1, 2*closing_size+1 ), Point( closing_size, closing_size ) );

		Ptr<cuda::Filter> closeFilter = cuda::createMorphologyFilter(MORPH_CLOSE, gpu_frame.type(), element4);
   		closeFilter->apply(gpu_frame, gpu_frame);

		Mat vis = imageROI.clone();

		// end pre-process timer

	    auto end_pre_time = high_resolution_clock::now();

	    // add elapsed iteration time

	    timers[1].push_back(duration_cast<milliseconds>(end_pre_time - start_pre_time).count() / 1000.0);


		// Calculate optical flow

		// start optical flow timer

		auto start_of_time = high_resolution_clock::now();


		if (!tracks.empty()) {

			//track last position of points, in forward and backward direction
			// in order to find points that were predicted correctly

			p0.clear();

			for (auto i = 0; i < tracks.size(); i++ ) {

				p0.push_back(tracks[i].back());
				
			}

			// start optical flow timer

			auto start_opt_flow_time = high_resolution_clock::now();

			d_p0(p0.getMat().reshape(2, 1));

			// create optical flow instance
            Ptr< cuda::SparsePyrLKOpticalFlow > d_pyrLK = cuda::SparsePyrLKOpticalFlow::create(Size(31,31), 2, 30);

            d_pyrLK->calc(gpu_frame_prev, gpu_frame, d_p0, d_p1, status, err);

            cv::Mat& nextPtsRef = p1.getMatRef();
            d_p1.download(p1);
            nextPtsRef = nextPtsRef.t(); //revert the matrix to its actual shape

            d_status.download(status);
            d_err.download(err);


            d_pyrLK->calc(gpu_frame, gpu_frame_prev, d_p1, d_p0r, status, err);

            cv::Mat& nextPtsRef = p0r.getMatRef();
            d_p0r.download(p0r);
            nextPtsRef = nextPtsRef.t(); //revert the matrix to its actual shape

			
			// end post pipeline timer

			auto end_opt_flow_time = high_resolution_clock::now();

			// add elapsed iteration time

			timers[2].push_back(duration_cast<milliseconds>(end_opt_flow_time - start_opt_flow_time).count() / 1000.0);


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
		}

		// display number of tracked points

		putText(vis, format("track count: %ld", tracks.size()), cv::Point(10, vis.rows / 2), cv::FONT_HERSHEY_DUPLEX,
            1.0,
            CV_RGB(0, 255, 0), //font color
           2);


		// Every 'detect_interval' frames, the contours get updated


		if ((frame_idx % detect_interval == 0) || (tracks.empty())) {

			findContours(frame, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
			
			// Compute the centers of the contours/blobs

			std::vector<cv::Moments> mu(contours.size());

			for (int i = 0; i<contours.size(); i++ ) {
				mu[i] = moments( contours[i], false ); 
			}

			// Get the centroid of figures 

			std::vector<cv::Point2f> mc(contours.size());

			for (int i = 0; i<contours.size(); i++) { 

				if (mu[i].m00 != 0) {
					mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
				}
			}

			// Make centroids feature points

			p.clear();

			for (int i = 0; i<contours.size(); i++) {

				p.push_back(mc[i]);

			}

			std::vector<cv::Point2f> v1;
			std::vector<cv::Point2f> neighbor_p;
			std::vector<cv::Point2f> p_new;
			std::vector<cv::Point2f> p_unique;
			cv::Point2f mo_point;
			double mo_x, mo_y;
			double dist1;
			int match;

			p_unique.clear();

			// Nea points sygkrish me ta palia wste na mhn yparxoyn dipla

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

			// Group ta kontina points

			if (!p_unique.empty()) {

				//gia kathe point

				for (auto i = p_unique.begin(); i != p_unique.end(); i++) {

					//ypologise thn apostash me ta ypoloipa

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


						// pare to meso oro tvn shmeivn kai ftiakse 1 monadiko shmeio pou antiprosopeyei to group
	         
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

        // Next iteration/frame

        frame_idx++;
        frame_prev = frame.clone();

        // end optical flow timer

		auto end_of_time = high_resolution_clock::now();

		// add elapsed iteration time

		timers[3].push_back(duration_cast<milliseconds>(end_of_time - start_of_time).count() / 1000.0);


        imshow("Optical flow tracks", vis);
		

		int key = waitKey(1);

        if (key == 'q' || key == 27)
            break;

        if (key == ' ')    //pause video
        	waitKey(-1);

         // end optical flow timer

	}

	// end full pipeline timer

	auto end_full_time = high_resolution_clock::now();

	// add elapsed iteration time

	timers[4].push_back(duration_cast<milliseconds>(end_full_time - start_full_time).count() / 1000.0);
		
	// elapsed time at each stage

	printf("Elapsed time\n");

	for (auto k = 0; k < timers.size(); k++) {

		if (k == 0) {
			printf("Reading ");
		}
		else if (k == 1) {
			printf("Pre-process ");
		}
		else if (k == 2) {
			printf("Actual optical flow ");
		}
		else if (k == 3) {
			printf("Optical flow ");
		}
		else if (k == 4) {
			printf("Full pipeline ");
		}

		double timer=0;

		for (auto j = 0; j< timers[k].size(); j++) {

			timer = timer + timers[k][j];

		}
 
		printf("%f seconds\n", timer);

	}


}


