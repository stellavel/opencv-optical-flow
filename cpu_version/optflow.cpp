
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/tracking.hpp>
#include <stdlib.h>

using namespace cv;
using namespace std;


int main() {

	cv::Mat frame, frame_prev;
	int opening_size = 5;
	int closing_size = 5;
	int detect_interval = 8;
	int frame_idx = 0;	

	std::vector<std::vector<cv::Point> > contours;
   	std::vector<cv::Vec4i> hierarchy;

	std::vector<std::vector<cv::Point2f> > tracks;

	std::vector<cv::Point2f> p0, p1, p0r, p;
	std::vector<uchar> status;
	std::vector<float> err;
	std::vector<int> good;
	
   	std::vector<Scalar> colors;
   	std::vector<Scalar> new_colors;
    RNG rng;

    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);


	cv::VideoCapture video("my_video.mp4");

	if (!video.isOpened()) {
		
        std::cout << "Error: Unable to open file" << std::endl;
        return false;
    }
	

	while (1) {
	
		if (!video.read(frame))
			break;


		// ***********Image Pre-Processing Operations***********


		// Crop frame
		
		Rect Rec(430, 530, 1280, 780);     //x,y,width,height
		rectangle(frame, Rec, Scalar(255), 1, 8, 0);

		frame = frame(Rec);

		Mat vis = frame.clone();  // for visualization

		// Add mask

		cv::Rect roi(440,  530, 360, 280);

		rectangle(frame, roi, Scalar(100,100,100),-1);

		// Perform operations

		cv::cvtColor(frame, frame, COLOR_BGR2GRAY);
		
		// CLAHE

		Ptr<CLAHE> clahe = createCLAHE();

		clahe->setClipLimit(5);

		clahe->apply(frame, frame);
	
		// Otsu Threshold
		
		threshold(frame, frame, 0, 255, THRESH_BINARY | THRESH_OTSU);
		
		// Morphological Opening

		Mat open_element = getStructuringElement(MORPH_CROSS, Size(2*opening_size+1, 2*opening_size+1), Point(opening_size, opening_size));

		morphologyEx(frame, frame, MORPH_OPEN, open_element);

		// Morphological Closing

		Mat close_element = getStructuringElement(MORPH_CROSS, Size(2*closing_size+1, 2*closing_size+1), Point(closing_size, closing_size));

		morphologyEx(frame, frame, MORPH_CLOSE, close_element);


		if (!tracks.empty()) {

			
			// ***********Calculate optical flow***********


			//track last position of points, in forward and backward direction
			// in order to find points that were predicted correctly

			p0.clear();

			for (auto i = 0; i < tracks.size(); i++ ) {

				p0.push_back(tracks[i].back());
				
			}

			calcOpticalFlowPyrLK(frame_prev, frame, p0, p1, status, err, Size(31,31), 2, criteria);

			calcOpticalFlowPyrLK(frame, frame_prev, p1, p0r, status, err, Size(31,31), 2, criteria);


			//  keep only good matches

			// The absolute difference is calculated between initial points (p0) and the backwards predicted points (p0r). 
			// if the value is below one then the points were predicted correctly, if not it is a "bad" point.
			
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


			// ***********Detect objects/contours***********


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
	               
							if (dist2 < 50) {

								neighbor_p.push_back(*i);

								neighbor_p.push_back(*j);

								(*j).x = 0;
								(*j).y = 0;
							}
						}

						if (neighbor_p.size() > 0) {

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

        // Visualization
       
        imshow("Optical flow tracks", vis);

		int key = waitKey(1);

        if (key == 'q' || key == 27)
            break;

        if (key == ' ')    //pause video
        	waitKey(-1);
	}

	destroyAllWindows();

}


