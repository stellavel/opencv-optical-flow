

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

using namespace cv;

cv::Mat frame, frame_prev;
cv::Mat img;
cv::Mat imageROI;
Mat erosion_dst, dilation_dst;
int applyCLAHE = 0;
int threshMin = 0;
int roi_top = 1560;
int roi_bottom = 1560;
int roi_left = 2104;
int roi_right = 2104;
int runbutton = 0;
int boxselect = 0;
int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 0;
int opening_elem = 0;
int closing_elem = 0;
int opening_size = 0;
int closing_size = 0;
int detect_interval = 5;
int frame_idx = 0;
int track_len = 15;
int const max_elem = 2;
int const max_kernel_size = 21;


int main() {

	cv::VideoCapture input("NIR_1.mp4");
		
	std::vector<std::vector<cv::Point> > contours, contours_prev;
   	std::vector<cv::Vec4i> hierarchy;
	
	int desiredWidth=640, desiredheight=480;
	

	std::vector<std::vector<cv::Point2f> > tracks;

	std::vector<cv::Point2f> p0, p1, p0r, p;

	std::vector<float> d;

	std::vector<uchar> status;
	std::vector<float> err;
	std::vector<int> good;
	
	cv::VideoCapture video("NIR_1.mp4");
	
	cv::destroyAllWindows ();
	
	//namedWindow("Contours", WINDOW_NORMAL);
	//resizeWindow("Contours", desiredWidth, desiredheight);

	namedWindow("Optical flow tracks", WINDOW_NORMAL);
	resizeWindow("Optical flow tracks", desiredWidth, desiredheight);

	//namedWindow("Original", WINDOW_NORMAL);
	//resizeWindow("Original", desiredWidth, desiredheight);

	
	int erosion_type = 0;
	int dilation_type = 0;
	
	// Print values used

	applyCLAHE = 1;
	erosion_elem = 0; 
	erosion_size = 0;
	dilation_elem = 0;
	dilation_size = 0;
	opening_elem = 1;
	opening_size = 5;
	closing_elem = 1;
	closing_size = 8;
	
	printf("CLAHE: %d \n", applyCLAHE);
	printf("Threshold: %d \n", threshMin);
	printf("Erosion type & size: %d %d \n", erosion_elem, erosion_size);
	printf("Dilation typ e& size: %d %d \n", dilation_elem, dilation_size);
	printf("Opening type & size: %d %d \n", opening_elem, opening_size);
	printf("Closing type & size: %d %d \n", closing_elem, closing_size);

	
	
   	std::vector<Scalar> colors;
   	std::vector<Scalar> new_colors;
    RNG rng;
    

    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
	
	for (;;) {

		double e1 = getTickCount();
	
		if (!video.read(img))
			break;
		
		
		// Crop the image

		
		Rect Rec(430, 530, 1280, 800);     //x,y,width,height
		rectangle(img, Rec, Scalar(255), 1, 8, 0);

		imageROI = img(Rec);

		frame = imageROI.clone();

		cv::cvtColor(frame, frame, COLOR_BGR2GRAY);

		
		// Perform the selected operations
		
		if (applyCLAHE) {

			Ptr<CLAHE> clahe = createCLAHE();

			clahe->setClipLimit(5);

			clahe->apply(frame, frame);
		}
	
	   	
		// Otsu Threshold
		
		threshold(frame, frame, 0, 255, THRESH_BINARY | THRESH_OTSU);

		cv::Rect roi2(440,  530, 340, 280);

		rectangle(frame, roi2, Scalar(0,0,0),-1);

		// Morphological Erosion

		if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
		else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
		else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
		Mat element1 = getStructuringElement( erosion_type,
		       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
		       Point( erosion_size, erosion_size ) );
		erode( frame, frame, element1 );

		// Morphological Dilation	

		if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
		else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
		else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }
		Mat element2 = getStructuringElement( dilation_type,
		Size( 2*dilation_size + 1, 2*dilation_size+1 ),
		Point( dilation_size, dilation_size ) );
		dilate( frame, frame, element2 );
		
		// Morphological Opening

		Mat element3 = getStructuringElement( opening_elem, Size( 2*opening_size + 1, 2*opening_size+1 ), Point( opening_size, opening_size ) );

		morphologyEx(frame, frame, MORPH_OPEN, element3);


		// Morphological Closing

		Mat element4 = getStructuringElement( closing_elem, Size( 2*closing_size + 1, 2*closing_size+1 ), Point( closing_size, closing_size ) );

		morphologyEx(frame, frame, MORPH_CLOSE, element4);

		Mat vis = imageROI.clone();


		// Calculate optical flow


		if (!tracks.empty()) {

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

			d.clear();
			good.clear();

			for (auto i = 0; i < p0.size(); i++) {

				circle(imageROI, p0[i], 3, Scalar(0,0,0), -1);
			}

			for (auto i = 0; i < p0.size(); i++) {

				double dist = max(abs(p0[i].x-p0r[i].x), abs(p0[i].y-p0r[i].y));

				if ((dist < 1) && (status[i])) {

				//if ((norm(p0[i] - p0r[i]) < 18) && (status[i])) {
				//continue;
			
					good.push_back(1);

					circle(imageROI, p0r[i], 3, Scalar(0,0,255), -1);
				}
				else {
					good.push_back(0);

					circle(imageROI, p0r[i], 3, Scalar(255,0,0), -1);
				}

				line(vis, p0[i], p0r[i], colors[i], 1.5);

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

		imshow("Contours", imageROI);

		int keyboard = waitKey(0);

		// display number of tracked points

		putText(vis, format("track count: %ld", tracks.size()), cv::Point(10, vis.rows / 2), cv::FONT_HERSHEY_DUPLEX,
            1.0,
            CV_RGB(0, 255, 0), //font color
           2);


		// Every 'detect_interval' frames, the contours get updated

		
		if ((frame_idx % detect_interval == 0) || (tracks.empty())) {

			/*if (!tracks.empty()) {

            	//try to find new points by masking-out last track positions

				Mat mask = frame.clone();

				//bitwise_not(mask, mask);

				//Mat mask = Mat::zeros(frame.size(),CV_8UC1);

				for (auto i = 0; i < tracks.size(); i++) {

					circle(mask, tracks[i].back(), 10, Scalar(0,0,0), -1);

				}

        		//imshow("MASK", mask);

        		findContours(mask, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

			}
			else {*/

			findContours(frame, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
			//}

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

			//for (auto i = 0; i < tracks.size(); i++) {

			//	circle(imageROI, tracks[i].back(), 3, Scalar(0,0,0), -1);
			//}

			// Nea points sygkrish me ta palia wste na mhn yparxoyn dipla

			if (!p.empty()) {

				if (!tracks.empty()) {

					for (auto i = 0; i<p.size(); i++) {

						match = 0;

						for (auto j = 0; j<tracks.size(); j++) {

							dist1 = cv::norm(p[i]-tracks[j].back());
       
							if (dist1 < 13) {

								//circle(imageROI, p[i], 3, Scalar(255,0,0), 2);

								match = 1;

								break;
							}
						}

						if (match == 0) {

							//circle(imageROI, p[i], 3, Scalar(0,0,255), 2);

							p_unique.push_back(p[i]);

						}
					}
				}
				else {

					p_unique = p;
				}
			}

			//imshow("Contours", imageROI);

			//int keyboard = waitKey(0);


			neighbor_p.clear();
			p_new.clear();

			int a=0;
			float dist2;

			//for (auto i = 0; i < tracks.size(); i++) {

				//circle(imageROI, tracks[i].back(), 3, Scalar(0,0,0), -1);
			//}


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

							//printf("size %ld\n", neighbor_p.size());

							//for (auto m = 0; m < neighbor_p.size(); m++) {

								//circle(imageROI, neighbor_p[m], 3, Scalar(0,255,0), -1);
							//}

							Rect box = boundingRect(neighbor_p); 
							rectangle(imageROI, box, Scalar(0,0,255),1,8,0);


							//mo_x = (*i).x;
							//mo_y = (*i).y;

							mo_x = 0;
							mo_y = 0;
							
							for (auto k = 0; k<neighbor_p.size(); k++) {


								mo_x = mo_x + neighbor_p[k].x;
								mo_y = mo_y + neighbor_p[k].y;

							}

							//mo_point.x = mo_x / (neighbor_p.size()+1);
							//mo_point.y = mo_y / (neighbor_p.size()+1);

							mo_point.x = mo_x / neighbor_p.size();
							mo_point.y = mo_y / neighbor_p.size();


							p_new.push_back(mo_point);

							//circle(imageROI, mo_point, 3, Scalar(0,0,255), -1);


						}
						else {

							//circle(imageROI, *i, 3, Scalar(0,255,0), -1);

							p_new.push_back(*i);

							(*i).x = 0;
							(*i).y = 0;
						}

					}


				}
			}

			//imshow("Contours", imageROI);

			//int keyboard = waitKey(0);

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

		
		double e2 = getTickCount();
		float t = (e2 - e1)/getTickFrequency();

		//printf("Time frame %d : %f s\n", frame_idx, t);

		//Mat combine;

		//hconcat(vis,imageROI, combine);

        imshow("Optical flow tracks", vis);
		

		int key = waitKey(30);

        if (key == 'q' || key == 27)
            break;

        if (key == ' ')
        	waitKey(-1);

	}
	
	 
}


