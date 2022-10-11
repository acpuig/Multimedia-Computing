 //
//  metadata.h
//  videoPlayerExample
//
//  Created by Ariadna Cortés i Puig
//
#pragma once


#ifndef metadata_h
#define metadata_h

#include "ofMain.h"
#include "ofxXmlSettings.h"
#include "ofxCvHaarFinder.h"
#include "ofxCvHaarFinder.h"
#include "ofxCv.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <array>



using namespace std;
using namespace cv;
using namespace ofxCv;

//#include <opencv2/opencv.hpp>

class Metadata {
public:
   
    float           readFile(ofxXmlSettings file1, ofxXmlSettings file2);
    ofxXmlSettings  writeFile(string TAG, float luminance, array<int,3> color, int Nfaces, array<float,6> edge_dist, array<float,12> texture_char, int Ntimes);
    
    array<int,3>    color(ofImage image, bool bin);
    float           luminance(array<int, 3> media);
    int             faceDetection(ofImage image, bool bin); // image = 0 ; video = 1
    array<float,6>  edgeDetection(ofImage image, bool bin);
    array<float,12> texChar(ofImage image, bool bin);
    array<float,2>  histogramCalc(Mat image);
    int             NObject(ofImage image1, ofImage image2, bool bin);

    ofxCvHaarFinder finder;
    
}
;
#endif
