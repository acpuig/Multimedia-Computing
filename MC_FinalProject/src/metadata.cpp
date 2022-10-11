//
//  metadata.cpp
//  videoPlayerExample
//
//  Created by Ariadna Cortés i Puig
//

#include <stdio.h>
#include "metadata.h"
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>


#include<opencv2/opencv.hpp>
using namespace cv;


float Metadata::readFile( ofxXmlSettings file1, ofxXmlSettings file2){
    
    //Tags (keywords)
    string tag     = file1.getValue("TAG", "errorTag");
    //Luminance
    int lum         = file1.getValue("LUMINANCE", 170);
    //Color
    int red         = file1.getValue("COLOR:RED", 170);
    int green       = file1.getValue("COLOR:GREEN", 190);
    int blue        = file1.getValue("COLOR:BLUE", 240);
    //Number of faces appearing in the image or video
    int NFaces      = file1.getValue("NFACES", 0);
    //Edge distribution
    float meanLap   = file1.getValue("EDGE_DIST:LAPLACIAN:MEAN", 0);
    float stdLap    = file1.getValue("EDGE_DIST:LAPLACIAN:STD", 1);
    float meanRob   = file1.getValue("EDGE_DIST:ROBERTS:MEAN", 0);
    float stdRob    = file1.getValue("EDGE_DIST:ROBERTS:STD", 1);
    float meanSob   = file1.getValue("EDGE_DIST:SOBEL:MEAN", 0);
    float stdSob    = file1.getValue("EDGE_DIST:SOBEL:STD", 1);
    //Texture characteristics
    float txG1Mean  = file1.getValue("TEX_CHAR:G1:MEAN", 0);
    float txG1Std   = file1.getValue("TEX_CHAR:G1:STD", 1);
    float txG2Mean  = file1.getValue("TEX_CHAR:G2:MEAN", 0);
    float txG2Std   = file1.getValue("TEX_CHAR:G2:STD ", 1);
    float txG3Mean  = file1.getValue("TEX_CHAR:G3:MEAN ", 0);
    float txG3Std   = file1.getValue("TEX_CHAR:G3:STD ", 1);
    float txG4Mean  = file1.getValue("TEX_CHAR:G4:MEAN ", 0);
    float txG4Std   = file1.getValue("TEX_CHAR:G4:STD ", 1);
    float txG5Mean  = file1.getValue("TEX_CHAR:G5:MEAN ", 0);
    float txG5Std   = file1.getValue("TEX_CHAR:G5:STD ", 1);
    float txG6Mean  = file1.getValue("TEX_CHAR:G6:MEAN ", 0);
    float txG6Std   = file1.getValue("TEX_CHAR:G6:STD ", 1);
    //Number of times a specific object (input as an image) appears in the video frame
    int Ntimes      = file1.getValue("NTIMES", 0);
    //Scene change (only for videos)
    int Nscene      = file1.setValue("NSCENE", 0);
    
    
    //Tags (keywords)
    string tag2     = file2.getValue("TAG", "errorTag");
    //Luminance
    int lum2         = file2.getValue("LUMINANCE", 170);
    //Color
    int red2         = file2.getValue("COLOR:RED", 170);
    int green2      = file2.getValue("COLOR:GREEN", 190);
    int blue2        = file2.getValue("COLOR:BLUE", 240);
    //Number of faces appearing in the image or video
    int NFaces2      = file2.getValue("NFACES", 0);
    //Edge distribution
    float meanLap2   = file2.getValue("EDGE_DIST:LAPLACIAN:MEAN", 0);
    float stdLap2    = file2.getValue("EDGE_DIST:LAPLACIAN:STD", 1);
    float meanRob2   = file2.getValue("EDGE_DIST:ROBERTS:MEAN", 0);
    float stdRob2    = file2.getValue("EDGE_DIST:ROBERTS:STD", 1);
    float meanSob2   = file2.getValue("EDGE_DIST:SOBEL:MEAN", 0);
    float stdSob2    = file2.getValue("EDGE_DIST:SOBEL:STD", 1);
    //Texture characteristics
    float txG1Mean2  = file2.getValue("TEX_CHAR:G1:MEAN", 0);
    float txG1Std2   = file2.getValue("TEX_CHAR:G1:STD", 1);
    float txG2Mean2  = file2.getValue("TEX_CHAR:G2:MEAN", 0);
    float txG2Std2   = file2.getValue("TEX_CHAR:G2:STD ", 1);
    float txG3Mean2  = file2.getValue("TEX_CHAR:G3:MEAN ", 0);
    float txG3Std2   = file2.getValue("TEX_CHAR:G3:STD ", 1);
    float txG4Mean2  = file2.getValue("TEX_CHAR:G4:MEAN ", 0);
    float txG4Std2   = file2.getValue("TEX_CHAR:G4:STD ", 1);
    float txG5Mean2  = file2.getValue("TEX_CHAR:G5:MEAN ", 0);
    float txG5Std2   = file2.getValue("TEX_CHAR:G5:STD ", 1);
    float txG6Mean2  = file2.getValue("TEX_CHAR:G6:MEAN ", 0);
    float txG6Std2   = file2.getValue("TEX_CHAR:G6:STD ", 1);
    //Number of times a specific object (input as an image) appears in the video frame
    int Ntimes2      = file2.getValue("NTIMES", 0);
    //Scene change (only for videos)
    int Nscene2      = file2.setValue("NSCENE", 0);
    
    file2.popTag();
    
    //Compare metadata
    float lumMetadata     = abs  (lum - lum2);
    float colorMetadata   = abs  ((red+blue+green)/3 - (red2+blue2+green2)/3 );
    float meanLapMetadata = abs  (meanLap - meanLap2);
    float stdLapMetadata  = abs  (stdLap - stdLap2);
    float meanRobMetadata = abs  (meanRob - meanRob2);
    float stdRobMetadata  = abs  (stdRob - stdRob2);
    float meanSobMetadata = abs  (meanSob - meanSob2);
    float stdSobMetadata  = abs  (stdSob - stdSob2);
    float txG1MeanMetadata = abs (txG1Mean - txG1Mean2);
    float txG1StdMetadata  = abs (txG1Std - txG1Std2);
    float txG2MeanMetadata = abs (txG2Mean - txG2Mean2);
    float txG2StdMetadata  = abs (txG2Std - txG2Std2);
    float txG3MeanMetadata = abs (txG3Mean - txG3Mean2);
    float txG3StdMetadata  = abs (txG3Std - txG3Std2);
    float txG4MeanMetadata = abs (txG4Mean - txG4Mean2);
    float txG4StdMetadata  = abs (txG4Std - txG4Std2);
    float txG5MeanMetadata = abs (txG5Mean - txG5Mean2);
    float txG5StdMetadata  = abs (txG5Std - txG5Std2);
    float txG6MeanMetadata = abs (txG6Mean - txG6Mean2);
    float txG6StdMetadata  = abs (txG6Std - txG6Std2);
    float NFacesMetadata = abs (Ntimes - Ntimes2);
    float NSceneMetadata = abs (Nscene -  Nscene2);
    
    // Difference between two images
    int diference = (lumMetadata + colorMetadata + meanLapMetadata + stdLapMetadata +  meanRobMetadata + stdRobMetadata + meanSobMetadata + stdSobMetadata + txG1MeanMetadata + txG1StdMetadata + txG2MeanMetadata + txG2StdMetadata + txG3MeanMetadata + txG3StdMetadata + txG4MeanMetadata + txG4StdMetadata + txG5MeanMetadata + txG5StdMetadata + txG6MeanMetadata + txG6StdMetadata + NFacesMetadata + NSceneMetadata);

    //Return number that feets on the window
    return diference*0.01;
}


ofxXmlSettings Metadata::writeFile(string TAG, float luminance, array<int,3> color, int NFaces, array<float,6> edge_dist, array<float,12> texture_char, int Ntimes)
{
    
    ofxXmlSettings XML;
    XML.clear();        //set blank file
    //Tags (keywords)
	XML.addTag("TAG");
	XML.setValue("TAG", TAG);

    //Luminance
	XML.addTag("LUMINANCE");
	XML.setValue("LUMINANCE", luminance);

    //Color
	XML.addTag("COLOR");
	XML.setValue("COLOR:RED", color[0]);
	XML.setValue("COLOR:GREEN", color[1]);
	XML.setValue("COLOR:BLUE", color[2]);

    //Number of faces appearing in the image or video
	XML.addTag("NFACES");
	XML.setValue("NFACES", NFaces);

    //Edge distribution
	XML.addTag("EDGE_DIST");
    XML.addTag("EDGE_DIST:LAPLACIAN");
    XML.setValue("EDGE_DIST:LAPLACIAN:MEAN ", edge_dist[0]);
    XML.setValue("EDGE_DIST:LAPLACIAN:STD ", edge_dist[1]);

    XML.addTag("EDGE_DIST:ROBERTS");
    XML.setValue("EDGE_DIST:ROBERTS:MEAN ", edge_dist[2]);
    XML.setValue("EDGE_DIST:ROBERTS:STD ", edge_dist[3]);
    
    XML.addTag("EDGE_DIST:SOBEL");
    XML.setValue("EDGE_DIST:SOBEL:MEAN ", edge_dist[4]);
    XML.setValue("EDGE_DIST:SOBEL:STD ", edge_dist[5]);
    
    //Texture characteristics
	XML.addTag("TEX_CHAR");
    XML.addTag("TEX_CHAR:G1");
    XML.setValue("TEX_CHAR:G1:MEAN ", texture_char[0]);
    XML.setValue("TEX_CHAR:G1:STD ", texture_char[1]);
    
    XML.addTag("TEX_CHAR:G2");
    XML.setValue("TEX_CHAR:G2:MEAN ", texture_char[2]);
    XML.setValue("TEX_CHAR:G2:STD ", texture_char[3]);
    
    XML.addTag("TEX_CHAR:G3");
    XML.setValue("TEX_CHAR:G3:MEAN ", texture_char[4]);
    XML.setValue("TEX_CHAR:G3:STD ", texture_char[5]);
    
    XML.addTag("TEX_CHAR:G4");
    XML.setValue("TEX_CHAR:G4:MEAN ", texture_char[6]);
    XML.setValue("TEX_CHAR:G4:STD ", texture_char[7]);
    
    XML.addTag("TEX_CHAR:G5");
    XML.setValue("TEX_CHAR:G5:MEAN ", texture_char[8]);
    XML.setValue("TEX_CHAR:G5:STD ", texture_char[9]);
    
    XML.addTag("TEX_CHAR:G6");
    XML.setValue("TEX_CHAR:G6:MEAN ", texture_char[10]);
    XML.setValue("TEX_CHAR:G6:STD ", texture_char[11]);

    //Number of times a specific object (input as an image) appears in the video frame
	XML.addTag("NTIMES");
    XML.setValue("NTIMES", Ntimes);
    
    XML.addTag("END");

    return XML;


}

int Metadata::NObject(ofImage image1,ofImage image2, bool bin){
    //code from https://docs.opencv.org/2.4/doc/user_guide/ug_features2d.html
    
    ofxCvColorImage im;
    im.setFromPixels(image1.getPixels());
    Mat img1 = toCv(im.getPixels()) ; //model
    
    ofxCvColorImage im2;
    im2.setFromPixels(image2.getPixels());
    Mat img2 = toCv(im2.getPixels()) ;
    
    // detecting keypoints
    vector<KeyPoint> keypoints1, keypoints2;
    
    //defining de ORB detector 
    cv::Ptr<FeatureDetector> detector = ORB::create();
    cv::Ptr<DescriptorExtractor> extractor = ORB::create();

    
    detector->detect(img1, keypoints1);
    detector->detect(img2, keypoints2); //matching two diferrent images

    // computing descriptors
    Mat descriptors1, descriptors2;
    extractor->compute(img1,keypoints1, descriptors1);
    extractor->compute(img2,keypoints2, descriptors2);

    
    // matching descriptors
    BFMatcher matcher ;
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    return (int)matches.size();
    
}


array<int,3> Metadata::color(ofImage image, bool bin)
{
    int r = 0, g = 0, b = 0;
    int total_r = 0, total_g = 0, total_b = 0;
    int media_r = 0, media_g = 0, media_b = 0;

    if (bin==0){
        for (int i = 0; i < (int)image.getPixels().size()-3; i+=3) {
            
            //get RGB colors
            r = image.getColor(i)[0];
            g = image.getColor(i)[1];
            b = image.getColor(i)[2];

            total_r = total_r + r;
            total_g = total_g + g;
            total_b = total_b + b;
        }
        //finde the media
        media_r = total_r / 9000;
        media_g = total_g / 9000;
        media_b = total_b / 9000;
        array<int, 3> media = { media_r, media_g, media_b };
    
    return media;
    } /*else {                                  //for video
        video.getCurrentFrame();
        ofPixels & pixels = video.getPixels();
        
        for (int i = 0; i < (int)video.getPixels().size()-3; i+=3) {

            r = pixels.getColor(i)[0];
            g = pixels.getColor(i)[1];
            b = pixels.getColor(i)[2];

            total_r = total_r + r;
            total_g = total_g + g;
            total_b = total_b + b;
        }
        
        int vidWidth = pixels.getWidth();
        int vidHeight = pixels.getHeight();
        int totalPixel = vidWidth * vidHeight;
        
        media_r = total_r / 9000;
        media_g = total_g / 9000 ;
        media_b = total_b / 9000 ;
        array<int, 3> media = { media_r, media_g, media_b };
        return media;
    }*/

}

int Metadata::faceDetection(ofImage image, bool bin)
{
    
    if (bin==0){
        finder.setup("haarcascade_frontalface_default.xml");
        if (finder.findHaarObjects(image)==0){
            return 0;
        }else return 1;
    } /*else{                                   //for video
        finder.setup("haarcascade_frontalface_default.xml");
        for (int i=0 ; i< video.getTotalNumFrames() ; i++){
            video.setFrame(i);
            ofPixels & pixels = video.getPixels();
            if (finder.findHaarObjects(pixels)==0){
                return 0;
            }else return 1;
        }
    }*/
}



float Metadata::luminance(array<int, 3> media)
{
        float Y = 0.2125 * media[0] + 0.7154 * media[1] + 0.07121 * media[2];
        return Y;
}

array<float,6> Metadata::edgeDetection(ofImage image, bool bin)
{
    Mat src_image;
    Mat graymat;
    if (bin==0){                   // for images
        ofxCvColorImage img;
        img.setFromPixels(image.getPixels());
        src_image = toCv(img.getPixels());
        cv::cvtColor(src_image, graymat, CV_BGR2GRAY);  //transform image to B&W
        
    } /* else{                     // for videos
        video.getCurrentFrame();
        ofxCvColorImage img;
        img.setFromPixels(video.getPixels());
        src_image = toCv(img.getPixels());
    }*/
        //kernel Laplacian
        Mat kernelLapl1 = (Mat_<double>(3,3) << -1,-1,-1,-1,8,-1,-1,-1,-1);
        Mat resultLapl;
        filter2D(graymat,resultLapl, -1, kernelLapl1);
    
        //kernel Roberts
        Mat kernelRoberts1 = (Mat_<double>(2,2) << 0,-1,1,0);
        Mat resultRoberts1;
        filter2D(graymat,resultRoberts1, -1, kernelRoberts1);
        Mat kernelRoberts2 = (Mat_<double>(2,2) << -1,0,0,1);
        Mat resultRoberts2;
        filter2D(graymat,resultRoberts2, -1, kernelRoberts2);
        Mat resultRoberts = resultRoberts1 + resultRoberts2;


        //kernel Sobel
        Mat kernelSobel1 = (Mat_<double>(3,3) << 1,0,-1,2,0,-2,1,0,-1);
        Mat resultSobel1;
        filter2D(graymat,resultSobel1, -1, kernelSobel1);
        Mat kernelSobel2 = (Mat_<double>(3,3) << 1,2,1,0,0,0,-1,-2,-1);
        Mat resultSobel2;
        filter2D(graymat,resultSobel2, -1, kernelSobel2);
        Mat resultSobel = resultSobel1 + resultSobel2;

     
        //Resulting mean and std
        array<float,2> Laplacian  = histogramCalc(resultLapl);
        array<float,2> Roberts  =  histogramCalc(resultRoberts);
        array<float,2> Sobel  = histogramCalc(resultSobel);
        array<float,6> result = {Laplacian[0],Laplacian[1],Roberts[0],Roberts[1],Sobel[0],Sobel[1]};
        
        return result;
    
}


array<float,12> Metadata::texChar(ofImage image, bool bin){
   
    Mat src_image;
    if (bin==0){                // for images
        ofxCvColorImage img;
        img.setFromPixels(image.getPixels());
        src_image = toCv(img.getPixels());      //translate image to MAT
        
    } /*else{                   // for videos
        video.getCurrentFrame();                //To analyze
        ofxCvColorImage img;
        img.setFromPixels(video.getPixels());
        src_image = toCv(img.getPixels());
    }*/
        
        int ksize = 31;
        int sigma = 1;
        int gamma = 1;
        float pi = 3.1415926;
        cv::Size KernalSize(ksize,ksize);
        
        //kernel Gabor theta = 0 | lambd = 7
        Mat kernelG1 = cv::getGaborKernel( KernalSize, sigma, 0, 7 , gamma, 0);
        Mat resultG1;
        filter2D(src_image,resultG1, -1, kernelG1);
        //kernel Gabor theta = 25 | lambd = 8
        Mat kernelG2 = cv::getGaborKernel( KernalSize, sigma, 25* pi/180, 8 , gamma, 0);
        Mat resultG2;
        filter2D(src_image,resultG2, -1, kernelG2);
        //kernel Gabor theta = 50 | lambd = 9
        Mat kernelG3 = cv::getGaborKernel( KernalSize, sigma, 50* pi/180, 9 , gamma, 0);
        Mat resultG3;
        filter2D(src_image,resultG3, -1, kernelG3);
        //kernel Gabor theta = 75 | lambd = 7
        Mat kernelG4 = cv::getGaborKernel( KernalSize, sigma, 75* pi/180, 7 , gamma, 0);
        Mat resultG4;
        filter2D(src_image,resultG4, -1, kernelG4);
        //kernel Gabor theta = 100 | lambd = 8
        Mat kernelG5 = cv::getGaborKernel( KernalSize, sigma, 100* pi/180, 8 , gamma, 0);
        Mat resultG5;
        filter2D(src_image,resultG5, -1, kernelG5);
        //kernel Gabor theta = 125 | lambd = 9
        Mat kernelG6 = cv::getGaborKernel( KernalSize, sigma, 125* pi/180, 9 , gamma, 0);
        Mat resultG6;
        filter2D(src_image,resultG6, -1, kernelG6);
    
        Mat result = resultG1 + resultG2 + resultG3 + resultG4 + resultG5 + resultG6;
             
        array<float,2> results  =  histogramCalc(result);         //Resulting mean and std

        return result;
    
    
}


array<float,2>  Metadata::histogramCalc(Mat image)
{
    //code from https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/page_tutorial_histogram_calculation.html
   
      vector<Mat> bgr_planes;
      split(image, bgr_planes );

      int histSize = 256;

      float range[] = { 0, 256 } ;
      const float* histRange = { range };

      bool uniform = true; bool accumulate = false;

      Mat b_hist, g_hist, r_hist;

      calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
      calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
      calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

      // Draw the histograms for B, G and R
      int hist_w = 512; int hist_h = 400;
      int bin_w = cvRound( (double) hist_w/histSize );

      Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

      normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
      normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
      normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

      for( int i = 1; i < histSize; i++ )
      {
          cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                                cv::Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                           Scalar( 255, 0, 0), 2, 8, 0  );
          cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                            cv::Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                           Scalar( 0, 255, 0), 2, 8, 0  );
          cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                            cv::Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                           Scalar( 0, 0, 255), 2, 8, 0  );
        
      }
    cv::Scalar mean, stddev;
    cv::meanStdDev(histImage, mean, stddev);
    
    //from Scalar to float
    float mean2= sum(mean)[0];
    float stddev2 = sum(stddev)[0];

    array<float, 2> result = {mean2, stddev2};
    return result;
    
}
