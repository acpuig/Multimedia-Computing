#pragma once

#include "ofMain.h"
#include "ofxXmlSettings.h"
#include "metadata.h"

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
        string extension(string file_name);
        ofxXmlSettings findMeta(int index);
    
        Metadata metadata;

        ofDirectory dir;
        ofTrueTypeFont TTF;
    
        vector<ofImage> images;
        vector<ofVideoPlayer> video;
    
        //camera stuff
        ofVideoGrabber vidGrabber;
        ofxCvColorImage         camera;
        
        ofxCvGrayscaleImage     grayImage;
        ofxCvGrayscaleImage     grayBg;
        ofxCvGrayscaleImage     grayDiff;

        ofxCvContourFinder     contourFinder;
        int                    threshold;
        bool                   bLearnBakground;
        bool                   cameraON;


        int currentImage;
        string xmlStructure;
        string message;

		
};
