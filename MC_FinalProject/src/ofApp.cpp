#include "ofApp.h"



//--------------------------------------------------------------
void ofApp::setup(){
    
    //Camera
    vidGrabber.setVerbose(true);
    vidGrabber.setup(320,240);
    
    camera.allocate(320,240);
    grayImage.allocate(320,240);
    grayBg.allocate(320,240);
    grayDiff.allocate(320,240);
    
    bLearnBakground = true;
    cameraON = false;
    threshold = 80;
    
    //Directory assigment
    dir.listDir("images/of_logos/");
    dir.allowExt("jpg");
    dir.sort();
    
    if (dir.size()) {
        //video.assign(dir.size(), ofVideoPlayer());
        images.assign(dir.size(), ofImage());
    }
    
    //load images to the directoy
    for (int i =0; i < dir.size();i++){
        //video[i].load(dir.getPath(i));
        images[i].load(dir.getPath(i));
    }
    
    currentImage = 0;
    ofBackground(ofColor::white);
    
}

//--------------------------------------------------------------
void ofApp::update(){
    
    //Camera 
    bool bNewFrame = false;
    vidGrabber.update();
    bNewFrame = vidGrabber.isFrameNew();
    
    if (bNewFrame){
        camera.setFromPixels(vidGrabber.getPixels());
        grayImage = camera;
        if (bLearnBakground == true){
            grayBg = grayImage;        // the = sign copys the pixels from grayImage into grayBg (operator overloading)
            bLearnBakground = false;
        }

        // take the abs value of the difference between background and incoming and then threshold:
        grayDiff.absDiff(grayBg, grayImage);
        grayDiff.threshold(threshold);

        // find contours which are between the size of 20 pixels and 1/3 the w*h pixels.
        // also, find holes is set to true so we will get interior contours as well....
        contourFinder.findContours(grayDiff, 20, (320*240)/3, 10, true);    // find holes
    }
}

//--------------------------------------------------------------
void ofApp::draw(){

    ofSetColor(ofColor::white);
    images[currentImage].draw(421,284,200,200);                 //draw center image
    
    if (cameraON == false ){
        //camara mode
        if (dir.size() > 0){
            for (int j=0; j < dir.size() ;j++){
                float dist  = metadata.readFile(findMeta(currentImage),findMeta(j));                 //read and compare every file
                if ((currentImage != j)){
                    images[j].draw((421+50) * dist ,(284+50) * dist ,100,100);                       //draw images arround
                }
            }
            ofSetColor(ofColor::gray);
            string Instructions =  string("N for next picture") + "\n\n" + "C for camara control ON ";
            ofDrawBitmapString(Instructions, 690,200);
        }
    }else{
        
    //Camera
    camera.draw(704,568);
        
    //we can draw each blob individually from the blobs vector,
    //this is how to get access to them:
     for (int i = 0; i < contourFinder.nBlobs; i++){
        contourFinder.blobs[i].draw(704,568);
         // draw over the centroid if the blob is a hole
        ofSetColor(255);
        if(contourFinder.blobs[i].hole){
            ofDrawBitmapString("hole",
                contourFinder.blobs[i].boundingRect.getCenter().x + 704,
                contourFinder.blobs[i].boundingRect.getCenter().y + 568);
            }

         //Detect motion
         if((contourFinder.blobs[i].boundingRect.getCenter().x + 704) < 864 && (contourFinder.blobs[i].boundingRect.getCenter().y + 568) < 598  ){
             //Right hand detecor
             int timerR;
             timerR++;
             if (timerR>30){
                 currentImage ++;
             }
         }else if ((contourFinder.blobs[i].boundingRect.getCenter().x + 704) > 864 && (contourFinder.blobs[i].boundingRect.getCenter().y + 568) < 598 ){
             //Left hand detecor
             int timerL;
             timerL++;
             if (timerL>30){
                 currentImage --;
             }
             
         }
     }
    ofSetColor(ofColor::gray);
    string Instructions =  string("Raise right hand for the next photo") + "\n\n" + "Raise left hand for the previous photo" +"\n\n"+ "SPACE for setting the bakground" + "\n\n" +  "X for camara control OFF" ;
    ofDrawBitmapString(Instructions, 690,450);
    }
    
    ofSetColor(ofColor::gray);
    string pathInfo = dir.getName(currentImage);
    ofDrawBitmapString(pathInfo, 421, 500);

}

ofxXmlSettings ofApp::findMeta(int index){

    array<int,3> color = metadata.color(images[index], 0);
    float lum =  metadata.luminance(color);
    int nobj = metadata.NObject(images[currentImage],images[index],0);
    int faces = metadata.faceDetection(images[index], 0);
    array<float,6> edge = {0,0,0,0,0,0};
    array<float,12> tex ={0,0,0,0,0,0,0,0,0,0,0,0};
    //array<float,6> edge = metadata.edgeDetection(images[index], 0);
    //array<float,12> tex =  metadata.texChar(images[index], 0);
    string text = "file ";
    string keyword = "ID ";

    ofxXmlSettings file = metadata.writeFile( keyword += to_string(index) , lum, color , faces, edge, tex  ,nobj);
    file.saveFile(text += to_string(index));
        return file;
    
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    
    switch (key){
        case ' ':
            bLearnBakground = true;
            break;
        case 'n':
            currentImage++;
            currentImage %= dir.size();
            break;
        case 'c':
            cameraON = true;
            break;
        case 'x':
            cameraON = false;
            break;
    }
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
