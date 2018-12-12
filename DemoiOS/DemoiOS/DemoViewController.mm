//
//  DemoViewController.m
//  DemoiOS
//
//  Created by haiphan on 5/7/18.
//  Copyright © 2018 haiphan. All rights reserved.
//

#import "DemoViewController.h"
#include <numeric>
#include "caffe/caffe.hpp"
#include "ImageReader.h"
#import <opencv2/imgcodecs/ios.h>
#import "FastFaceDetector.h"
#import <opencv2/videoio/cap_ios.h>
#include <chrono>

@interface DemoViewController ()<CvVideoCameraDelegate>
@property (nonatomic, strong) UIImageView * imageView;
@property (nonatomic, strong) CvVideoCamera * videoCamera;
@end

@implementation DemoViewController

FastFaceDetector _faceDetector;

bool isBack;
bool isCamera;

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view from its nib.
    //self.myTextField.text = @"Hello World";
    
    [self setupImageView];
    
    isCamera = NO;
    
    // If use Camera
    if (isCamera) [self setupCameraWithType:AVCaptureDevicePositionFront];
    
    isBack = NO;
    
    NSString* model_file = FilePathForResourceName(@"MobileNetFaceSSD_deploy", @"protobin");
    NSString* trained_file = FilePathForResourceName(@"MobileNetSSD_deploy_face_i120k", @"caffemodel");

    string model_file_str = std::string([model_file UTF8String]);
    string trained_file_str = std::string([trained_file UTF8String]);
    _faceDetector = FastFaceDetector(model_file_str, trained_file_str);
    if (!isCamera) [self detectFaceImage];
    
}

- (void)setupImageView {
    self.imageView = [[UIImageView alloc] initWithFrame:CGRectMake(0, 0, self.view.frame.size.width, self.view.frame.size.height - 30)];
    [self.view addSubview:self.imageView];
}

- (void)detectFaceImage {
    
    UIImage * image = [UIImage imageNamed:@"face.jpg"];
    cv::Mat img;
    UIImageToMat(image, img);
    cv::Mat result = [self faceDetect:img];
    cv::cvtColor(result, result, CV_RGB2BGRA);
    self.imageView.image = MatToUIImage(result);
    
}

- (void)setupCameraWithType:(AVCaptureDevicePosition)type {
    if (self.videoCamera) {
        self.videoCamera = nil;
    }

    self.videoCamera = [[CvVideoCamera alloc] init];

    self.videoCamera.delegate = self;

    self.videoCamera.defaultAVCaptureDevicePosition = type;
    self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset640x480;
    self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    self.videoCamera.defaultFPS = 30;
    [self.videoCamera lockFocus];
    [self.videoCamera start];
}

unsigned long long getTickCount() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
                                                                 std::chrono::system_clock::now().time_since_epoch())
    .count();
}

- (cv::Mat)faceDetect:(const cv::Mat)image {
    cv::Mat image_copy;
    cv::cvtColor(image, image_copy, CV_BGRA2RGB);
    int64 start = getTickCount();
    
    std::vector<DETECTED_FACE> faces;
    faces = _faceDetector.execute(image_copy);
    int64 face_detect_duration = getTickCount() - start;
//    int face_fps = static_cast<int>(1000 / face_detect_duration);
    
    int fontFace = cv::FONT_HERSHEY_TRIPLEX;
    start = getTickCount();
    for (DETECTED_FACE face : faces) {
        cv::rectangle(image_copy, face.bbox, cv::Scalar(0, 255, 0), 2);
        int x = face.bbox.x;
        int y = face.bbox.y;
        
        std::string className = _faceDetector.getClassName(face.class_id);
        className = className + ": " + std::to_string(face.score);
        cv::rectangle(image_copy, cv::Point(x, y-10), cv::Point(x + face.bbox.width, y), cv::Scalar(0, 255, 0), CV_FILLED);
        cv::putText(image_copy, std::to_string(face.score), cv::Point(x + 4, y), fontFace, 0.5, cv::Scalar(255, 255, 255), 1.5);
    }
    
    return image_copy;
}

- (void)processImage:(cv::Mat &)image {
    dispatch_async(dispatch_get_main_queue(), ^{
        
        cv::Mat result;
        
        // Get results
        result = [self faceDetect:image];
        
        // Update UIs
        self.imageView.image = MatToUIImage(result);
    });
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

/*
#pragma mark - Navigation

// In a storyboard-based application, you will often want to do a little preparation before navigation
- (void)prepareForSegue:(UIStoryboardSegue *)segue sender:(id)sender {
    // Get the new view controller using [segue destinationViewController].
    // Pass the selected object to the new view controller.
}
*/

@end
