#import <UIKit/UIKit.h>
#import "caffe/caffe.hpp"
#import <opencv2/opencv.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iomanip>

using namespace caffe;
using std::string;

const std::vector<std::string> classes = {"background", "face"};

struct DETECTED_FACE {
    /** */
    uint32_t id;
    /** */
    float score;
    /** */
    cv::Rect bbox;
    
    int class_id;
};

class FastFaceDetector {
public:
    FastFaceDetector();
    FastFaceDetector(const string& model_file,
                     const string& trained_file);
    std::vector<DETECTED_FACE> execute(const cv::Mat image);
    
    std::string getClassName(int classId);
    
private:
    void WrapInputLayer(std::vector<cv::Mat>* input_channels);
    
    void Preprocess(const cv::Mat& img,
                    std::vector<cv::Mat>* input_channels);
    
    void setMean(const std::string& meanFile,
                 const std::string& meanValue);
    
    //shared_ptr<Net<float> > net_;
    caffe::Net<float>* net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat odMean;
};
