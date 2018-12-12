//
//  FastObjectDetector.cpp
//  CaffeSimple
//
//  Created by haiphan on 9/29/17.
//  Copyright © 2017 CMU. All rights reserved.
//

#include "FastFaceDetector.h"

using namespace caffe;
using std::string;

FastFaceDetector::FastFaceDetector() {
    
}

FastFaceDetector::FastFaceDetector(const string& model_file,
                                   const string& trained_file) {
    /* Load the network. */
    net_ = new caffe::Net<float>(model_file, caffe::TEST);
    
    //net_->reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);
    
    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    
    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    
    std::string mean_file = "";
    std::string mean_value = "127.5,127.5,127.5";
    setMean(mean_file, mean_value);
}

std::string FastFaceDetector::getClassName(int classId) {
    return classes[classId];
}

std::vector<DETECTED_FACE> FastFaceDetector::execute(const cv::Mat image) {
    caffe::Blob<float>* inputLayer = net_->input_blobs()[0];
    inputLayer->Reshape(1, num_channels_,
                        input_geometry_.height, input_geometry_.width);
    
    /* Forward dimension change to all layers. */
    net_->Reshape();
    
    std::vector<cv::Mat> inputChannels;
    WrapInputLayer(&inputChannels);
    
    Preprocess(image, &inputChannels);
    
    net_->Forward();
    
    /* Copy the output layer to a std::vector */
    caffe::Blob<float>* resultBlob = net_->output_blobs()[0];
    const float* result = resultBlob->cpu_data();
    const int numberOfDetections = resultBlob->height();
    std::vector<std::vector<float> > detections;
    for (int k = 0; k < numberOfDetections; ++k) {
        if (result[0] == -1) {
            // Skip invalid detection.
            result += 7;
            continue;
        }
        std::vector<float> detection(result, result + 7);
        detections.push_back(detection);
        result += 7;
    }
    
    std::vector<DETECTED_FACE> detectedObjs;
    
    for (int i = 0; i < detections.size(); ++i) {
        DETECTED_FACE detectedObj;
        
        std::vector<float> d = detections[i];
        
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        if (d.size() != 7) {
            std::cerr << "Malformed detection result with size "
            << d.size() << "." << std::endl;
            exit(1);
        }
        
        float score = d[2];
        int classId = d[1];
        detectedObj.class_id = classId;
        
        if (score >= 0.8) {
            detectedObj.id = i+1;
            detectedObj.score = score;
            detectedObj.bbox.x = static_cast<uint32_t>(d[3] * image.cols);
            detectedObj.bbox.y = static_cast<uint32_t>(d[4] * image.rows);
            detectedObj.bbox.width = static_cast<uint32_t>(d[5] * image.cols) - static_cast<uint32_t>(d[3] * image.cols);
            detectedObj.bbox.height = static_cast<uint32_t>(d[6] * image.rows) - static_cast<uint32_t>(d[4] * image.rows);
            
            detectedObjs.push_back(detectedObj);
        }
    }
    
    return detectedObjs;
}

/* Load the mean file in binaryproto format. */
void FastFaceDetector::setMean(const std::string& meanFile,
                               const std::string& meanValue) {
    cv::Scalar channelMean;
    if (!meanFile.empty()) {
        CHECK(meanValue.empty()) <<
        "Cannot specify meanFile and meanValue at the same time";
        caffe::BlobProto protoBlob;
        ReadProtoFromBinaryFileOrDie(meanFile.c_str(), &protoBlob);
        
        /* Convert from BlobProto to Blob<float> */
        caffe::Blob<float> meanBlob;
        meanBlob.FromProto(protoBlob);
        CHECK_EQ(meanBlob.channels(), num_channels_)
        << "Number of channels of mean file doesn't match input layer.";
        
        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        std::vector<cv::Mat> channels;
        float* data = meanBlob.mutable_cpu_data();
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(meanBlob.height(), meanBlob.width(), CV_32FC1, data);
            channels.push_back(channel);
            data += meanBlob.height() * meanBlob.width();
        }
        
        /* Merge the separate channels into a single image. */
        cv::Mat mean;
        cv::merge(channels, mean);
        
        /* Compute the global mean pixel value and create a mean image
         * filled with this value. */
        channelMean = cv::mean(mean);
        odMean = cv::Mat(input_geometry_, mean.type(), channelMean);
    }
    if (!meanValue.empty()) {
        CHECK(meanFile.empty()) <<
        "Cannot specify meanFile and meanValue at the same time";
        std::stringstream ss(meanValue);
        std::vector<float> values;
        std::string item;
        while (getline(ss, item, ',')) {
            float value = std::atof(item.c_str());
            values.push_back(value);
        }
        CHECK(values.size() == 1 || values.size() == num_channels_) <<
        "Specify either 1 meanValue or as many as channels: " << num_channels_;
        
        std::vector<cv::Mat> channels;
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
                            cv::Scalar(values[i]));
            channels.push_back(channel);
        }
        cv::merge(channels, odMean);
    }
}


void FastFaceDetector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    
    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}



void FastFaceDetector::Preprocess(const cv::Mat& img,
                                  std::vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, CV_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, CV_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, CV_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, CV_GRAY2BGR);
    else
        sample = img;
    
    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;
    
    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);
    
    cv::Mat sample_normalized;
    cv::subtract(sample_float, odMean, sample_normalized);
    cv::multiply(sample_normalized, 0.007843, sample_normalized);
    
    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);
    
    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}
