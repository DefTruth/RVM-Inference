//
// Created by DefTruth on 2021/11/29.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_FCN_RESNET101_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_FCN_RESNET101_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  class LITE_EXPORTS TNNFCNResNet101
  {
  public:
    explicit TNNFCNResNet101(const std::string &_proto_path,
                             const std::string &_model_path,
                             unsigned int _num_threads = 1);

    ~TNNFCNResNet101();

  private:
    const char *log_id = nullptr;
    const char *proto_path = nullptr;
    const char *model_path = nullptr;
    // Note, tnn:: actually is TNN_NS::, I prefer the first one.
    std::shared_ptr<tnn::TNN> net;
    std::shared_ptr<tnn::Instance> instance;
    std::shared_ptr<tnn::Mat> input_mat;

  private:
    std::vector<float> scale_vals = {(1.f / 0.229f) * (1.f / 255.f),
                                     (1.f / 0.224f) * (1.f / 255.f),
                                     (1.f / 0.225f) * (1.f / 255.f)};
    std::vector<float> bias_vals = {-0.485f * 255.f * (1.f / 0.229f) * (1.f / 255.f),
                                    -0.456f * 255.f * (1.f / 0.224f) * (1.f / 255.f),
                                    -0.406f * 255.f * (1.f / 0.225f) * (1.f / 255.f)}; // RGB

  private:
    const unsigned int num_threads; // initialize at runtime.
    int dynamic_input_height = 512; // init only, will change according to input mat.
    int dynamic_input_width = 512; // init only, will change according to input mat.
    tnn::DataFormat input_data_format;  // e.g DATA_FORMAT_NHWC
    tnn::MatType input_mat_type; // e.g NCHW_FLOAT
    tnn::DeviceType input_device_type; // only CPU, namely ARM or X86
    tnn::DeviceType output_device_type; // only CPU, namely ARM or X86
    tnn::DeviceType network_device_type; // e.g DEVICE_X86 DEVICE_NAIVE DEVICE_ARM
    tnn::DimsVector input_shape; // debug
    tnn::DimsVector output_shape;

    const char *class_names[20] = {
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor"
    }; // 20 classes

    // un-copyable
  protected:
    TNNFCNResNet101(const TNNFCNResNet101 &) = delete; //
    TNNFCNResNet101(TNNFCNResNet101 &&) = delete; //
    TNNFCNResNet101 &operator=(const TNNFCNResNet101 &) = delete; //
    TNNFCNResNet101 &operator=(TNNFCNResNet101 &&) = delete; //

  private:
    void print_debug_string(); // debug information

  private:
    void transform(const cv::Mat &mat); //

    void initialize_instance(); // init net & instance

  public:
    void detect(const cv::Mat &mat, types::SegmentContent &content);
  };
}

#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_FCN_RESNET101_H
