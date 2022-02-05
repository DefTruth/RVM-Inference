//
// Created by DefTruth on 2021/11/29.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CV_NCNN_DEEPLABV3_RESNET101_H
#define LITE_AI_TOOLKIT_NCNN_CV_NCNN_DEEPLABV3_RESNET101_H

#include "lite/ncnn/core/ncnn_core.h"

namespace ncnncv
{
  class LITE_EXPORTS NCNNDeepLabV3ResNet101 : public BasicNCNNHandler
  {
  public:
    explicit NCNNDeepLabV3ResNet101(const std::string &_param_path,
                                    const std::string &_bin_path,
                                    unsigned int _num_threads = 1); //
    ~NCNNDeepLabV3ResNet101() override = default;

  private:
    const float norm_vals[3] = {(1.f / 0.229f) * (1.f / 255.f),
                                (1.f / 0.224f) * (1.f / 255.f),
                                (1.f / 0.225f) * (1.f / 255.f)};
    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f}; // RGB

  private:
    int input_height = 512; // init only, will change according to input mat.
    int input_width = 512; // init only, will change according to input mat.

    const char *class_names[20] = {
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor"
    }; // 20 classes

  private:

    void transform(const cv::Mat &mat, ncnn::Mat &in) override;

  public:
    void detect(const cv::Mat &mat, types::SegmentContent &content);

  };
}

#endif //LITE_AI_TOOLKIT_NCNN_CV_NCNN_DEEPLABV3_RESNET101_H
