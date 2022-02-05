//
// Created by DefTruth on 2021/11/29.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_FCN_RESNET101_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_FCN_RESNET101_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNFCNResNet101
  {
  public:
    explicit MNNFCNResNet101(const std::string &_mnn_path,
                             unsigned int _num_threads = 8); //
    ~MNNFCNResNet101();

  private:
    std::shared_ptr<MNN::Interpreter> mnn_interpreter;
    MNN::Session *mnn_session = nullptr;
    MNN::ScheduleConfig schedule_config;
    std::shared_ptr<MNN::CV::ImageProcess> pretreat; // init at runtime
    const char *log_id = nullptr;
    const char *mnn_path = nullptr;
    MNN::Tensor *input_tensor = nullptr;

  private:
    const float norm_vals[3] = {(1.f / 0.229f) * (1.f / 255.f),
                                (1.f / 0.224f) * (1.f / 255.f),
                                (1.f / 0.225f) * (1.f / 255.f)};
    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f}; // RGB

  private:
    const unsigned int num_threads; // initialize at runtime.
    int dynamic_input_height = 512; // init only, will change according to input mat.
    int dynamic_input_width = 512; // init only, will change according to input mat.
    int dimension_type; // hint only

    const char *class_names[20] = {
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor"
    }; // 20 classes

    // un-copyable
  protected:
    MNNFCNResNet101(const MNNFCNResNet101 &) = delete; //
    MNNFCNResNet101(MNNFCNResNet101 &&) = delete; //
    MNNFCNResNet101 &operator=(const MNNFCNResNet101 &) = delete; //
    MNNFCNResNet101 &operator=(MNNFCNResNet101 &&) = delete; //

  private:
    void print_debug_string();

  private:
    void transform(const cv::Mat &mat);

    void initialize_pretreat(); //

    void initialize_interpreter();

  public:
    void detect(const cv::Mat &mat, types::SegmentContent &content);

  };
}

#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_FCN_RESNET101_H
