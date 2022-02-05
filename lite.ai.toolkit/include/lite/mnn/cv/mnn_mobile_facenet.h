//
// Created by DefTruth on 2021/11/14.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_MOBILE_FACENET_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_MOBILE_FACENET_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNMobileFaceNet : public BasicMNNHandler
  {
  public:
    explicit MNNMobileFaceNet(const std::string &_mnn_path, unsigned int _num_threads = 1); //
    ~MNNMobileFaceNet() override = default;

  private:
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f}; // RGB
    const float norm_vals[3] = {1.f / 128.0f, 1.f / 128.0f, 1.f / 128.0f};

  private:
    void initialize_pretreat(); //

    void transform(const cv::Mat &mat) override; //

  public:
    void detect(const cv::Mat &mat, types::FaceContent &face_content);
  };
}


#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_MOBILE_FACENET_H
