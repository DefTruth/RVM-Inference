//
// Created by DefTruth on 2021/11/21.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_FACE_LANDMARKS_1000_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_FACE_LANDMARKS_1000_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  class LITE_EXPORTS TNNFaceLandmark1000 : public BasicTNNHandler
  {
  public:
    explicit TNNFaceLandmark1000(const std::string &_proto_path,
                                 const std::string &_model_path,
                                 unsigned int _num_threads = 1); //
    ~TNNFaceLandmark1000() override = default;

  private:
    // In TNN: x*scale + bias
    std::vector<float> scale_vals = {1.0f};
    std::vector<float> bias_vals = {0.f};

  private:
    void transform(const cv::Mat &mat) override; //

  public:
    void detect(const cv::Mat &mat, types::Landmarks &landmarks);
  };
}


#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_FACE_LANDMARKS_1000_H
