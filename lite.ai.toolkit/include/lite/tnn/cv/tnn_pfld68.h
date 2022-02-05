//
// Created by DefTruth on 2021/11/21.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_PFLD68_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_PFLD68_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  class LITE_EXPORTS TNNPFLD68 : public BasicTNNHandler
  {
  public:
    explicit TNNPFLD68(const std::string &_proto_path,
                       const std::string &_model_path,
                       unsigned int _num_threads = 1); //
    ~TNNPFLD68() override = default;

  private:
    // In TNN: x*scale + bias
    std::vector<float> scale_vals = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    std::vector<float> bias_vals = {0.f, 0.f, 0.f};

  private:
    void transform(const cv::Mat &mat) override; //

  public:
    void detect(const cv::Mat &mat, types::Landmarks &landmarks);
  };
}

#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_PFLD68_H
