//
// Created by DefTruth on 2021/11/21.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_PFLD68_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_PFLD68_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNPFLD68 : public BasicMNNHandler
  {
  public:
    explicit MNNPFLD68(const std::string &_mnn_path, unsigned int _num_threads = 1); //
    ~MNNPFLD68() override = default;

  private:
    const float mean_vals[3] = {0.0f, 0.0f, 0.0f};
    const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};

  private:
    void initialize_pretreat(); //

    void transform(const cv::Mat &mat) override; //

  public:
    void detect(const cv::Mat &mat, types::Landmarks &landmarks);
  };
}


#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_PFLD68_H
