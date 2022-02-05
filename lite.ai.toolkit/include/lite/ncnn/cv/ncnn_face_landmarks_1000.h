//
// Created by DefTruth on 2021/11/21.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CV_NCNN_FACE_LANDMARKS_1000_H
#define LITE_AI_TOOLKIT_NCNN_CV_NCNN_FACE_LANDMARKS_1000_H

#include "lite/ncnn/core/ncnn_core.h"

namespace ncnncv
{
  class LITE_EXPORTS NCNNFaceLandmark1000 : public BasicNCNNHandler
  {
  public:
    explicit NCNNFaceLandmark1000(const std::string &_param_path,
                                  const std::string &_bin_path,
                                  unsigned int _num_threads = 1);

    ~NCNNFaceLandmark1000() override = default;

  private:
    const int input_height = 128;
    const int input_width = 128;
    const float mean_vals[1] = {0.0f};
    const float norm_vals[1] = {1.0f};

  private:
    void transform(const cv::Mat &mat, ncnn::Mat &in) override;

  public:
    void detect(const cv::Mat &mat, types::Landmarks &landmarks);
  };
}


#endif //LITE_AI_TOOLKIT_NCNN_CV_NCNN_FACE_LANDMARKS_1000_H
