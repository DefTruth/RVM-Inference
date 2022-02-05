//
// Created by DefTruth on 2021/11/14.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CV_NCNN_MOBILESE_FOCAL_FACE_H
#define LITE_AI_TOOLKIT_NCNN_CV_NCNN_MOBILESE_FOCAL_FACE_H

#include "lite/ncnn/core/ncnn_core.h"

namespace ncnncv
{
  class LITE_EXPORTS NCNNMobileSEFocalFace : public BasicNCNNHandler
  {
  public:
    explicit NCNNMobileSEFocalFace(const std::string &_param_path,
                                   const std::string &_bin_path,
                                   unsigned int _num_threads = 1) :
        BasicNCNNHandler(_param_path, _bin_path, _num_threads)
    {};

    ~NCNNMobileSEFocalFace() override = default;

  private:
    const float mean_vals[3] = {0.f, 0.f, 0.f}; // RGB
    const float norm_vals[3] = {1.f / 255.0f, 1.f / 255.0f, 1.f / 255.0f};
    static constexpr const int input_width = 128;
    static constexpr const int input_height = 128;

  private:
    void transform(const cv::Mat &mat, ncnn::Mat &in) override;

  public:
    void detect(const cv::Mat &mat, types::FaceContent &face_content);
  };
}


#endif //LITE_AI_TOOLKIT_NCNN_CV_NCNN_MOBILESE_FOCAL_FACE_H
