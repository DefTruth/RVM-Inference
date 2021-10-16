//
// Created by DefTruth on 2021/7/28.
//

#ifndef LITE_AI_ORT_CV_FACE_LANDMARKS_1000_H
#define LITE_AI_ORT_CV_FACE_LANDMARKS_1000_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS FaceLandmark1000 : public BasicOrtHandler
  {

  public:
    explicit FaceLandmark1000(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~FaceLandmark1000() override = default; // override

  private:
    Ort::Value transform(const cv::Mat &mat) override;

  public:
    void detect(const cv::Mat &mat, types::Landmarks &landmarks);

  };
}

#endif //LITE_AI_ORT_CV_FACE_LANDMARKS_1000_H
