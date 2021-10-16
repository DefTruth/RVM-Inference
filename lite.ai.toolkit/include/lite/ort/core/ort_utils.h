//
// Created by DefTruth on 2021/3/28.
//

#ifndef LITE_AI_ORT_CORE_ORT_UTILS_H
#define LITE_AI_ORT_CORE_ORT_UTILS_H

#include "ort_config.h"

namespace ortcv
{
  // specific utils for ONNXRuntime
  namespace utils
  {
    namespace transform
    {
      enum
      {
        CHW = 0, HWC = 1
      };

      /**
       * @param mat CV:Mat with type 'CV_32FC3|2|1'
       * @param tensor_dims e.g {1,C,H,W} | {1,H,W,C}
       * @param memory_info It needs to be a global variable in a class
       * @param tensor_value_handler It needs to be a global variable in a class
       * @param data_format CHW | HWC
       * @return
       */
      LITE_EXPORTS Ort::Value create_tensor(const cv::Mat &mat, const std::vector<int64_t> &tensor_dims,
                                               const Ort::MemoryInfo &memory_info_handler,
                                               std::vector<float> &tensor_value_handler,
                                               unsigned int data_format = CHW) throw(std::runtime_error);

      LITE_EXPORTS cv::Mat normalize(const cv::Mat &mat, float mean, float scale);

      LITE_EXPORTS cv::Mat normalize(const cv::Mat &mat, const float mean[3], const float scale[3]);

      LITE_EXPORTS void normalize(const cv::Mat &inmat, cv::Mat &outmat, float mean, float scale);

      LITE_EXPORTS void normalize_inplace(cv::Mat &mat_inplace, float mean, float scale);

      LITE_EXPORTS void normalize_inplace(cv::Mat &mat_inplace, const float mean[3], const float scale[3]);
    }

  } // NAMESPACE UTILS
} // NAMESPACE ORTCV

#endif //LITE_AI_ORT_CORE_ORT_UTILS_H
