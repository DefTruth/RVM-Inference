//
// Created by DefTruth on 2021/3/28.
//

#ifndef LITE_AI_ORT_CORE_ORT_UTILS_H
#define LITE_AI_ORT_CORE_ORT_UTILS_H

#include "ort_config.h"
#include "ort_types.h"

namespace ortcv
{
  namespace utils
  {
    static constexpr const float _PI = 3.1415926f;

    LITE_EXPORTS std::wstring to_wstring(const std::string &str);

    LITE_EXPORTS std::string to_string(const std::wstring &wstr);

    LITE_EXPORTS cv::Mat draw_axis(const cv::Mat &mat, const types::EulerAngles &euler_angles, float size = 50.f, int thickness = 2);

    LITE_EXPORTS cv::Mat draw_boxes(const cv::Mat &mat, const std::vector<types::Boxf> &boxes);

    LITE_EXPORTS cv::Mat draw_landmarks(const cv::Mat &mat, types::Landmarks &landmarks);

    LITE_EXPORTS cv::Mat draw_age(const cv::Mat &mat, types::Age &age);

    LITE_EXPORTS cv::Mat draw_gender(const cv::Mat &mat, types::Gender &gender);

    LITE_EXPORTS cv::Mat draw_emotion(const cv::Mat &mat, types::Emotions &emotions);

    LITE_EXPORTS void draw_boxes_inplace(cv::Mat &mat_inplace, const std::vector<types::Boxf> &boxes);

    LITE_EXPORTS void
    draw_axis_inplace(cv::Mat &mat_inplace, const types::EulerAngles &euler_angles, float size = 50.f, int thickness = 2);

    LITE_EXPORTS void draw_landmarks_inplace(cv::Mat &mat, types::Landmarks &landmarks);

    LITE_EXPORTS void draw_age_inplace(cv::Mat &mat_inplace, types::Age &age);

    LITE_EXPORTS void draw_gender_inplace(cv::Mat &mat_inplace, types::Gender &gender);

    LITE_EXPORTS void draw_emotion_inplace(cv::Mat &mat_inplace, types::Emotions &emotions);

    LITE_EXPORTS void
    hard_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output, float iou_threshold, unsigned int topk);

    LITE_EXPORTS void
    blending_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output, float iou_threshold, unsigned int topk);

    LITE_EXPORTS void
    offset_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output, float iou_threshold, unsigned int topk);

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

    namespace math
    {
      template<typename T=float>
      std::vector<T> softmax(const std::vector<T> &logits, unsigned int &max_id);

      template LITE_EXPORTS std::vector<float> softmax(const std::vector<float> &logits, unsigned int &max_id);

      template<typename T=float>
      std::vector<T> softmax(const T *logits, unsigned int _size, unsigned int &max_id);

      template LITE_EXPORTS std::vector<float> softmax(const float *logits, unsigned int _size, unsigned int &max_id);

      template<typename T=float>
      std::vector<unsigned int> argsort(const std::vector<T> &arr);

      template LITE_EXPORTS std::vector<unsigned int> argsort(const std::vector<float> &arr);

      template<typename T=float>
      std::vector<unsigned int> argsort(const T *arr, unsigned int _size);

      template LITE_EXPORTS std::vector<unsigned int> argsort(const float *arr, unsigned int _size);

      template<typename T=float>
      T cosine_similarity(const std::vector<T> &a, const std::vector<T> &b);

      template LITE_EXPORTS float cosine_similarity(const std::vector<float> &a, const std::vector<float> &b);
    }

  } // NAMESPACE UTILS
} // NAMESPACE ORTCV

namespace ortnlp
{
  namespace utils
  {

  } // NAMESPACE UTILS
} // NAMESPACE ORTNLP

namespace ortasr
{
  namespace utils
  {

  } // NAMESPACE UTILS
} // NAMESPACE ORTASR

template<typename T>
std::vector<T> ortcv::utils::math::softmax(const T *logits, unsigned int _size, unsigned int &max_id)
{
  ::__assert_type<T>();
  if (_size == 0 or logits == nullptr) return {};
  T max_prob = static_cast<T>(0), total_exp = static_cast<T>(0);
  std::vector<float> softmax_probs(_size);
  for (unsigned int i = 0; i < _size; ++i)
  {
    softmax_probs[i] = std::expf(logits[i]);
    total_exp += softmax_probs[i];
  }
  for (unsigned int i = 0; i < _size; ++i)
  {
    softmax_probs[i] = softmax_probs[i] / total_exp;
    if (softmax_probs[i] > max_prob)
    {
      max_id = i;
      max_prob = softmax_probs[i];
    }
  }
  return softmax_probs;
}

template<typename T>
std::vector<T> ortcv::utils::math::softmax(const std::vector<T> &logits, unsigned int &max_id)
{
  ::__assert_type<T>();
  if (logits.empty()) return {};
  const unsigned int _size = logits.size();
  T max_prob = static_cast<T>(0), total_exp = static_cast<T>(0);
  std::vector<float> softmax_probs(_size);
  for (unsigned int i = 0; i < _size; ++i)
  {
    softmax_probs[i] = std::expf(logits[i]);
    total_exp += softmax_probs[i];
  }
  for (unsigned int i = 0; i < _size; ++i)
  {
    softmax_probs[i] = softmax_probs[i] / total_exp;
    if (softmax_probs[i] > max_prob)
    {
      max_id = i;
      max_prob = softmax_probs[i];
    }
  }
  return softmax_probs;
}

template<typename T>
std::vector<unsigned int> ortcv::utils::math::argsort(const std::vector<T> &arr)
{
  ::__assert_type<T>();
  if (arr.empty()) return {};
  const unsigned int _size = arr.size();
  std::vector<unsigned int> indices;
  for (unsigned int i = 0; i < _size; ++i) indices.push_back(i);
  std::sort(indices.begin(), indices.end(),
            [&arr](const unsigned int a, const unsigned int b)
            { return arr[a] > arr[b]; });
  return indices;
}

template<typename T>
std::vector<unsigned int> ortcv::utils::math::argsort(const T *arr, unsigned int _size)
{
  ::__assert_type<T>();
  if (_size == 0 || arr == nullptr) return {};
  std::vector<unsigned int> indices;
  for (unsigned int i = 0; i < _size; ++i) indices.push_back(i);
  std::sort(indices.begin(), indices.end(),
            [arr](const unsigned int a, const unsigned int b)
            { return arr[a] > arr[b]; });
  return indices;
}

template<typename T>
T ortcv::utils::math::cosine_similarity(const std::vector<T> &a, const std::vector<T> &b)
{
  ::__assert_type<T>();
  T zero_vale = static_cast<T>(0);
  if (a.empty() || b.empty() || (a.size() != b.size())) return zero_vale;
  const unsigned int _size = a.size();
  T mul_a = zero_vale, mul_b = zero_vale, mul_ab = zero_vale;
  for (unsigned int i = 0; i < _size; ++i)
  {
    mul_a += a[i] * a[i];
    mul_b += b[i] * b[i];
    mul_ab += a[i] * b[i];
  }
  if (mul_a == zero_vale || mul_b == zero_vale) return zero_vale;
  return static_cast<T>(mul_ab / (std::sqrt(mul_a) * std::sqrt(mul_b)));
}

#endif //LITE_AI_ORT_CORE_ORT_UTILS_H
