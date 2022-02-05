//
// Created by DefTruth on 2021/10/7.
//

#ifndef LITE_AI_TOOLKIT_UTILS_H
#define LITE_AI_TOOLKIT_UTILS_H

#include "types.h"

namespace lite
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

    LITE_EXPORTS cv::Mat draw_boxes_with_landmarks(const cv::Mat &mat, const std::vector<types::BoxfWithLandmarks> &boxes_kps, bool text = false);

    LITE_EXPORTS void draw_boxes_inplace(cv::Mat &mat_inplace, const std::vector<types::Boxf> &boxes);

    LITE_EXPORTS void draw_axis_inplace(cv::Mat &mat_inplace, const types::EulerAngles &euler_angles, float size = 50.f, int thickness = 2);

    LITE_EXPORTS void draw_landmarks_inplace(cv::Mat &mat, types::Landmarks &landmarks);

    LITE_EXPORTS void draw_age_inplace(cv::Mat &mat_inplace, types::Age &age);

    LITE_EXPORTS void draw_gender_inplace(cv::Mat &mat_inplace, types::Gender &gender);

    LITE_EXPORTS void draw_emotion_inplace(cv::Mat &mat_inplace, types::Emotions &emotions);

    LITE_EXPORTS void draw_boxes_with_landmarks_inplace(cv::Mat &mat_inplace, const std::vector<types::BoxfWithLandmarks> &boxes_kps, bool text = false);

    LITE_EXPORTS void hard_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output, float iou_threshold, unsigned int topk);

    LITE_EXPORTS void blending_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output, float iou_threshold, unsigned int topk);

    LITE_EXPORTS void offset_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output, float iou_threshold, unsigned int topk);

    namespace math
    {
      template<typename T=float>
      std::vector<float> softmax(const std::vector<T> &logits, unsigned int &max_id);

      template LITE_EXPORTS std::vector<float> softmax(const std::vector<float> &logits, unsigned int &max_id);

      template<typename T=float>
      std::vector<float> softmax(const T *logits, unsigned int _size, unsigned int &max_id);

      template LITE_EXPORTS std::vector<float> softmax(const float *logits, unsigned int _size, unsigned int &max_id);

      template<typename T=float>
      std::vector<unsigned int> argsort(const std::vector<T> &arr);

      template LITE_EXPORTS std::vector<unsigned int> argsort(const std::vector<float> &arr);

      template<typename T=float>
      std::vector<unsigned int> argsort(const T *arr, unsigned int _size);

      template LITE_EXPORTS std::vector<unsigned int> argsort(const float *arr, unsigned int _size);

      template<typename T=float>
      float cosine_similarity(const std::vector<T> &a, const std::vector<T> &b);

      template LITE_EXPORTS float cosine_similarity(const std::vector<float> &a, const std::vector<float> &b);
    }

  } // NAMESPACE UTILS
}

template<typename T>
std::vector<float> lite::utils::math::softmax(const T *logits, unsigned int _size, unsigned int &max_id)
{
  types::__assert_type<T>();
  if (_size == 0 || logits == nullptr) return {};
  float max_prob = 0.f, total_exp = 0.f;
  std::vector<float> softmax_probs(_size);
  for (unsigned int i = 0; i < _size; ++i)
  {
    softmax_probs[i] = std::exp((float) logits[i]);
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
std::vector<float> lite::utils::math::softmax(const std::vector<T> &logits, unsigned int &max_id)
{
  types::__assert_type<T>();
  if (logits.empty()) return {};
  const unsigned int _size = logits.size();
  float max_prob = 0.f, total_exp = 0.f;
  std::vector<float> softmax_probs(_size);
  for (unsigned int i = 0; i < _size; ++i)
  {
    softmax_probs[i] = std::exp((float) logits[i]);
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
std::vector<unsigned int> lite::utils::math::argsort(const std::vector<T> &arr)
{
  types::__assert_type<T>();
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
std::vector<unsigned int> lite::utils::math::argsort(const T *arr, unsigned int _size)
{
  types::__assert_type<T>();
  if (_size == 0 || arr == nullptr) return {};
  std::vector<unsigned int> indices;
  for (unsigned int i = 0; i < _size; ++i) indices.push_back(i);
  std::sort(indices.begin(), indices.end(),
            [arr](const unsigned int a, const unsigned int b)
            { return arr[a] > arr[b]; });
  return indices;
}

template<typename T>
float lite::utils::math::cosine_similarity(const std::vector<T> &a, const std::vector<T> &b)
{
  types::__assert_type<T>();
  float zero_vale = 0.f;
  if (a.empty() || b.empty() || (a.size() != b.size())) return zero_vale;
  const unsigned int _size = a.size();
  float mul_a = zero_vale, mul_b = zero_vale, mul_ab = zero_vale;
  for (unsigned int i = 0; i < _size; ++i)
  {
    mul_a += (float) a[i] * (float) a[i];
    mul_b += (float) b[i] * (float) b[i];
    mul_ab += (float) a[i] * (float) b[i];
  }
  if (mul_a == zero_vale || mul_b == zero_vale) return zero_vale;
  return (mul_ab / (std::sqrt(mul_a) * std::sqrt(mul_b)));
}

#endif //LITE_AI_TOOLKIT_UTILS_H
