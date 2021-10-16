//
// Created by DefTruth on 2021/3/14.
//

#ifndef LITE_AI_ORT_CV_YOLOV4_H
#define LITE_AI_ORT_CV_YOLOV4_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS YoloV4 : public BasicOrtHandler
  {
  public:
    explicit YoloV4(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~YoloV4() override = default;

  private:
    static constexpr const float mean_val = 0.f;
    static constexpr const float scale_val = 1.0 / 255.f;
    const char *class_names[20] = {
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor"
    }; // voc 20 classes
    enum NMS
    {
      HARD = 0, BLEND = 1, OFFSET = 2
    };
    static constexpr const unsigned int max_nms = 30000;

  private:
    Ort::Value transform(const cv::Mat &mat) override;

    void generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                         std::vector<Ort::Value> &output_tensors,
                         float score_threshold, float img_height,
                         float img_width); // rescale & exclude
    void nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
             float iou_threshold, unsigned int topk, unsigned int nms_type);

  public:
    void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                float score_threshold = 0.25f, float iou_threshold = 0.45f,
                unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);

  };
}

#endif //LITE_AI_ORT_CV_YOLOV4_H
