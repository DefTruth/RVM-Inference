//
// Created by DefTruth on 2021/8/14.
//

#ifndef LITE_AI_ORT_CV_EFFICIENTDET_H
#define LITE_AI_ORT_CV_EFFICIENTDET_H

#include "ort/core/ort_core.h"

namespace ortcv
{
  typedef struct
  {
    float y1;
    float x1;
    float y2;
    float x2;
  } EfficientDetAnchor;

  class LITE_EXPORTS EfficientDet : public BasicOrtHandler
  {
  public:
    explicit EfficientDet(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~EfficientDet() override = default;

  private:
    const float mean_vals[3] = {0.406f, 0.456, 0.486f};
    const float scale_vals[3] = {1.0f / 0.225f, 1.0f / 0.224f, 1.0f / 0.229f};
    static constexpr const float anchor_scale = 4.0f; // for d0-d6
    std::vector<int> pyramid_levels = {3, 4, 5, 6, 7}; // for d0-d6
    std::vector<float> strides = {
        std::pow(2.0f, 3.0f),
        std::pow(2.0f, 4.0f),
        std::pow(2.0f, 5.0f),
        std::pow(2.0f, 6.0f),
        std::pow(2.0f, 7.0f)
    };
    std::vector<float> scales = {
        std::pow(2.0f, 0.0f),
        std::pow(2.0f, 1.0f / 3.0f),
        std::pow(2.0f, 2.0f / 3.0f)
    };
    std::vector<std::vector<float>> ratios = {
        {1.0f, 1.0f},
        {1.4f, 0.7f},
        {0.7f, 1.4f}
    };
    std::vector<EfficientDetAnchor> anchors_buffer;

    const char *class_names[90] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "", "backpack", "umbrella", "", "", "handbag", "tie",
        "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "", "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
        "cake", "chair", "couch", "potted plant", "bed", "", "dining table", "", "", "toilet", "", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    };
    enum NMS
    {
      HARD = 0, BLEND = 1, OFFSET = 2
    };
    static constexpr const unsigned int max_nms = 30000;

  private:
    Ort::Value transform(const cv::Mat &mat) override; //

    void generate_anchors(const float target_height, const float target_width);

    void generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                         std::vector<Ort::Value> &output_tensors,
                         float score_threshold, float img_height,
                         float img_width); // rescale & exclude

    void nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
             float iou_threshold, unsigned int topk, unsigned int nms_type);

  public:
    void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                float score_threshold = 0.2f, float iou_threshold = 0.2f,
                unsigned int topk = 1000, unsigned int nms_type = NMS::OFFSET);

  };
}

#endif //LITE_AI_ORT_CV_EFFICIENTDET_H
