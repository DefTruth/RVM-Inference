//
// Created by DefTruth on 2021/10/2.
//

#ifndef LITE_AI_TOOLKIT_ORT_CV_NANODET_H
#define LITE_AI_TOOLKIT_ORT_CV_NANODET_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  typedef struct
  {
    float grid0;
    float grid1;
    float stride;
  } NanoCenterPoint;

  typedef struct
  {
    float ratio;
    int dw;
    int dh;
    bool flag;
  } NanoScaleParams;

  class LITE_EXPORTS NanoDet : public BasicOrtHandler
  {
  public:
    explicit NanoDet(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~NanoDet() override = default;

  private:
    const float mean_vals[3] = {103.53f, 116.28f, 123.675f}; // BGR
    const float scale_vals[3] = {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f};

    const char *class_names[80] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    enum NMS
    {
      HARD = 0, BLEND = 1, OFFSET = 2
    };
    static constexpr const unsigned int nms_pre = 1000;
    static constexpr const unsigned int max_nms = 30000;

    // multi-levels center points
    std::vector<unsigned int> strides = {8, 16, 32};
    std::unordered_map<unsigned int, std::vector<NanoCenterPoint>> center_points;
    bool center_points_is_update = false;

  private:
    Ort::Value transform(const cv::Mat &mat_rs) override; // without resize

    void resize_unscale(const cv::Mat &mat,
                        cv::Mat &mat_rs,
                        int target_height,
                        int target_width,
                        NanoScaleParams &scale_params);

    // only generate once
    void generate_points(unsigned int target_height, unsigned int target_width);

    void generate_bboxes_single_stride(const NanoScaleParams &scale_params,
                                       Ort::Value &cls_pred,
                                       Ort::Value &dis_pred,
                                       unsigned int stride,
                                       float score_threshold,
                                       float img_height,
                                       float img_width,
                                       std::vector<types::Boxf> &bbox_collection);

    void generate_bboxes(const NanoScaleParams &scale_params,
                         std::vector<types::Boxf> &bbox_collection,
                         std::vector<Ort::Value> &output_tensors,
                         float score_threshold, float img_height,
                         float img_width); // rescale & exclude

    void nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
             float iou_threshold, unsigned int topk, unsigned int nms_type);

  public:
    /**
     * @param mat cv::Mat BGR format
     * @param detected_boxes vector of Boxf to catch detected boxes.
     * @param score_threshold default 0.45f, only keep the result which >= score_threshold.
     * @param iou_threshold default 0.3f, iou threshold for NMS.
     * @param topk default 100, maximum output boxes after NMS.
     * @param nms_type the method.
     */
    void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                float score_threshold = 0.45f, float iou_threshold = 0.3f,
                unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);

  };

}

#endif //LITE_AI_TOOLKIT_ORT_CV_NANODET_H
