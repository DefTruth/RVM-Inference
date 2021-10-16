//
// Created by DefTruth on 2021/10/9.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CV_NCNN_NANODET_EFFICIENTDET_LITE_DEPRECIATED_H
#define LITE_AI_TOOLKIT_NCNN_CV_NCNN_NANODET_EFFICIENTDET_LITE_DEPRECIATED_H

#include "lite/ncnn/core/ncnn_core.h"

namespace ncnncv
{
  typedef struct
  {
    float grid0;
    float grid1;
    float stride;
  } NanoLiteDepreciatedCenterPoint;

  typedef struct
  {
    float ratio;
    int dw;
    int dh;
    bool flag;
  } NanoLiteDepreciatedScaleParams;

  class LITE_EXPORTS NCNNNanoDetEfficientNetLiteDepreciated : public BasicNCNNHandler
  {
  public:
    explicit NCNNNanoDetEfficientNetLiteDepreciated(const std::string &_param_path,
                                                    const std::string &_bin_path,
                                                    unsigned int _num_threads = 1,
                                                    int _input_height = 320,
                                                    int _input_width = 320); //
    ~NCNNNanoDetEfficientNetLiteDepreciated() override = default;

  private:
    const float mean_vals[3] = {103.53f, 116.28f, 123.675f}; // BGR
    const float norm_vals[3] = {0.017429f, 0.017507f, 0.017125f};

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
    static constexpr const bool is_dynamic_input = false;

    // multi-levels center points
    int input_height = 320;
    int input_width = 320;
    std::vector<unsigned int> strides = {8, 16, 32};
    std::unordered_map<unsigned int, std::vector<NanoLiteDepreciatedCenterPoint>> center_points;
    bool center_points_is_update = false;

  private:

    void transform(const cv::Mat &mat_rs, ncnn::Mat &in) override;

    void resize_unscale(const cv::Mat &mat,
                        cv::Mat &mat_rs,
                        int target_height,
                        int target_width,
                        NanoLiteDepreciatedScaleParams &scale_params);

    // only generate once
    void generate_points(unsigned int target_height, unsigned int target_width);

    void generate_bboxes_single_stride(const NanoLiteDepreciatedScaleParams &scale_params,
                                       ncnn::Mat &cls_pred,
                                       ncnn::Mat &dis_pred,
                                       unsigned int stride,
                                       float score_threshold,
                                       float img_height,
                                       float img_width,
                                       std::vector<types::Boxf> &bbox_collection);

    void generate_bboxes(const NanoLiteDepreciatedScaleParams &scale_params,
                         std::vector<types::Boxf> &bbox_collection,
                         ncnn::Extractor &extractor,
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

#endif //LITE_AI_TOOLKIT_NCNN_CV_NCNN_NANODET_EFFICIENTDET_LITE_DEPRECIATED_H
