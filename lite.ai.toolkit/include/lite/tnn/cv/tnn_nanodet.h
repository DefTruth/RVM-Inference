//
// Created by DefTruth on 2021/10/18.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_NANODET_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_NANODET_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  class LITE_EXPORTS TNNNanoDet : public BasicTNNHandler
  {
  public:
    explicit TNNNanoDet(const std::string &_proto_path,
                        const std::string &_model_path,
                        unsigned int _num_threads = 1); //
    ~TNNNanoDet() override = default;

  private:
    // nested classes
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

  private:
    // In TNN: x*scale + bias
    std::vector<float> scale_vals = {0.017429f, 0.017507f, 0.017125f}; // BGR
    std::vector<float> bias_vals = {-103.53f * 0.0174291f, -116.28f * 0.0175070f, -123.675f * 0.0171247f};

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
    void transform(const cv::Mat &mat_rs) override; // without resize

    void resize_unscale(const cv::Mat &mat,
                        cv::Mat &mat_rs,
                        int target_height,
                        int target_width,
                        NanoScaleParams &scale_params);

    // only generate once
    void generate_points(unsigned int target_height, unsigned int target_width);

    void generate_bboxes_single_stride(const NanoScaleParams &scale_params,
                                       const std::shared_ptr<tnn::Mat> &cls_pred,
                                       const std::shared_ptr<tnn::Mat> &dis_pred,
                                       unsigned int stride,
                                       float score_threshold,
                                       float img_height,
                                       float img_width,
                                       std::vector<types::Boxf> &bbox_collection);

    void generate_bboxes(const NanoScaleParams &scale_params,
                         std::vector<types::Boxf> &bbox_collection,
                         std::shared_ptr<tnn::Instance> &_instance,
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

#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_NANODET_H
