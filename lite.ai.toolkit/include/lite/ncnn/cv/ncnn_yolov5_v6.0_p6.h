//
// Created by DefTruth on 2021/11/11.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CV_NCNN_YOLOV5_V6_0_P6_H
#define LITE_AI_TOOLKIT_NCNN_CV_NCNN_YOLOV5_V6_0_P6_H

#include "lite/ncnn/core/ncnn_core.h"

namespace ncnncv
{
  class LITE_EXPORTS NCNNYoloV5_V_6_0_P6
  {
  private:
    ncnn::Net *net = nullptr;
    const char *log_id = nullptr;
    const char *param_path = nullptr;
    const char *bin_path = nullptr;
    std::vector<const char *> input_names;
    std::vector<const char *> output_names;
    std::vector<int> input_indexes;
    std::vector<int> output_indexes;

  private:
    // nested classes
    typedef struct
    {
      int grid0;
      int grid1;
      int stride;
      float width;
      float height;
    } YoloV5Anchor;

    typedef struct
    {
      float r;
      int dw;
      int dh;
      int new_unpad_w;
      int new_unpad_h;
      bool flag;
    } YoloV5ScaleParams;

  public:
    explicit NCNNYoloV5_V_6_0_P6(const std::string &_param_path,
                                 const std::string &_bin_path,
                                 unsigned int _num_threads = 1,
                                 int _input_height = 640,
                                 int _input_width = 640); //
    ~NCNNYoloV5_V_6_0_P6();

  private:
    const unsigned int num_threads; // initialize at runtime.
    // target image size after resize
    const int input_height; // 640/1280
    const int input_width; // 640/1280

    const float mean_vals[3] = {0.f, 0.f, 0.f}; // RGB
    const float norm_vals[3] = {1.0 / 255.f, 1.0 / 255.f, 1.0 / 255.f};
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

    std::vector<unsigned int> strides = {8, 16, 32, 64}; // p6 have 4 detection heads
    std::unordered_map<unsigned int, std::vector<YoloV5Anchor>> center_anchors;
    bool center_anchors_is_update = false;

  protected:
    NCNNYoloV5_V_6_0_P6(const NCNNYoloV5_V_6_0_P6 &) = delete; //
    NCNNYoloV5_V_6_0_P6(NCNNYoloV5_V_6_0_P6 &&) = delete; //
    NCNNYoloV5_V_6_0_P6 &operator=(const NCNNYoloV5_V_6_0_P6 &) = delete; //
    NCNNYoloV5_V_6_0_P6 &operator=(NCNNYoloV5_V_6_0_P6 &&) = delete; //

  private:
    void print_debug_string();

    void transform(const cv::Mat &mat_rs, ncnn::Mat &in);

    void resize_unscale(const cv::Mat &mat,
                        cv::Mat &mat_rs,
                        int target_height,
                        int target_width,
                        YoloV5ScaleParams &scale_params);

    // only generate once
    void generate_anchors(unsigned int target_height, unsigned int target_width);

    void generate_bboxes_single_stride(const YoloV5ScaleParams &scale_params,
                                       ncnn::Mat &det_pred,
                                       unsigned int stride,
                                       float score_threshold,
                                       float img_height,
                                       float img_width,
                                       std::vector<types::Boxf> &bbox_collection);

    void generate_bboxes(const YoloV5ScaleParams &scale_params,
                         ncnn::Extractor &extractor,
                         std::vector<types::Boxf> &bbox_collection,
                         float score_threshold, float img_height,
                         float img_width);

    void nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
             float iou_threshold, unsigned int topk, unsigned int nms_type);

  public:
    void detect(const cv::Mat &mat,
                std::vector<types::Boxf> &detected_boxes,
                float score_threshold = 0.25f, float iou_threshold = 0.45f,
                unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);

  };

}

#endif //LITE_AI_TOOLKIT_NCNN_CV_NCNN_YOLOV5_V6_0_P6_H
