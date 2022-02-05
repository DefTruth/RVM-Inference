//
// Created by DefTruth on 2021/10/18.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CV_NCNN_YOLOX_H
#define LITE_AI_TOOLKIT_NCNN_CV_NCNN_YOLOX_H

#include "lite/ncnn/core/ncnn_core.h"

namespace ncnncv
{
  class LITE_EXPORTS NCNNYoloX
  {
  private:
    ncnn::Net *net = nullptr;
    const char *log_id = nullptr;
    const char *param_path = nullptr;
    const char *bin_path = nullptr;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<int> input_indexes;
    std::vector<int> output_indexes;

  public:
    explicit NCNNYoloX(const std::string &_param_path,
                       const std::string &_bin_path,
                       unsigned int _num_threads = 1,
                       int _input_height = 640,
                       int _input_width = 640); //
    ~NCNNYoloX();

  private:
    // nested classes
    typedef struct GridAndStride
    {
      int grid0;
      int grid1;
      int stride;
    } YoloXAnchor;

    typedef struct
    {
      float r;
      int dw;
      int dh;
      int new_unpad_w;
      int new_unpad_h;
      bool flag;
    } YoloXScaleParams;

  private:
    const unsigned int num_threads; // initialize at runtime.
    // target image size after resize, might use 416 for small model(nano/tiny)
    const int input_height; // 640(s/m/l/x), 416(nano/tiny)
    const int input_width; // 640(s/m/l/x), 416(nano/tiny)

    const float mean_vals[3] = {255.f * 0.485f, 255.f * 0.456, 255.f * 0.406f};
    const float norm_vals[3] = {1.f / (255.f * 0.229f), 1.f / (255.f * 0.224f), 1.f / (255.f * 0.225f)};

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
    static constexpr const unsigned int max_nms = 30000;

  protected:
    NCNNYoloX(const NCNNYoloX &) = delete; //
    NCNNYoloX(NCNNYoloX &&) = delete; //
    NCNNYoloX &operator=(const NCNNYoloX &) = delete; //
    NCNNYoloX &operator=(NCNNYoloX &&) = delete; //

  private:
    void print_debug_string();

    void transform(const cv::Mat &mat_rs, ncnn::Mat &in);

    void resize_unscale(const cv::Mat &mat,
                        cv::Mat &mat_rs,
                        int target_height,
                        int target_width,
                        YoloXScaleParams &scale_params);

    void generate_anchors(const int target_height,
                          const int target_width,
                          std::vector<int> &strides,
                          std::vector<YoloXAnchor> &anchors);

    void generate_bboxes(const YoloXScaleParams &scale_params,
                         std::vector<types::Boxf> &bbox_collection,
                         ncnn::Extractor &extractor,
                         float score_threshold, int img_height,
                         int img_width); // rescale & exclude

    void nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
             float iou_threshold, unsigned int topk, unsigned int nms_type);

  public:
    void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                float score_threshold = 0.25f, float iou_threshold = 0.45f,
                unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);
  };
}

#endif //LITE_AI_TOOLKIT_NCNN_CV_NCNN_YOLOX_H
