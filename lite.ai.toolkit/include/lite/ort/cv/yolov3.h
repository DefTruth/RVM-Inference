//
// Created by DefTruth on 2021/5/23.
//

#ifndef LITE_AI_ORT_CV_YOLOV3_H
#define LITE_AI_ORT_CV_YOLOV3_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS YoloV3
  {
  private:
    Ort::Env ort_env;
    Ort::Session *ort_session = nullptr;
    std::vector<const char *> input_node_names;
    std::vector<std::vector<int64_t>> input_node_dims; // >=1 inputs.
    std::vector<size_t> input_tensor_sizes;
    Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<const char *> output_node_names;
    const LITEORT_CHAR *onnx_path = nullptr;
    const char *log_id = nullptr;
    unsigned int num_outputs = 1;
    unsigned int num_inputs = 1;
    std::vector<float> input_1_values_handler;
    std::vector<float> image_shape_values_handler;

  protected:
    const unsigned int num_threads; // initialize at runtime.

  public:
    // yolov3 is an multi-inputs & multi-outputs & dynamic shape
    // (dynamic input shape: batch,input_height,input_width)
    // & (dynamic output shape: bactch, num_anchors, num_selected)
    explicit YoloV3(const std::string &_onnx_path, unsigned int _num_threads = 1);

    ~YoloV3();

  protected:
    YoloV3(const YoloV3 &) = delete;

    YoloV3(YoloV3 &&) = delete;

    YoloV3 &operator=(const YoloV3 &) = delete;

    YoloV3 &operator=(YoloV3 &&) = delete;

  private:
    static constexpr const float mean_val = 0.f;
    static constexpr const float scale_val = 1.0 / 255.f;
    static constexpr const unsigned int input_height = 416;
    static constexpr const unsigned int input_width = 416;
    static constexpr const unsigned int batch_size = 1;

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

  private:
    std::vector<Ort::Value> transform(const std::vector<cv::Mat> &mats);

    void generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                         std::vector<Ort::Value> &output_tensors); // rescale & exclude

    void print_debug_string();

  public:
    void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes);

  };
}

#endif //LITE_AI_ORT_CV_YOLOV3_H
