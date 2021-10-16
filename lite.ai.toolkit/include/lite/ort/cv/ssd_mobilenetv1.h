//
// Created by DefTruth on 2021/6/5.
//

#ifndef LITE_AI_ORT_CV_SSD_MOBILENETV1_H
#define LITE_AI_ORT_CV_SSD_MOBILENETV1_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS SSDMobileNetV1
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
    std::vector<uchar> input_values_handler; // uint8

  protected:
    const unsigned int num_threads; // initialize at runtime.

  public:
    // dynamic input and multi dynamic outputs.
    explicit SSDMobileNetV1(const std::string &_onnx_path, unsigned int _num_threads = 1);

    ~SSDMobileNetV1();

  protected:
    SSDMobileNetV1(const SSDMobileNetV1 &) = delete;

    SSDMobileNetV1(SSDMobileNetV1 &&) = delete;

    SSDMobileNetV1 &operator=(const SSDMobileNetV1 &) = delete;

    SSDMobileNetV1 &operator=(SSDMobileNetV1 &&) = delete;

  private:
    static constexpr const unsigned int input_height = 480;
    static constexpr const unsigned int input_width = 640;
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
    enum NMS
    {
      HARD = 0, BLEND = 1, OFFSET = 2
    };

  private:
    Ort::Value transform(const cv::Mat &mat);//
    void generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                         std::vector<Ort::Value> &output_tensors,
                         float score_threshold, float img_height,
                         float img_width); // rescale & exclude
    void nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
             float iou_threshold, unsigned int topk, unsigned int nms_type); //
    void print_debug_string();


  public:
    void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                float score_threshold = 0.25f, float iou_threshold = 0.45f,
                unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);

  };
}

#endif //LITE_AI_ORT_CV_SSD_MOBILENETV1_H
