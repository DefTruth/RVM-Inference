//
// Created by DefTruth on 2021/10/18.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_RVM_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_RVM_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  class LITE_EXPORTS TNNRobustVideoMatting
  {
  public:
    explicit TNNRobustVideoMatting(const std::string &_proto_path,
                                   const std::string &_model_path,
                                   unsigned int _num_threads = 1);

    ~TNNRobustVideoMatting();

  private:
    const char *log_id = nullptr;
    const char *proto_path = nullptr;
    const char *model_path = nullptr;
    // Note, tnn:: actually is TNN_NS::, I prefer the first one.
    std::shared_ptr<tnn::TNN> net;
    std::shared_ptr<tnn::Instance> instance;

  private:
    std::vector<float> scale_vals = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    std::vector<float> bias_vals = {0.f, 0.f, 0.f}; // RGB
    // hardcode input node names, hint only.
    // downsample_ratio has been freeze while onnx exported
    // and, the input size of each input has been freeze, also.
    std::vector<std::string> input_names = {
        "src",
        "r1i",
        "r2i",
        "r3i",
        "r4i"
    };
    // hardcode output node names, hint only.
    std::vector<std::string> output_names = {
        "fgr",
        "pha",
        "r1o",
        "r2o",
        "r3o",
        "r4o"
    };
    bool context_is_update = false;
    bool context_is_initialized = false;

  private:
    const unsigned int num_threads; // initialize at runtime.
    // multi inputs, rxi will be update inner video matting process.
    std::shared_ptr<tnn::Mat> src_mat;
    std::shared_ptr<tnn::Mat> r1i_mat;
    std::shared_ptr<tnn::Mat> r2i_mat;
    std::shared_ptr<tnn::Mat> r3i_mat;
    std::shared_ptr<tnn::Mat> r4i_mat;
    // input size , initialize at runtime.
    int input_height;
    int input_width;
    tnn::DataFormat input_data_format;  // e.g DATA_FORMAT_NHWC
    tnn::MatType input_mat_type; // e.g NCHW_FLOAT
    tnn::DeviceType input_device_type; // only CPU, namely ARM or X86
    tnn::DeviceType output_device_type; // only CPU, namely ARM or X86
    tnn::DeviceType network_device_type; // e.g DEVICE_X86 DEVICE_NAIVE DEVICE_ARM
    std::map<std::string, tnn::DimsVector> input_shapes;
    std::map<std::string, tnn::DimsVector> output_shapes;
    unsigned int src_size;
    unsigned int r1i_size;
    unsigned int r2i_size;
    unsigned int r3i_size;
    unsigned int r4i_size;

    // un-copyable
  protected:
    TNNRobustVideoMatting(const TNNRobustVideoMatting &) = delete; //
    TNNRobustVideoMatting(TNNRobustVideoMatting &&) = delete; //
    TNNRobustVideoMatting &operator=(const TNNRobustVideoMatting &) = delete; //
    TNNRobustVideoMatting &operator=(TNNRobustVideoMatting &&) = delete; //

  private:
    void print_debug_string(); // debug information

  private:
    void transform(const cv::Mat &mat); //

    void initialize_instance(); // init net & instance

    void initialize_context();

    int value_size_of(tnn::DimsVector &shape);

    void generate_matting(std::shared_ptr<tnn::Instance> &_instance,
                          types::MattingContent &content,
                          int img_h, int img_w);

    void update_context(std::shared_ptr<tnn::Instance> &_instance);

  public:
    /**
     * Image Matting Using RVM(https://github.com/PeterL1n/RobustVideoMatting)
     * @param mat: cv::Mat BGR HWC
     * @param content: types::MattingContent to catch the detected results.
     * @param video_mode: false by default.
     * See https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/inference_zh_Hans.md
     */
    void detect(const cv::Mat &mat, types::MattingContent &content, bool video_mode = false);

    /**
     * Video Matting Using RVM(https://github.com/PeterL1n/RobustVideoMatting)
     * @param video_path: eg. xxx/xxx/input.mp4
     * @param output_path: eg. xxx/xxx/output.mp4
     * @param contents: vector of MattingContent to catch the detected results.
     * @param save_contents: false by default, whether to save MattingContent.
     * See https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/inference_zh_Hans.md
     * @param writer_fps: FPS for VideoWriter, 20 by default.
     */
    void detect_video(const std::string &video_path,
                      const std::string &output_path,
                      std::vector<types::MattingContent> &contents,
                      bool save_contents = false,
                      unsigned int writer_fps = 20);

  };
}

#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_RVM_H
