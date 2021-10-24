//
// Created by DefTruth on 2021/10/10.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_RVM_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_RVM_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNRobustVideoMatting
  {
  public:
    explicit MNNRobustVideoMatting(const std::string &_mnn_path,
                                   unsigned int _num_threads = 1,
                                   unsigned int _variant_type = VARIANT::MOBILENETV3); //
    ~MNNRobustVideoMatting();

  private:
    std::shared_ptr<MNN::Interpreter> mnn_interpreter;
    MNN::Session *mnn_session = nullptr;
    MNN::ScheduleConfig schedule_config;
    std::shared_ptr<MNN::CV::ImageProcess> pretreat; // init at runtime
    const char *log_id = nullptr;
    const char *mnn_path = nullptr;

  private:
    const float mean_vals[3] = {0.f, 0.f, 0.f}; // RGB
    const float norm_vals[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    // hardcode input node names, hint only.
    // downsample_ratio has been freeze while onnx exported
    // and, the input size of each input has been freeze, also.
    std::vector<const char *> input_node_names = {
        "src",
        "r1i",
        "r2i",
        "r3i",
        "r4i"
    };
    // hardcode output node names, hint only.
    std::vector<const char *> output_node_names = {
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
    enum VARIANT
    {
      MOBILENETV3 = 0,
      RESNET50 = 1
    };

    const unsigned int num_threads; // initialize at runtime.
    // multi inputs, rxi will be update inner video matting process.
    MNN::Tensor *src_tensor = nullptr;
    MNN::Tensor *r1i_tensor = nullptr;
    MNN::Tensor *r2i_tensor = nullptr;
    MNN::Tensor *r3i_tensor = nullptr;
    MNN::Tensor *r4i_tensor = nullptr;
    // input size & variant_type, initialize at runtime.
    const unsigned int variant_type;
    int input_height;
    int input_width;
    int dimension_type; // hint only
    unsigned int src_size;
    unsigned int r1i_size;
    unsigned int r2i_size;
    unsigned int r3i_size;
    unsigned int r4i_size;

    // un-copyable
  protected:
    MNNRobustVideoMatting(const MNNRobustVideoMatting &) = delete; //
    MNNRobustVideoMatting(MNNRobustVideoMatting &&) = delete; //
    MNNRobustVideoMatting &operator=(const MNNRobustVideoMatting &) = delete; //
    MNNRobustVideoMatting &operator=(MNNRobustVideoMatting &&) = delete; //

  private:
    void print_debug_string();

  private:
    void transform(const cv::Mat &mat_rs); // without resize

    void initialize_pretreat(); //

    void initialize_interpreter();

    void initialize_context();

    void generate_matting(const std::map<std::string, MNN::Tensor*> &output_tensors,
                          types::MattingContent &content,
                          int img_h, int img_w);

    void update_context(const std::map<std::string, MNN::Tensor*> &output_tensors);

  public:
    /**
     * Image Matting Using RVM(https://github.com/PeterL1n/RobustVideoMatting)
     * @param mat: cv::Mat BGR HWC
     * @param content: types::MattingContent to catch the detected results.
     * See https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/inference_zh_Hans.md
     */
    void detect(const cv::Mat &mat, types::MattingContent &content);
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


#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_RVM_H

























































