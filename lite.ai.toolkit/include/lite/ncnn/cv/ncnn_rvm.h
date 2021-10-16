//
// Created by DefTruth on 2021/10/10.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CV_NCNN_RVM_H
#define LITE_AI_TOOLKIT_NCNN_CV_NCNN_RVM_H

#include "lite/ncnn/core/ncnn_core.h"

namespace ncnncv
{
  class LITE_EXPORTS NCNNRobustVideoMatting : public BasicNCNNHandler
  {
  public:
    explicit NCNNRobustVideoMatting(const std::string &_param_path,
                                    const std::string &_bin_path,
                                    unsigned int _num_threads = 1,
                                    int _input_height = 480,
                                    int _input_width = 640,
                                    unsigned int _variant_type = VARIANT::MOBILENETV3); //
    ~NCNNRobustVideoMatting() override = default;

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
    // will be update inner video matting process.
    ncnn::Mat r1i, r2i, r3i, r4i;
    // input size & variant_type, initialize at runtime.
    const int input_height;
    const int input_width;
    const unsigned int variant_type;

  private:

    void transform(const cv::Mat &mat, ncnn::Mat &in) override;

    void initialize_context();

    void generate_matting(ncnn::Extractor &extractor,
                          types::MattingContent &content,
                          int img_h, int img_w);

    void update_context(ncnn::Extractor &extractor);

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

#endif //LITE_AI_TOOLKIT_NCNN_CV_NCNN_RVM_H


























































