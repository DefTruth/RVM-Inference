//
// Created by DefTruth on 2021/10/6.
//

#ifndef LITE_AI_TOOLKIT_MNN_CORE_MNN_HANDLER_H
#define LITE_AI_TOOLKIT_MNN_CORE_MNN_HANDLER_H

#include "mnn_config.h"

namespace mnncore
{
  class LITE_EXPORTS BasicMNNHandler
  {
  protected:
    std::shared_ptr<MNN::Interpreter> mnn_interpreter;
    MNN::Session *mnn_session = nullptr;
    MNN::Tensor *input_tensor = nullptr; // single input.
    MNN::ScheduleConfig schedule_config;
    std::shared_ptr<MNN::CV::ImageProcess> pretreat; // init at subclass
    const char *log_id = nullptr;
    const char *mnn_path = nullptr;

  protected:
    const unsigned int num_threads; // initialize at runtime.
    int input_batch;
    int input_channel;
    int input_height;
    int input_width;
    int dimension_type;

  protected:
    explicit BasicMNNHandler(const std::string &_mnn_path, unsigned int _num_threads = 1);

    virtual ~BasicMNNHandler();

    // un-copyable
  protected:
    BasicMNNHandler(const BasicMNNHandler &) = delete; //
    BasicMNNHandler(BasicMNNHandler &&) = delete; //
    BasicMNNHandler &operator=(const BasicMNNHandler &) = delete; //
    BasicMNNHandler &operator=(BasicMNNHandler &&) = delete; //

  protected:
    virtual void transform(const cv::Mat &mat) = 0; // ? needed ?

  private:
    void initialize_handler();

    void print_debug_string();

  };
}

#endif //LITE_AI_TOOLKIT_MNN_CORE_MNN_HANDLER_H
