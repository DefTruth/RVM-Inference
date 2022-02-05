//
// Created by DefTruth on 2021/10/31.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CORE_NCNN_CUSTOM_H
#define LITE_AI_TOOLKIT_NCNN_CORE_NCNN_CUSTOM_H

#include "ncnn_config.h"

// YOLOX|YOLOP|YOLOR ... use the same focus in yolov5
class YoloV5Focus : public ncnn::Layer
{
public:
  YoloV5Focus()
  {
    one_blob_only = true;
  }

  virtual int forward(const ncnn::Mat &bottom_blob, ncnn::Mat &top_blob, const ncnn::Option &opt) const;
};

ncnn::Layer* YoloV5Focus_layer_creator(void * /*userdata*/);

#endif //LITE_AI_TOOLKIT_NCNN_CORE_NCNN_CUSTOM_H
