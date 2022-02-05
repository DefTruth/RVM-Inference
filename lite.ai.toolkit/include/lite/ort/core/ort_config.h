//
// Created by DefTruth on 2021/3/29.
//

#ifndef LITE_AI_ORT_CORE_ORT_CONFIG_H
#define LITE_AI_ORT_CORE_ORT_CONFIG_H

#include "ort_defs.h"
#include "lite/lite.ai.headers.h"

#ifdef ENABLE_ONNXRUNTIME
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
/* Need to define USE_CUDA macro manually by users who want to
 * enable onnxruntime and lite.ai.toolkit with CUDA support. It
 * seems that the latest onnxruntime will no longer pre-defined the
 * USE_CUDA macro and just let the decision make by users
 * who really know the environments of running device.*/
// #define USE_CUDA
#  ifdef USE_CUDA
#include "onnxruntime/core/providers/cuda/cuda_provider_factory.h"
#  endif
#endif

namespace core {}

#endif //LITE_AI_ORT_CORE_ORT_CONFIG_H
