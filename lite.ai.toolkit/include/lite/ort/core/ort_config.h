//
// Created by DefTruth on 2021/3/29.
//

#ifndef LITE_AI_ORT_CORE_ORT_CONFIG_H
#define LITE_AI_ORT_CORE_ORT_CONFIG_H

#include "ort_defs.h"
#include "lite/lite.ai.headers.h"

#ifdef ENABLE_ONNXRUNTIME
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#  ifdef USE_CUDA
#include "onnxruntime/core/providers/cuda/cuda_provider_factory.h"
#  endif
#endif

namespace core {}

#endif //LITE_AI_ORT_CORE_ORT_CONFIG_H
