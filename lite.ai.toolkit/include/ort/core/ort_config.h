//
// Created by DefTruth on 2021/3/29.
//

#ifndef LITE_AI_ORT_CORE_ORT_CONFIG_H
#define LITE_AI_ORT_CORE_ORT_CONFIG_H

#include "ort_defs.h"

#include <cmath>
#include <vector>
#include <cassert>
#include <locale.h>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include "opencv2/opencv.hpp"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

#ifdef USE_CUDA
#include "onnxruntime/core/providers/cuda/cuda_provider_factory.h"
#endif

namespace core {}

#endif //LITE_AI_ORT_CORE_ORT_CONFIG_H
