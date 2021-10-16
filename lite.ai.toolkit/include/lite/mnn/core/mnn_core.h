//
// Created by DefTruth on 2021/10/6.
//

#ifndef LITE_AI_TOOLKIT_MNN_CORE_MNN_CORE_H
#define LITE_AI_TOOLKIT_MNN_CORE_MNN_CORE_H

#include "mnn_config.h"
#include "mnn_handler.h"
#include "mnn_types.h"

namespace mnncv
{
  class LITE_EXPORTS MNNNanoDet;                     // [0] * reference: https://github.com/RangiLyu/nanodet
  class LITE_EXPORTS MNNNanoDetEfficientNetLite;     // [1] * reference: https://github.com/RangiLyu/nanodet
  class LITE_EXPORTS MNNRobustVideoMatting;          // [2] * reference: https://github.com/PeterL1n/RobustVideoMatting
  class LITE_EXPORTS MNNYoloX;                       // [3] * reference: https://github.com/Megvii-BaseDetection/YOLOX
}

namespace mnncv
{
  using mnncore::BasicMNNHandler;
}

namespace mnnnlp
{
  using mnncore::BasicMNNHandler;
}

namespace mnnasr
{
  using mnncore::BasicMNNHandler;
}

#endif //LITE_AI_TOOLKIT_MNN_CORE_MNN_CORE_H
