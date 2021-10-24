//
// Created by DefTruth on 2021/10/17.
//

#ifndef LITE_AI_TOOLKIT_TNN_CORE_TNN_CORE_H
#define LITE_AI_TOOLKIT_TNN_CORE_TNN_CORE_H

#include "tnn_config.h"
#include "tnn_handler.h"
#include "tnn_types.h"

namespace tnncv
{
  class LITE_EXPORTS TNNNanoDet;                     // [0] * reference: https://github.com/RangiLyu/nanodet
  class LITE_EXPORTS TNNNanoDetEfficientNetLite;     // [1] * reference: https://github.com/RangiLyu/nanodet
  class LITE_EXPORTS TNNRobustVideoMatting;          // [2] * reference: https://github.com/PeterL1n/RobustVideoMatting
  class LITE_EXPORTS TNNYoloX;                       // [3] * reference: https://github.com/Megvii-BaseDetection/YOLOX
  class LITE_EXPORTS TNNYOLOP;                       // [4] * reference: https://github.com/hustvl/YOLOP
}

namespace tnncv
{
  using tnncore::BasicTNNHandler;
}

namespace tnnnlp
{
  using tnncore::BasicTNNHandler;
}

namespace tnnasr
{
  using tnncore::BasicTNNHandler;
}

#endif //LITE_AI_TOOLKIT_TNN_CORE_TNN_CORE_H
