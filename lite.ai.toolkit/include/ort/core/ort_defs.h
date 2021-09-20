//
// Created by DefTruth on 2021/6/19.
//

#ifndef LITE_AI_ORT_CORE_ORT_DEFS_H
#define LITE_AI_ORT_CORE_ORT_DEFS_H

#include "lite/lite.ai.defs.h"

#define LITEORT_DEBUG 1

#ifdef LITE_WIN32
# define LITEORT_CHAR wchar_t
#else
# define LITEORT_CHAR char
#endif

#ifdef LITE_WIN32
# define NONMINMAX
#endif

#endif //LITE_AI_ORT_CORE_ORT_DEFS_H
