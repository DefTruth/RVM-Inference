//
// Created by DefTruth on 2021/7/17.
//

#ifndef LITE_AI_LITE_AI_DEFS_H
#define LITE_AI_LITE_AI_DEFS_H

#include <iostream>

#ifndef LITE_EXPORTS
# if (defined _WIN32 || defined WINCE || defined __CYGWIN__)
#   define LITE_EXPORTS __declspec(dllexport)
# elif defined __GNUC__ && __GNUC__ >= 4 && (defined(__APPLE__))
#   define LITE_EXPORTS __attribute__ ((visibility ("default")))
# endif
#endif

#if (defined _WIN32 || defined WINCE || defined __CYGWIN__)
# define LITE_WIN32
#elif defined __GNUC__ && __GNUC__ >= 4 && (defined(__APPLE__))
# define LITE_UNIX
#endif

#ifdef LITE_WIN32
# define NONMINMAX
#endif

#ifndef LITE_EXPORTS
# define LITE_EXPORTS
#endif

#ifndef __unused
# define __unused
#endif

#endif //LITE_AI_LITE_AI_DEFS_H
