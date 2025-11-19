//Behrooz
#pragma once


#include "nvtx3/nvtx3.hpp"

// Helper class for RAII-style range marking
class NVTXRange {
public:
    NVTXRange(const char* name, uint32_t color = 0xFF00FF00) {
        nvtxEventAttributes_t eventAttrib = {0};
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.colorType = NVTX_COLOR_ARGB;
        eventAttrib.color = color;
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii = name;
        nvtxRangePushEx(&eventAttrib);
    }
    
    ~NVTXRange() {
        nvtxRangePop();
    }
};

// Convenient macros
#define NVTX_RANGE(name) NVTXRange nvtx_range(name)
#define NVTX_RANGE_COLOR(name, color) NVTXRange nvtx_range(name, color)
#define NVTX_MARK(name) nvtxMark(name)
