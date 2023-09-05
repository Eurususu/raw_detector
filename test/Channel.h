#ifndef __CHANNEL_H__
#define __CHANNEL_H__

#include <utils.hpp>
#include "CUDACtxMan/CUDACtxMan.h"
#include <ptr_queue.hpp>
#include "AQCodec/AQCodec.h"
#include <functional>

struct Frames {
    int width, height;
    uint64_t ts;
    uint8_t *data;
    ~Frames();
};

class Channel : public IDecoderCallback
{
public:
    Channel(string strChnID, std::shared_ptr<PtrQueue<Frames>> frame_queue);
    ~Channel();

    virtual void OnVideoDecodedData(char* pGpuData, int32_t nLen, uint64_t ulTimestamp, int32_t nWidth, int32_t nHeight);
    virtual void OnVideDecodeTimestamp(uint64_t ulTimestamp);
    virtual uint64_t GetVideoDecodeTimestamp();

    void Open(string strFile, int start = 0, int size = -1);
    void Stop();
    void FastForward();
    void ThreadReadH264File(int start, int size);
    void ThreadReadAqmsFile(int start, int size);
    static void ConvertNV12ToRGB(const uint8_t *yuvBuf, uint8_t *rbgBuf,int width,int height,int linesize);
    
    bool b_stop = false;
    FILE* m_pFile;
    IDecoder* m_pDecoder;
    uint64_t m_ulTimestamp;
    thread m_thread;
    char* m_pDataRGB;
    string m_strChnID;
    bool b_ff = false;
    std::shared_ptr<PtrQueue<Frames>> m_queue;
};

#endif