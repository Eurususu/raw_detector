#ifndef __ENCODER_H__
#define __ENCODER_H__

#include "NvEncoderCuda.h"
#include "NvEncoderCLIOptions.h"
#include "cuviddec.h"
#include "nvcuvid.h"
#include "NvCodecUtils.h"
#include "CUDACtxMan/CUDACtxMan.h"
#include <cuda.h>
#include "AQCodec/AQCodec.h"

class CEncoder : public NvEncoderNotify
               , public IEncoder
{
public:
    CEncoder(IEncoderCallback& rCallback, int nGpuID);
    ~CEncoder();

    bool Init(int nCodecType, int nMaxWidth, int nMaxHeight) override;
    void EncodeOneFrame(char *pFrame, bool bForceIDR) override;

    void EndEncode();
    void SetEncodeConfig(int BitRate, int RateModel);
    void SetEncodeParam(int nWidth, int nHeight, int nBitRate, int nFrameRate, int nRateMode);

    void OnVideoEncodedData(char* pData, int32_t nLen, uint64_t ulTimestamp, int32_t nWidth, int32_t nHeight, bool bKeyFrame = false);

private:
    NvEncoderCuda*      m_pEnc;
    CUcontext           m_cContext;
    CUmemorytype        m_cSrcMemoryType;
    IEncoderCallback&   m_rCallback;
    int                 m_nGpuID;
    int                 m_nBitRate{8000};
    int                 m_nRateModel{0};
};

#endif