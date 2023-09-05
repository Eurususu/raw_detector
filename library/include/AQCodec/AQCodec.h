#ifndef __AQCODEC_H__
#define __AQCODEC_H__


class IDecoderCallback
{
  public:
    virtual void OnVideoDecodedData(char *pGpuData, int32_t nLen, uint64_t ulTimestamp, int32_t nWidth,
                                    int32_t nHeight) = 0;
    virtual void OnVideDecodeTimestamp(uint64_t ulTimestamp) = 0;
    virtual uint64_t GetVideoDecodeTimestamp() = 0;
};

class IDecoder
{
  public:
    IDecoder()
    {
    }
    virtual ~IDecoder()
    {
    }

    static IDecoder *Create(int nMaxWidth, int nMaxHeight, uint32_t nCodecType, IDecoderCallback &rCallback);
    virtual bool Init(string strChnID, bool bFirstChn, int nGpuID) = 0;
    virtual void Decode(char *pFrameData, int nLen) = 0;
    virtual void Close() = 0;
    virtual void SetDelay(int nFrames) = 0;
    virtual bool IsFull() = 0;
    virtual void ReadyStop() = 0;
    virtual bool IsRunning() = 0;
};

class IEncoderCallback
{
  public:
    virtual ~IEncoderCallback()
    {
    }
    virtual void OnVideoEncodedData(char *pData, int32_t nLen, uint64_t ulTimestamp, int32_t nWidth, int32_t nHeight,
                                    bool bKeyFrame = false) = 0;
};

class IEncoder
{
  public:
    IEncoder()
    {
    }
    virtual ~IEncoder()
    {
    }

    static IEncoder *Create(int nGpuID, IEncoderCallback &rCallback);
    virtual bool Init(int nCodecType, int nMaxWidth, int nMaxHeight) = 0;
    virtual void EncodeOneFrame(char *pFrame, bool bForceIDR) = 0;
    virtual void SetEncodeParam(int nWidth, int nHeight, int nBitRate, int nFrameRate, int nRateMode) = 0;
};

#endif