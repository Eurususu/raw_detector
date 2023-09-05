#ifndef __DECODER__
#define __DECODER__

#include "CUDACtxMan/CUDACtxMan.h"
#include "ColorSpace.h"
#include "NvCodecUtils.h"
#include "cuviddec.h"
#include "nvcuvid.h"
#include <cuda.h>

#include "AQCodec/AQCodec.h"
#include "AQHeader.h"

enum EVideoSurfaceFormat
{
    Nv12 = 0,
    P016
};
enum EPixelColorFormat
{
    RGBA_32 = 0,
    BGRA_32
};

class CDecoder : public IDecoder
{
  public:
    CDecoder(int nMaxWidth, int nMaxHeight, uint32_t nCodecType, EVideoSurfaceFormat eVideoSurfaceFormat,
             EPixelColorFormat ePixelColorFormat, IDecoderCallback &rNotify, CUstream cuStream = 0);
    ~CDecoder();

  public:
    bool Init(string strChnID, bool bFirstChn, int nGpuID) override;
    void Decode(char *pFrameData, int nLen) override;
    void Close() override;
    void SetDelay(int nFrames) override;
    bool IsFull() override;

  private:
    void MainThread();
    int DecodedFrameSize(int nWidth, int nHeight);
    bool CheckGPUCaps();
    static int GetChromaPlaneCount(cudaVideoChromaFormat eChromaFormat);
    static float GetChromaHeightFactor(cudaVideoChromaFormat eChromaFormat);

    static int CUDAAPI HandleVideoSequence(void *pUserData, CUVIDEOFORMAT *pVideoFormat);
    static int CUDAAPI HandlePictureDecode(void *pUserData, CUVIDPICPARAMS *pPicParams);
    static int CUDAAPI HandlePictureDisplay(void *pUserData, CUVIDPARSERDISPINFO *pPicParams);

  public:
    int m_bFull;
    int m_nGpuID;
    int m_nMaxWidth;
    int m_nMaxHeight;
    int m_nWidth;
    int m_nHeight;
    int m_nBitDepthMinus8;
    int m_nBPP;
    int m_nDstPitch;
    int m_nWidthInBytes;
    cudaVideoChromaFormat m_eChromaFormat;
    uint32_t m_nCodecType;
    cudaVideoCodec m_eCodecType;
    EVideoSurfaceFormat m_eVideoSurfaceFormat;
    EPixelColorFormat m_ePixelColorFormat;
    CUstream m_cCuvidStream;
    CUvideodecoder m_cDecoder;
    CUcontext m_cContext;
    CUvideoctxlock m_cVidCtxLock;
    CUvideoparser m_cParser;
    uint8_t *m_pDecodedFrameBufGPU;
    uint8_t *m_pColor32FrameBuf;
    IDecoderCallback &m_rNotify;
    int m_nIndex;
    uint64_t m_ulTimestamp;
    // int                                 m_nSensorID;
    int m_nDelayFrames{0};
    string m_strChnID;
    bool m_bFirstChn;

    list<shared_ptr<VideoPacket>> m_listPack;
    mutex m_mutexPack;
    mutex m_mtx_gear;
    bool m_bMainRuning{false};
    thread m_tMainThread;

    list<uint64_t> m_listDecodeTimestamp;
    mutex m_mtxDecodeTimestamp;
};

#endif