#include "Decoder.h"

#include "opencv2/highgui.hpp"
// #include "opencv2/cudaimgproc.hpp"
#include "AQVideoHeader.hpp"
#include "utils.hpp"


IDecoder *IDecoder::Create(int nMaxWidth, int nMaxHeight, uint32_t nCodecType, IDecoderCallback &rCallback)
{
    return new CDecoder(nMaxWidth, nMaxHeight, nCodecType, (EVideoSurfaceFormat)0, (EPixelColorFormat)1, rCallback);
}

CDecoder::CDecoder(int nMaxWidth, int nMaxHeight, uint32_t nCodecType, EVideoSurfaceFormat eVideoSurfaceFormat,
                   EPixelColorFormat ePixelColorFormat, IDecoderCallback &rNotify, CUstream cuStream)
    : m_nMaxWidth(nMaxWidth), m_nMaxHeight(nMaxHeight), m_nWidth(0), m_nHeight(0), m_nCodecType(nCodecType),
      m_eCodecType(cudaVideoCodec_H264), m_eVideoSurfaceFormat(eVideoSurfaceFormat),
      m_ePixelColorFormat(ePixelColorFormat), m_cCuvidStream(cuStream), m_cDecoder(0), m_cContext(0), m_cVidCtxLock(0),
      m_cParser(0), m_rNotify(rNotify), m_ulTimestamp(0), m_bFull(false)
{
}

CDecoder::~CDecoder()
{
}

bool CDecoder::CheckGPUCaps()
{
    CUVIDDECODECAPS sDecodecaps;
    memset(&sDecodecaps, 0, sizeof(sDecodecaps));
    sDecodecaps.eCodecType = m_eCodecType;
    sDecodecaps.eChromaFormat = m_eChromaFormat;
    sDecodecaps.nBitDepthMinus8 = m_nBitDepthMinus8;
    cuCtxPushCurrent(m_cContext);
    cuvidGetDecoderCaps(&sDecodecaps);
    cuCtxPopCurrent(NULL);

    if (!sDecodecaps.bIsSupported)
    {
        INFO("Can't support the codec type\n");
        return false;
    }

    if (m_nMaxWidth > (int)sDecodecaps.nMaxWidth || m_nMaxWidth > (int)sDecodecaps.nMaxHeight)
    {
        INFO("Can't support the resolution: Width %d, Height %d\n", m_nMaxWidth, m_nMaxHeight);
        return false;
    }

    if ((m_nMaxWidth >> 4) * (m_nMaxWidth >> 4) > (int)sDecodecaps.nMaxMBCount)
    {
        INFO("Can't support the resolution: Width %d, Height %d\n", m_nMaxWidth, m_nMaxHeight);
        return false;
    }
    return true;
}

bool CDecoder::Init(string strChnID, bool bFirstChn, int nGpuID)
{
    m_strChnID = strChnID;
    m_bFirstChn = bFirstChn;
    m_cContext = CCUDACtxMan::GetCtx(nGpuID);
    if (cuvidCtxLockCreate(&m_cVidCtxLock, m_cContext) != CUDA_SUCCESS)
    {
        INFO("Fail to create Context\n");
        return false;
    }

    if (m_nCodecType == 0)
    {
        m_eCodecType = cudaVideoCodec_H264;
    }
    else if (m_nCodecType == 1)
    {
        m_eCodecType = cudaVideoCodec_HEVC;
    }
    else
    {
        INFO("Don't support codec type: %d, available types: 0=h264, 1=h265\n", m_nCodecType);
        return false;
    }

    if (m_eVideoSurfaceFormat == Nv12)
    {
        m_nBitDepthMinus8 = 0;
    }
    else
    {
        m_nBitDepthMinus8 = 2;
    }

    m_eChromaFormat = cudaVideoChromaFormat_420;

    if (!CheckGPUCaps())
    {
        INFO("Codec not supported on this GPU\n");
        return false;
    }

    if (m_nMaxWidth == 0)
    {
        m_nMaxWidth = 3840;
    }
    if (m_nMaxHeight == 0)
    {
        m_nMaxHeight = 2160;
    }

    // Create video decoder
    CUVIDDECODECREATEINFO sDecodeInfo = {0};
    sDecodeInfo.CodecType = m_eCodecType;
    sDecodeInfo.ulMaxWidth = m_nMaxWidth;
    sDecodeInfo.ulMaxHeight = m_nMaxHeight;
    sDecodeInfo.ulWidth = m_nMaxWidth;
    sDecodeInfo.ulHeight = m_nMaxHeight;
    sDecodeInfo.ulNumDecodeSurfaces = 3;
    sDecodeInfo.ChromaFormat = m_eChromaFormat;
    sDecodeInfo.ulTargetWidth = sDecodeInfo.ulWidth;
    sDecodeInfo.ulTargetHeight = sDecodeInfo.ulHeight;
    sDecodeInfo.ulNumOutputSurfaces = 2;
    sDecodeInfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    sDecodeInfo.vidLock = m_cVidCtxLock;
    sDecodeInfo.bitDepthMinus8 = m_nBitDepthMinus8;
    // INFO("m_nBitDepthMinus8: %d\n", m_nBitDepthMinus8);
    if (m_eVideoSurfaceFormat == Nv12)
    {
        sDecodeInfo.OutputFormat = cudaVideoSurfaceFormat_NV12;
    }
    else
    {
        sDecodeInfo.OutputFormat = cudaVideoSurfaceFormat_P016;
    }

    bool bSuccess = false;
    for (int i = 0; i < 10; i++)
    {
        if (cuvidCreateDecoder(&m_cDecoder, &sDecodeInfo) != CUDA_SUCCESS)
        {
            INFO("Fail to create NvDecoder, try again\n");
            usleep(2000);
        }
        else
        {
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        INFO("Fail to create NvDecoder\n");
        return false;
    }
    m_nBPP = sDecodeInfo.bitDepthMinus8 > 0 ? 2 : 1;
    m_nDstPitch = m_nMaxWidth * m_nBPP;
    m_nWidthInBytes = m_nMaxWidth * m_nBPP;

    // Careate Parser
    CUVIDPARSERPARAMS sVideoParserParameters;
    memset(&sVideoParserParameters, 0, sizeof(CUVIDPARSERPARAMS));
    sVideoParserParameters.CodecType = m_eCodecType;
    // sVideoParserParameters.ulMaxNumDecodeSurfaces = sDecodeInfo.ulNumDecodeSurfaces;
    sVideoParserParameters.ulMaxNumDecodeSurfaces = 1;
    sVideoParserParameters.ulMaxDisplayDelay =
        0; // this flag is needed so the parser will push frames out to the decoder as quickly as it can
    sVideoParserParameters.pUserData = this;
    sVideoParserParameters.pfnSequenceCallback =
        HandleVideoSequence; // Called before decoding frames and/or whenever there is a format change
    sVideoParserParameters.pfnDecodePicture =
        HandlePictureDecode; // Called when a picture is ready to be decoded (decode order)
    sVideoParserParameters.pfnDisplayPicture =
        HandlePictureDisplay; // Called whenever a picture is ready to be displayed (display order)
    if (cuvidCreateVideoParser(&m_cParser, &sVideoParserParameters) != CUDA_SUCCESS)
    {
        INFO("Fail to create Parser\n");
        return false;
    }
    // additional 2*sizeof(int32_t) bytes to store Width and Height
    cuCtxPushCurrent(m_cContext);
    cuMemAlloc((CUdeviceptr *)&m_pDecodedFrameBufGPU, m_nMaxWidth * m_nMaxHeight * 4);
    cuMemAlloc((CUdeviceptr *)&m_pColor32FrameBuf, m_nMaxWidth * m_nMaxHeight * 4);
    // cuMemAlloc((CUdeviceptr *)&m_pColor32FrameBuf, m_nMaxWidth*m_nMaxWidth*4 );
    // INFO("DecodedFrameSize: %d\n",DecodedFrameSize(m_nMaxWidth, m_nMaxHeight));
    cuCtxPopCurrent(NULL);

    // m_cTimestampQueue.Reset();
    m_nGpuID = nGpuID;
    if (!m_bMainRuning)
    {
        m_bMainRuning = true;
        m_tMainThread = thread(&CDecoder::MainThread, this);
        auto handle = m_tMainThread.native_handle();
#ifdef _WIN32
        SetThreadDescription(handle, L"Decoder");
#else
        pthread_setname_np(handle, "Decoder");
#endif
    }

    return true;
}

void CDecoder::Close()
{
    m_bMainRuning = false;
    if (m_tMainThread.joinable())
    {
        m_tMainThread.join();
    }
    {
        lock_guard<mutex> l(m_mutexPack);
        while (!m_listPack.empty())
        {
            m_listPack.pop_front();
        }
    }
    if (m_cParser != 0 && cuvidDestroyVideoParser(m_cParser) != CUDA_SUCCESS)
    {
        INFO("Fail to Close Destroy Parser\n");
    }
    if (m_cDecoder != 0 && cuvidDestroyDecoder(m_cDecoder) != CUDA_SUCCESS)
    {
        INFO("Fail to destroy Decoder\n");
    }
    if (m_cVidCtxLock != 0 && cuvidCtxLockDestroy(m_cVidCtxLock) != CUDA_SUCCESS)
    {
        INFO("Fail to destroy CtxLock\n");
    }
    m_cVidCtxLock = 0;
    m_cDecoder = 0;
    m_cParser = 0;
    m_cContext = 0;

    cuCtxPushCurrent(m_cContext);
    cuMemFree((CUdeviceptr)m_pDecodedFrameBufGPU);
    cuMemFree((CUdeviceptr)m_pColor32FrameBuf);
    cuCtxPopCurrent(NULL);
}

void CDecoder::SetDelay(int nFrames)
{
    m_nDelayFrames = nFrames;
}

bool CDecoder::IsFull()
{
    return m_bFull;
}

int CDecoder::GetChromaPlaneCount(cudaVideoChromaFormat eChromaFormat)
{
    int numPlane = 1;
    switch (eChromaFormat)
    {
    case cudaVideoChromaFormat_Monochrome:
        numPlane = 0;
        break;
    case cudaVideoChromaFormat_420:
    case cudaVideoChromaFormat_422:
        numPlane = 1;
        break;
    case cudaVideoChromaFormat_444:
        numPlane = 2;
        break;
    }

    return numPlane;
}

float CDecoder::GetChromaHeightFactor(cudaVideoChromaFormat eChromaFormat)
{
    float factor = 0.5;
    switch (eChromaFormat)
    {
    case cudaVideoChromaFormat_Monochrome:
        factor = 0.0;
        break;
    case cudaVideoChromaFormat_420:
        factor = 0.5;
        break;
    case cudaVideoChromaFormat_422:
        factor = 1.0;
        break;
    case cudaVideoChromaFormat_444:
        factor = 1.0;
        break;
    }

    return factor;
}

int CDecoder::DecodedFrameSize(int nWidth, int nHeight)
{
    return nWidth *
           (nHeight + nHeight * GetChromaHeightFactor(m_eChromaFormat) * GetChromaPlaneCount(m_eChromaFormat)) * m_nBPP;
}

void CDecoder::Decode(char *pFrameData, int nLen)
{
    AQ_HEADER_SET_TIMESTAMP(pFrameData, AQT::ntohl64(AQ_HEADER_GET_TIMESTAMP(pFrameData)));
    // AQ_HEADER_SET_CODEC_ID(pFrameData, ntohs(AQ_HEADER_GET_CODEC_ID(pFrameData)));
    shared_ptr<VideoPacket> pVideoPack(new VideoPacket);
    pVideoPack->nLen = nLen;
    pVideoPack->pData = new char[nLen];
    memcpy(pVideoPack->pData, pFrameData, nLen);
    lock_guard<mutex> l(m_mutexPack);
    m_listPack.push_back(pVideoPack);
    if (m_listPack.size() > 100)
    {
        m_bFull = true;
    }
}

int CUDAAPI CDecoder::HandleVideoSequence(void *pUserData, CUVIDEOFORMAT *pVideoFormat)
{

    int nDecodeSurface = 3;
    CDecoder *pDecoder = reinterpret_cast<CDecoder *>(pUserData);

    pDecoder->m_nWidth = pVideoFormat->display_area.right - pVideoFormat->display_area.left;
    pDecoder->m_nHeight = pVideoFormat->display_area.bottom - pVideoFormat->display_area.top;
    // pDecoder->m_nWidth = pDecoder->m_nMaxWidth;
    // pDecoder->m_nHeight = pDecoder->m_nMaxHeight;

    CUVIDRECONFIGUREDECODERINFO sReconfigParams = {0};
    sReconfigParams.ulWidth = pDecoder->m_nWidth;
    sReconfigParams.ulHeight = pDecoder->m_nHeight;
    sReconfigParams.ulNumDecodeSurfaces = nDecodeSurface;
    sReconfigParams.ulTargetWidth = pDecoder->m_nWidth;
    sReconfigParams.ulTargetHeight = pDecoder->m_nHeight;

    cuCtxPushCurrent(pDecoder->m_cContext);
    cuvidReconfigureDecoder(pDecoder->m_cDecoder, &sReconfigParams);
    cuCtxPopCurrent(NULL);

    pDecoder->m_nDstPitch = pDecoder->m_nWidth * pDecoder->m_nBPP;
    pDecoder->m_nWidthInBytes = pDecoder->m_nWidth * pDecoder->m_nBPP;

    return nDecodeSurface;
}

int CUDAAPI CDecoder::HandlePictureDecode(void *pUserData, CUVIDPICPARAMS *pPicParams)
{

    CDecoder *pDecoder = reinterpret_cast<CDecoder *>(pUserData);
    cuvidDecodePicture(pDecoder->m_cDecoder, pPicParams);
    return 1;
}

int CUDAAPI CDecoder::HandlePictureDisplay(void *pUserData, CUVIDPARSERDISPINFO *pPicParams)
{
    uint64_t ulTimestamp = 0;
    CDecoder *pDecoder = reinterpret_cast<CDecoder *>(pUserData);

    CUVIDPROCPARAMS oVideoProcessingParameters = {0};
    oVideoProcessingParameters.output_stream = pDecoder->m_cCuvidStream;

    CUdeviceptr pDecodedFrame = 0;
    unsigned int nDecodedPitch = 0;
    CUresult cResult = cuvidMapVideoFrame(pDecoder->m_cDecoder, pPicParams->picture_index, &pDecodedFrame,
                                          &nDecodedPitch, &oVideoProcessingParameters);
    if (cResult != CUDA_SUCCESS)
        return 0;

    CUVIDGETDECODESTATUS sDecodeStatus;
    memset(&sDecodeStatus, 0, sizeof(sDecodeStatus));
    cResult = cuvidGetDecodeStatus(pDecoder->m_cDecoder, pPicParams->picture_index, &sDecodeStatus);
    // uint64_t *pTimestampArray = pDecoder->m_cTimestampQueue.Next(DECODER_TIMESTAMP_QUEUE_TIMEOUT);
    // if (pTimestampArray == NULL)
    // {
    //     DEBUG("Decode Error occurred for picture %d, no timestamp\n", pPicParams->picture_index);
    //     return 0;
    // }

    if (cResult == CUDA_SUCCESS)
    {

        if (sDecodeStatus.decodeStatus != cuvidDecodeStatus_Error &&
            sDecodeStatus.decodeStatus != cuvidDecodeStatus_Error_Concealed)
        {

            // Copy resolution in to the begin of the frame memory in the queue
            cuCtxPushCurrent(pDecoder->m_cContext);
            uint8_t *pDecodedFrameBuf = pDecoder->m_pDecodedFrameBufGPU;
            // cudaMemcpy(pDecodedFrameBuf, &pDecoder->m_nWidth, sizeof(int32_t), cudaMemcpyHostToDevice);
            // cudaMemcpy(pDecodedFrameBuf + sizeof(int32_t), &pDecoder->m_nHeight, sizeof(int32_t),
            // cudaMemcpyHostToDevice); pDecodedFrameBuf += sizeof(int32_t) * 2;

            CUDA_MEMCPY2D m = {0};
            m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            m.srcDevice = pDecodedFrame;
            m.srcPitch = nDecodedPitch;

            m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            m.dstDevice = (CUdeviceptr)(m.dstHost = pDecodedFrameBuf);

            m.dstPitch = pDecoder->m_nDstPitch;
            m.WidthInBytes = pDecoder->m_nWidthInBytes;

            m.Height = pDecoder->m_nHeight;
            cuMemcpy2DAsync(&m, pDecoder->m_cCuvidStream);

            m.srcDevice = pDecodedFrame + m.srcPitch * ((m.Height + 1) & ~1);
            m.dstDevice = (CUdeviceptr)(m.dstHost = pDecodedFrameBuf + pDecoder->m_nHeight * m.dstPitch);
            m.Height = pDecoder->m_nHeight * 0.5;

            cuMemcpy2DAsync(&m, pDecoder->m_cCuvidStream);

            // m.srcDevice = (CUdeviceptr)((uint8_t *)pDecodedFrame + m.srcPitch * pDecoder->m_nHeight);
            // m.dstDevice = (CUdeviceptr)(m.dstHost = pDecodedFrameBuf + m.dstPitch * pDecoder->m_nHeight);
            // m.Height *= GetChromaHeightFactor(pDecoder->m_eChromaFormat);
            // cuMemcpy2DAsync(&m, pDecoder->m_cCuvidStream);

            cuStreamSynchronize(pDecoder->m_cCuvidStream);

            // Convert from Decoded Frame to pColor32FrameBuf
            int iPitch = pDecoder->m_nWidth * 4;
            // uint8_t *pColor32FrameBuf = (uint8_t *)(pDecoder->m_pColor32FrameQueue->GetPushBuf(0)); // here 0 means
            // using the default buffer size Copy resolution in to the begin of the frame memory in the queue uint8_t
            // *pColor32FrameBuf = pDecoder->m_pColor32FrameBuf; uint8_t* pColor32FrameBuf,  *pData;
            int32_t nDataLen = pDecoder->m_nWidth * pDecoder->m_nHeight * 3;
            // cuMemAlloc((CUdeviceptr *)&pColor32FrameBuf, nDataLen);
            // pData = pColor32FrameBuf;

            // cudaMemcpy(pColor32FrameBuf, &pDecoder->m_nWidth, sizeof(int32_t), cudaMemcpyHostToDevice);
            // cudaMemcpy(pColor32FrameBuf + sizeof(int32_t), &pDecoder->m_nHeight, sizeof(int32_t),
            // cudaMemcpyHostToDevice);

            // cudaMemcpy(pColor32FrameBuf, &pDecoder->m_ulTimestamp, sizeof(uint64_t), cudaMemcpyHostToDevice);
            // INFO("@@@Decode Timestamp: %lu\n", pDecoder->m_ulTimestamp);
            //  uint64_t iTimestamp = pTimestampArray[0];
            //  cudaMemcpy(pColor32FrameBuf + sizeof(int32_t) * 2, &iTimestamp, sizeof(uint64_t),
            //  cudaMemcpyHostToDevice);
            // pColor32FrameBuf +=  sizeof(uint64_t) ;

            if (pDecoder->m_eVideoSurfaceFormat == Nv12)
            { // Nv12
                if (pDecoder->m_ePixelColorFormat == BGRA_32)
                {
                    // ConvertNV12ToRGB(pDecodedFrameBuf,pDecoder->m_pColor32FrameBuf,pDecoder->m_nWidth,
                    // pDecoder->m_nHeight,m.dstPitch);
                    //  Nv12ToColor32<BGRA32>(pDecodedFrameBuf, pDecoder->m_nWidth, pDecoder->m_pColor32FrameBuf,
                    //  iPitch, pDecoder->m_nWidth, pDecoder->m_nHeight, NULL, NULL);
                }
                else
                {
                    Nv12ToColor32<RGBA32>(pDecodedFrameBuf, pDecoder->m_nWidth, pDecoder->m_pColor32FrameBuf, iPitch,
                                          pDecoder->m_nWidth, pDecoder->m_nHeight, NULL, NULL);
                }
            }
            else
            { // P016
                if (pDecoder->m_ePixelColorFormat == BGRA_32)
                {
                    P016ToColor32<BGRA32>(pDecodedFrameBuf, 2 * pDecoder->m_nWidth, pDecoder->m_pColor32FrameBuf,
                                          iPitch, pDecoder->m_nWidth, pDecoder->m_nHeight, NULL, NULL);
                }
                else
                {
                    P016ToColor32<RGBA32>(pDecodedFrameBuf, 2 * pDecoder->m_nWidth, pDecoder->m_pColor32FrameBuf,
                                          iPitch, pDecoder->m_nWidth, pDecoder->m_nHeight, NULL, NULL);
                }
            }

            if (sDecodeStatus.decodeStatus != 0)
            {
                pDecoder->m_rNotify.OnVideoDecodedData((char *)pDecodedFrameBuf, nDataLen, pPicParams->timestamp,
                                                       pDecoder->m_nWidth, pDecoder->m_nHeight);
            }
            else
            {
                // cuMemFree((CUdeviceptr)pColor32FrameBuf);
            }

            cuCtxPopCurrent(NULL);
        }
        else
        {
            INFO("Decode Error occurred for picture %d, sDecodeStatus.decodeStatus: %d\n", pPicParams->picture_index,
                 (int)sDecodeStatus.decodeStatus);
        }
    }

    if (cResult == CUDA_SUCCESS)
    {
        cuvidUnmapVideoFrame(pDecoder->m_cDecoder, pDecodedFrame);
        return 1;
    }
    else
    {
        return 0;
    }
}

void CDecoder::MainThread()
{
    bool bStart = false;
    uint64_t ulTimeLast = 0;
    uint64_t ulTimeFirst = 0;
    uint64_t ulTimeFirstEncode = 0;
    int nCount = 0;
    m_nDelayFrames = 0;
    while (m_bMainRuning)
    {
        shared_ptr<VideoPacket> pVideoPack;
        if (!bStart)
        {
            if (!m_bFirstChn)
            {
                bStart = true;
            }
            else
            {
                lock_guard<mutex> l(m_mutexPack);
                if (m_listPack.size() > 5 + m_nDelayFrames)
                {
                    bStart = true;
                }
            }
        }

        if (!bStart)
        {
            AQT::AQSleep(5);
            continue;
        }
        uint64_t ulNow = AQT::AQGetTimestamp();
        uint64_t ulStart = AQT::AQGetTimestamp();

        {

            lock_guard<mutex> l(m_mutexPack);
            if (m_bFirstChn)
            {
                if (m_listPack.size() > 5 + m_nDelayFrames)
                {
                    pVideoPack = m_listPack.front();
                    m_listPack.pop_front();
                }
                else
                {
                    if (m_bFirstChn)
                        bStart = false;
                }
            }
            else
            {
                if (m_listPack.size() > 0)
                {
                    pVideoPack = m_listPack.front();
                    m_listPack.pop_front();
                }
            }
            if (m_listPack.size() < 80) {
                m_bFull = false;
            }
        }
        CUresult cResult;

        if (pVideoPack)
        {

            m_ulTimestamp = AQ_HEADER_GET_TIMESTAMP(pVideoPack->pData);

            CUVIDSOURCEDATAPACKET sDataPacket;
            sDataPacket.flags = CUVID_PKT_TIMESTAMP;
            sDataPacket.payload_size = pVideoPack->nLen - AQ_HEADER_LENGTH();
            sDataPacket.payload = (unsigned char *)(pVideoPack->pData + AQ_HEADER_LENGTH());
            sDataPacket.timestamp = m_ulTimestamp;
            // sDataPacket.payload_size = pVideoPack->nLen;
            // sDataPacket.payload = (unsigned char *)(pVideoPack->pData);
            cResult = cuvidParseVideoData(m_cParser, &sDataPacket);

            if (cResult != CUDA_SUCCESS)
            {
            }
            ulTimeLast = ulNow;

        }

        AQT::AQSleep(1);
    }
}