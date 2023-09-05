#include "Encoder.h"
#include "AQHeader.h"
#include "Def.h"

IEncoder* IEncoder::Create(int nGpuID, IEncoderCallback& rCallback)
{
    return new CEncoder(rCallback, nGpuID);
}

CEncoder::CEncoder(IEncoderCallback& rCallback, int nGpuID)
    : m_pEnc(NULL)
    , m_cContext(CCUDACtxMan::GetCtx(nGpuID))
    , m_cSrcMemoryType(CU_MEMORYTYPE_DEVICE)
    , m_rCallback(rCallback)
    , m_nGpuID(nGpuID)
{
}

CEncoder::~CEncoder()
{
    if (m_pEnc != NULL)
    {
        m_pEnc->DestroyEncoder();
        delete m_pEnc;
        m_pEnc = NULL;
    }
}

bool CEncoder::Init(int nCodecType, int nMaxWidth, int nMaxHeight)
{
    NV_ENC_BUFFER_FORMAT eFormat=NV_ENC_BUFFER_FORMAT_ARGB;
    if (m_pEnc == NULL)
    {
        try
        {
            m_pEnc = new NvEncoderCuda(m_cContext, nMaxWidth, nMaxHeight, eFormat, *this);
        }
        catch(const NVENCException e)
        {
            std::cerr << e.what() << '\n';
            return false;            
        }
        
        
    }
    const char *pCodecStr = NULL;
    if (nCodecType == 0)
    {
        pCodecStr = "-codec h264 -gop 30";
    }
    else if (nCodecType == 1)
    {
        pCodecStr = "-codec hevc -gop 25";
    }
    else
    {
        INFO("ERROR: Not support this codec type=%d\n", nCodecType);
    }
    assert(pCodecStr != NULL);

    NvEncoderInitParam sEncodeCLIOptions(pCodecStr);
    NV_ENC_INITIALIZE_PARAMS sInitParams = {NV_ENC_INITIALIZE_PARAMS_VER};
    NV_ENC_CONFIG sEncodeConfig = {NV_ENC_CONFIG_VER};
    sInitParams.encodeConfig = &sEncodeConfig;
    sInitParams.encodeConfig->rcParams.rateControlMode = (m_nRateModel==0) ? NV_ENC_PARAMS_RC_CBR: NV_ENC_PARAMS_RC_VBR;
    sInitParams.encodeConfig->rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR_HQ;
    sInitParams.encodeConfig->rcParams.averageBitRate = m_nBitRate *1024;
    sInitParams.encodeConfig->rcParams.maxBitRate = m_nBitRate *1024 ;
    m_pEnc->SetEncodeConfig(m_nBitRate, m_nRateModel);
    m_pEnc->CreateDefaultEncoderParams(&sInitParams, sEncodeCLIOptions.GetEncodeGUID(), sEncodeCLIOptions.GetPresetGUID());
    sEncodeCLIOptions.SetInitParams(&sInitParams, eFormat);
    m_pEnc->CreateEncoder(&sInitParams);
    return true;
}

void CEncoder::SetEncodeConfig(int BitRate, int RateModel)
{
    m_nBitRate = BitRate;
    m_nRateModel = RateModel;   
}

void CEncoder::SetEncodeParam(int nWidth, int nHeight, int nBitRate, int nFrameRate, int nRateMode)
{
    if (m_pEnc->GetEncodeWidth() != nWidth || m_pEnc->GetEncodeHeight() != nHeight || m_pEnc->GetBitRate() != nBitRate || m_pEnc->GetRateMode() != nRateMode)
    {
        NV_ENC_CONFIG sEncodeConfig = {0};
        NV_ENC_INITIALIZE_PARAMS sInitializeParams = {0};
        sInitializeParams.encodeConfig = &sEncodeConfig;

        m_pEnc->GetInitializeParams(&sInitializeParams);
        NV_ENC_RECONFIGURE_PARAMS sReconfigureParams = {NV_ENC_RECONFIGURE_PARAMS_VER};
        memcpy(&sReconfigureParams.reInitEncodeParams, &sInitializeParams, sizeof(sInitializeParams));
        NV_ENC_CONFIG sReInitCodecConfig = {NV_ENC_CONFIG_VER};
        memcpy(&sReInitCodecConfig, sInitializeParams.encodeConfig, sizeof(sReInitCodecConfig));
        NV_ENC_PARAMS_RC_MODE eMode = NV_ENC_PARAMS_RC_CBR;
        switch (nRateMode)
        {
        case 0:
            eMode = NV_ENC_PARAMS_RC_CBR;
            break;
        case 1:
            eMode = NV_ENC_PARAMS_RC_VBR;
            break;
        case 2:
            eMode = NV_ENC_PARAMS_RC_CONSTQP;
            break;
        
        default:
            eMode = NV_ENC_PARAMS_RC_CBR;
            break;
        }
        m_nRateModel = nRateMode;
        m_nBitRate = nBitRate;

        sReInitCodecConfig.rcParams.rateControlMode = eMode;
        sReInitCodecConfig.rcParams.averageBitRate = nBitRate * 1024;
        sReInitCodecConfig.rcParams.maxBitRate = nBitRate * 1024;
        //sReInitCodecConfig.rcParams.


        sReconfigureParams.reInitEncodeParams.encodeConfig = &sReInitCodecConfig;
        sReconfigureParams.reInitEncodeParams.encodeWidth = nWidth;
        sReconfigureParams.reInitEncodeParams.encodeHeight = nHeight;
        sReconfigureParams.reInitEncodeParams.frameRateNum = 30;

    
        sReconfigureParams.reInitEncodeParams.darWidth = sReconfigureParams.reInitEncodeParams.encodeWidth;
        sReconfigureParams.reInitEncodeParams.darHeight = sReconfigureParams.reInitEncodeParams.encodeHeight;
        sReconfigureParams.forceIDR = true;
        m_pEnc->Reconfigure(&sReconfigureParams);

        Init(0, nWidth, nHeight);
    }
}

void CEncoder::EncodeOneFrame(char *pFrame, bool bForceIDR)
{
    assert(pFrame != NULL);
    cuCtxPushCurrent(m_cContext);
    int32_t iWidth = 0;
    int32_t iHeight = 0;
    uint64_t iTimestamp = 0;
    if (m_cSrcMemoryType == CU_MEMORYTYPE_HOST)
    {
        iWidth = GET_DECODED_CPU_FRAME_WIDTH(pFrame);
        iHeight = GET_DECODED_CPU_FRAME_HEIGHT(pFrame);
        iTimestamp = GET_DECODED_CPU_FRAME_TS(pFrame);
    }
    else
    {
        cudaMemcpy(&iWidth, pFrame, sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&iHeight, pFrame + sizeof(int32_t), sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&iTimestamp, pFrame + sizeof(int32_t) * 2, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }
    if (m_pEnc->GetEncodeWidth() != iWidth || m_pEnc->GetEncodeHeight() != iHeight)
    {
        NV_ENC_CONFIG sEncodeConfig = {0};
        NV_ENC_INITIALIZE_PARAMS sInitializeParams = {0};
        sInitializeParams.encodeConfig = &sEncodeConfig;

        m_pEnc->GetInitializeParams(&sInitializeParams);
        NV_ENC_RECONFIGURE_PARAMS sReconfigureParams = {NV_ENC_RECONFIGURE_PARAMS_VER};
        memcpy(&sReconfigureParams.reInitEncodeParams, &sInitializeParams, sizeof(sInitializeParams));
        NV_ENC_CONFIG sReInitCodecConfig = {NV_ENC_CONFIG_VER};
        memcpy(&sReInitCodecConfig, sInitializeParams.encodeConfig, sizeof(sReInitCodecConfig));
        sReconfigureParams.reInitEncodeParams.encodeConfig = &sReInitCodecConfig;
        sReconfigureParams.reInitEncodeParams.encodeWidth = iWidth;
        sReconfigureParams.reInitEncodeParams.encodeHeight = iHeight;
        sReconfigureParams.reInitEncodeParams.darWidth = sReconfigureParams.reInitEncodeParams.encodeWidth;
        sReconfigureParams.reInitEncodeParams.darHeight = sReconfigureParams.reInitEncodeParams.encodeHeight;
        sReconfigureParams.forceIDR = true;
        m_pEnc->Reconfigure(&sReconfigureParams);
    }

    char *pFrameData = GET_DECODED_FRAME_DATA(pFrame);

    const NvEncInputFrame *pEncoderInputFrame = m_pEnc->GetNextInputFrame();
    NvEncoderCuda::CopyToDeviceFrame(m_cContext,
                                     pFrameData, 0,
                                     (CUdeviceptr)pEncoderInputFrame->inputPtr, (int)pEncoderInputFrame->pitch,
                                     iWidth,
                                     iHeight,
                                     m_cSrcMemoryType,
                                     pEncoderInputFrame->bufferFormat,
                                     pEncoderInputFrame->chromaOffsets,
                                     pEncoderInputFrame->numChromaPlanes);

    if (bForceIDR)
    {
        NV_ENC_PIC_PARAMS sPicParams = {NV_ENC_PIC_PARAMS_VER};
        sPicParams.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR;
        m_pEnc->EncodeFrame(iWidth, iHeight, iTimestamp, &sPicParams);
    }
    else
    {
        m_pEnc->EncodeFrame(iWidth, iHeight, iTimestamp);
    }
    cuCtxPopCurrent(NULL);
}

void CEncoder::EndEncode()
{
    uint64_t iTimestamp = 0;
    m_pEnc->EndEncode(iTimestamp);
}

void CEncoder::OnVideoEncodedData(char* pData, int32_t nLen, uint64_t ulTimestamp, int32_t nWidth, int32_t nHeight, bool bKeyFrame)
{
    m_rCallback.OnVideoEncodedData(pData, nLen, ulTimestamp, nWidth, nHeight, bKeyFrame);
}