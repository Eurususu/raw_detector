#include "Channel.h"
#include <AQVideoHeader.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nppi.h>


Frames::~Frames() {
    if (data != nullptr) cudaFree(data);
}
Channel::Channel(string strChnID, std::shared_ptr<PtrQueue<Frames>> frame_queue)
    : m_strChnID(strChnID), m_queue(frame_queue)
{
}

Channel::~Channel()
{

    Stop();
    m_pDecoder->Close();
    cudaFree(m_pDataRGB);
}

void Channel::OnVideoDecodedData(char *pGpuData, int32_t nLen, uint64_t ulTimestamp, int32_t nWidth, int32_t nHeight)
{

    std::shared_ptr<Frames> frame(new Frames);
    frame->height = nHeight;
    frame->width = nWidth;
    frame->ts = ulTimestamp;
    cudaMalloc(&frame->data, nWidth * nHeight * 3);
    const Npp8u *nv12[2] = {(Npp8u *)pGpuData, (Npp8u *)pGpuData + nHeight * nWidth};
    nppiNV12ToBGR_8u_P2C3R(nv12, nWidth, frame->data, nWidth * 3, {nWidth, nHeight});
    m_queue->push_wait(frame);

}

void Channel::OnVideDecodeTimestamp(uint64_t ulTimestamp)
{
    m_ulTimestamp = ulTimestamp;
}

uint64_t Channel::GetVideoDecodeTimestamp()
{
    return m_ulTimestamp;
}

void Channel::Open(string strFile, int start, int size)
{
    b_stop = false;
    cudaMalloc((void **)&m_pDataRGB, 3840 * 2160 * 3);
    m_pDecoder = IDecoder::Create(3840, 2160, 0, *this);
    m_pDecoder->Init(m_strChnID, false, 0);
    m_pFile = fopen(strFile.c_str(), "rb");
    auto idx = strFile.find_last_of('.');
    std::string suffix = strFile.substr(idx + 1, strFile.size() - idx - 1);
    if (suffix == "aqms")
        m_thread = thread(&Channel::ThreadReadAqmsFile, this, start, size);
    else if (suffix == "h264")
        m_thread = thread(&Channel::ThreadReadH264File, this, start, size);
}

void Channel::Stop()
{
    b_stop = true;
    if (m_thread.joinable())
        m_thread.join();
}

typedef struct _Header
{
    char shead[10] = {0};
    char sChanID[20] = {0};
    char sRecorder[19] = {0};
    uint8_t uFramrate = 30;
    uint16_t uVersion = 0;
    uint8_t uVideoChan = 1;
    uint8_t uAudioChan = 0;
    uint32_t uRecTime = 0; // 录像时间
    uint32_t uSeconds = 0; // 视频长度
    uint16_t uPosSize = 0; // 容量大小
} Header;

typedef struct _FrameHeader
{
    uint8_t uChan = 0;     // 音/视频通道信息
    uint16_t uTemplet = 0; // 显示模板
    uint16_t uDisplay = 0; // 显示位置
    uint16_t uMsg = 0;     // 消息命令
    uint8_t uFlag = 0;
} FrameHeader;

void Channel::FastForward()
{
    b_ff = true;
}
void Channel::ThreadReadAqmsFile(int start, int size)
{

    int nBuffLen = 800000;
    char *pBuff = new char[nBuffLen];
    int nLen;
    Header header;
    FrameHeader fheader;
    fread(&header, sizeof(Header), 1, m_pFile);
    fseeko64(m_pFile, sizeof(Header) + header.uPosSize * sizeof(uint64_t), SEEK_SET);
    if (start != 0) {
        for (int i = 0; i < start; i++) {
            if (feof(m_pFile)) break;
            fseek(m_pFile, sizeof(FrameHeader), SEEK_CUR);
            fread(&nLen, sizeof(int), 1, m_pFile);
            fread(pBuff, nLen, 1, m_pFile);
            if ((AQCodecType)ntohs(AQ_HEADER_GET_CODEC_ID(pBuff)) != AQCodec_H264)
            {
                i--;
            }
        }
    }
    int frame_id = 0;
    while (!b_stop && !feof(m_pFile))
    {
        if (m_pDecoder->IsFull()) {
            AQT::AQSleep(1);
            continue;
        }
        
        fread(&fheader, sizeof(FrameHeader), 1, m_pFile);
        fread(&nLen, sizeof(int), 1, m_pFile);
        // INFO("OnVideoDecodedData, ChnID: %s, len: %d\n", m_strChnID.c_str(), nLen);
        if (nLen > nBuffLen)
        {
            delete[] pBuff;
            nBuffLen = nLen * 2;
            pBuff = new char[nBuffLen];
        }
        fread(pBuff, nLen, 1, m_pFile);
        if ((AQCodecType)ntohs(AQ_HEADER_GET_CODEC_ID(pBuff)) != AQCodec_H264)
        {
            continue;
        }
        m_pDecoder->Decode(pBuff, nLen);
        frame_id++;
        if (frame_id == size) {
            break;
        }
    }
    delete[] pBuff;
}

void Channel::ThreadReadH264File(int start, int size)
{

    int nBuffLen = 800000;
    char *pBuff = new char[nBuffLen];
    int nLen;
    if (start != 0) {
        for (int i = 0; i < start; i++) {
            if (feof(m_pFile)) break;
            fread(&nLen, sizeof(int), 1, m_pFile);
            fread(pBuff, nLen - sizeof(int), 1, m_pFile);
            if ((AQCodecType)ntohs(AQ_HEADER_GET_CODEC_ID(pBuff)) != AQCodec_H264)
            {
                i--;
            }
        }
    }

    int frame_id = 0;
    while (!b_stop && !feof(m_pFile)) {
        if (m_pDecoder->IsFull()) {
            AQT::AQSleep(1);
            continue;
        }
        fread(&nLen, sizeof(int), 1, m_pFile);
        nLen -= sizeof(int);
        if (nLen > nBuffLen)
        {
            delete[] pBuff;
            nBuffLen = nLen * 2;
            pBuff = new char[nBuffLen];
        }
        fread(pBuff, nLen, 1, m_pFile);
        m_pDecoder->Decode(pBuff, nLen);
        frame_id++;
        if (frame_id == size) {
            break;
        }
    }
    
    delete[] pBuff;
}