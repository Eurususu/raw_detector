#ifndef __AQVIDEOHEADER_HPP__
#define __AQVIDEOHEADER_HPP__

typedef struct tagAQVideoHeader
{
    uint16_t    codecid;
    uint16_t    keyframe;
    uint16_t    actual_width;
    uint16_t    actual_height;
    uint16_t    virtual_width;
    uint16_t    virtual_heigth;
    uint16_t    type;
    uint16_t    seq;
    uint64_t    ts;
}AQ_VIDEO_HEADER, *PAQ_VIDEO_HEADER;

static uint16_t AQ_HEADER_LENGTH()
{
    return sizeof(AQ_VIDEO_HEADER);
}

static void AQ_HEADER_RESET(void *p)
{
    PAQ_VIDEO_HEADER pHeader=(PAQ_VIDEO_HEADER)p;
	pHeader->codecid=0;
	pHeader->keyframe=0;
    pHeader->actual_width=0;
    pHeader->actual_height=0;
    pHeader->virtual_width=0;
    pHeader->virtual_heigth=0;
    pHeader->type=0;
	pHeader->seq=0;
	pHeader->ts=0;
}

static uint16_t AQ_HEADER_GET_CODEC_ID(void *p)
{
    PAQ_VIDEO_HEADER pHeader=(PAQ_VIDEO_HEADER)p;
    return pHeader->codecid;
}

static void AQ_HEADER_SET_CODEC_ID(void *p, uint16_t v)
{
    PAQ_VIDEO_HEADER pHeader=(PAQ_VIDEO_HEADER)p;
    pHeader->codecid = v;
}

static uint16_t AQ_HEADER_GET_KEYFRAME(void *p)
{
    PAQ_VIDEO_HEADER pHeader=(PAQ_VIDEO_HEADER)p;
    return pHeader->keyframe;
}

static void AQ_HEADER_SET_KEYFRAME(void *p, uint16_t v)
{
    PAQ_VIDEO_HEADER pHeader=(PAQ_VIDEO_HEADER)p;
    pHeader->keyframe = v;
}

static uint16_t AQ_HEADER_GET_SEQUENCE(void *p)
{
    PAQ_VIDEO_HEADER pHeader=(PAQ_VIDEO_HEADER)p;
    return pHeader->seq;
}

static void AQ_HEADER_SET_SEQUENCE(void *p, uint16_t v)
{
    PAQ_VIDEO_HEADER pHeader=(PAQ_VIDEO_HEADER)p;
    pHeader->seq = v;
}

static uint64_t AQ_HEADER_GET_TIMESTAMP(void *p)
{
    PAQ_VIDEO_HEADER pHeader=(PAQ_VIDEO_HEADER)p;
    return pHeader->ts;
}

static void AQ_HEADER_SET_TIMESTAMP(void *p, uint64_t v)
{
    PAQ_VIDEO_HEADER pHeader=(PAQ_VIDEO_HEADER)p;
    pHeader->ts = v;
}

static uint16_t AQ_HEADER_GET_ACTUAL_WIDTH(void *p)
{
    PAQ_VIDEO_HEADER pHeader=(PAQ_VIDEO_HEADER)p;
    return pHeader->actual_width;
}

static void AQ_HEADER_SET_ACTUAL_WIDTH(void *p, uint16_t v)
{
    PAQ_VIDEO_HEADER pHeader=(PAQ_VIDEO_HEADER)p;
    pHeader->actual_width = v;
}

static uint16_t AQ_HEADER_GET_ACTUAL_HEIGHT(void *p)
{
    PAQ_VIDEO_HEADER pHeader=(PAQ_VIDEO_HEADER)p;
    return pHeader->actual_height;
}

static void AQ_HEADER_SET_ACTUAL_HEIGHT(void *p, uint16_t v)
{
    PAQ_VIDEO_HEADER pHeader=(PAQ_VIDEO_HEADER)p;
    pHeader->actual_height = v;
}

static uint16_t AQ_HEADER_GET_VIRTUAL_WIDTH(void *p)
{
    PAQ_VIDEO_HEADER pHeader=(PAQ_VIDEO_HEADER)p;
    return pHeader->virtual_width;
}

static void AQ_HEADER_SET_VIRTUAL_WIDTH(void *p, uint16_t v)
{
    PAQ_VIDEO_HEADER pHeader=(PAQ_VIDEO_HEADER)p;
    pHeader->virtual_width = v;
}

static uint16_t AQ_HEADER_GET_VIRTUAL_HEIGHT(void *p)
{
    PAQ_VIDEO_HEADER pHeader=(PAQ_VIDEO_HEADER)p;
    return pHeader->virtual_heigth;
}

static void AQ_HEADER_SET_VIRTUAL_HEIGHT(void *p, uint16_t v)
{
    PAQ_VIDEO_HEADER pHeader=(PAQ_VIDEO_HEADER)p;
    pHeader->virtual_heigth = v;
}

static uint16_t AQ_HEADER_GET_TYPE(void *p)
{
    PAQ_VIDEO_HEADER pHeader=(PAQ_VIDEO_HEADER)p;
    return pHeader->type;
}

static void AQ_HEADER_SET_TYPE(void* p, uint16_t v)
{
    PAQ_VIDEO_HEADER pHeader=(PAQ_VIDEO_HEADER)p;
    pHeader->type = v;
}

#endif