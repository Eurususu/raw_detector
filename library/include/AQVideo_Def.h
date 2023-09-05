#ifndef __AQVIDEO_DEF_HPP__
#define __AQVIDEO_DEF_HPP__

enum Stream_Type
{
    Stream_Main = 0,
    Stream_Sub,
    Stream_QSub    
};

enum AQCodecType
{
    AQCodec_H264 = 0,
    AQCodec_H265,
    AQCodec_Cur = 99
};

#endif