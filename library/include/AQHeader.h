#ifndef __AQHEADER_H__
#define __AQHEADER_H__

#ifndef AQ_EMBEDED
#include "common/log.hpp"
#include "common/json.hpp"
#else
#include <thread>
#endif

const int g_nPanoID = 999;
const bool g_bUsePano = true;


typedef struct VEDIO_PACKET
{
    VEDIO_PACKET()
    {
        nWidth = 0;
        nHeight = 0;
        nLen = 0;
        ulEncodeTimestamp = 0;
        ulRecvTimestamp = 0;
        pData = NULL;
    }
    ~VEDIO_PACKET()
    {
        if (pData != NULL)
        {
            delete pData;
            pData = NULL;
        }
    }
    int32_t  nWidth;
    int32_t  nHeight;
    int32_t  nLen;
    uint64_t ulEncodeTimestamp;
    uint64_t ulRecvTimestamp;
    char* pData;
}VideoPacket, *PVideoPacket;

typedef struct tag_ROI
{
	int nX;
	int nY;
	int nWidth;
	int nHeight;
	tag_ROI()
	{
		nX = -1;
		nY = -1;
		nWidth = -1;
		nHeight = -1;
	}
    tag_ROI(int x, int y, int w, int h)
    {
        nX = x;
        nY = y;
        nWidth = w;
        nHeight = h;
    }
}ROI, *PROI;


typedef struct tag_AQCAMERA
{
    tag_AQCAMERA()
    {
        memset(szName, 0, 255);
        memset(szType, 0, 255);
    }
	int nID;
	char szName[255];
	char szType[255];
    int nStatus;
}AQ_Camera, *PAQ_Camera;

typedef struct tag_AQSENSOR
{
    tag_AQSENSOR()
    {
        memset(szName, 0, 255);
    }
    int nID;
    char szName[255];
}AQ_Sensor, *PAQ_Sensor;

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
    AQCodec_Any = 99
};

enum AQChannelType
{
    AQChannel_Stitch = 0,
    AQChannel_Sensor,
    AQChannel_Pano
};

enum AQStitchMode
{
    AQStitchMode_Real = 0,
    AQStitchMode_Rec,
    AQStitchMode_Current = 99
};

enum ClientType
{
    ClientType_Client = 0,
    ClientType_AQSS,
    ClientType_VOD,
    ClientType_AQDS,
    ClientType_TMS,
    ClientType_MTS,
    ClientType_MGS,
    ClientType_DRS,
};

enum CameraType
{
    UnknownCameraType = -1,
    Mantis_18 = 0,
    PathFinder_18,
    BumbleBee_40,
    PathFinder_11,
    Mantis_18_169,
    PathFinder_4,
    Mantis_18_329x2,
    PathFinder_2_50s,
    PathFinder_2_25s,
    QIUJI_DAHUA,
    QIUJI_HAIKANG,
    CAMERA_ONVIF,

    CameraTypeCount
};

enum StitcherType
{
    Single_GPU = 0,
    Double_GPU
};

#endif