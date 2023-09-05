#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <chrono>
#include "AQHeader.h"
#include <sys/time.h>
#include <arpa/inet.h>
#include <fstream>
#ifndef AQ_EMBEDED
#include "common/json.hpp"
#include "common/log.hpp"
using namespace nlohmann;
#endif

#ifdef _WIN32
#include <tchar.h>
#include <WTypes.h>
#endif

namespace AQT
{
    static uint64_t AQGetTimestamp(bool bPerformance=false)
    {
#ifdef _WIN32
        if (!bPerformance)
        {
            using namespace chrono;
            time_point<system_clock, milliseconds> tp = time_point_cast<milliseconds>(system_clock::now());
            return tp.time_since_epoch().count();
        }
        LARGE_INTEGER freq, priv;
        if (QueryPerformanceFrequency(&freq) && QueryPerformanceCounter(&priv))
        {
            return (1000 * priv.QuadPart / freq.QuadPart);
        }
        return timeGetTime();
#else
        struct timeval now;
        gettimeofday(&now, NULL);
        return now.tv_sec * 1000 + now.tv_usec / 1000;
#endif
    }

    static uint64_t AQGetClientID()
    {
#ifdef _WIN32
        return ::timeGetTime();
#else
        struct timeval now;
        gettimeofday(&now, NULL);
        return now.tv_sec * 1000 + now.tv_usec;
#endif
    }

    static void AQSleep(uint64_t ulMS)
    {
#ifdef WIN32
        Sleep((DWORD)ulMS);
#else
        usleep(ulMS * 1000);
#endif
    }

    static uint64_t htonl64(uint64_t host)
    {
        uint64_t ret = 0;
        uint32_t high, low;
        low = host & 0xFFFFFFFF;
        high = (host >> 32) & 0xFFFFFFFF;
        low = htonl(low);
        high = htonl(high);
        ret = low;
        ret <<= 32;
        ret |= high;
        return ret;
    }

    static uint64_t ntohl64(uint64_t host)
    {
        uint64_t ret = 0;
        uint32_t high, low;
        low = host & 0xFFFFFFFF;
        high = (host >> 32) & 0xFFFFFFFF;
        low = ntohl(low);
        high = ntohl(high);
        ret = low;
        ret <<= 32;
        ret |= high;
        return ret;
    }

    static void GetResolution(Stream_Type streamType, int &nWidht, int &nHeight)
    {
        nWidht = 3840;
        nHeight = 2160;
        if (streamType == Stream_Sub)
        {
            nWidht = 1920;
            nHeight = 1080;
        }
    }

    static void GetResolutionRatio(CameraType eCamType, float& fRatioW, float& fRatioH)
    {
        switch (eCamType)
        {
        case Mantis_18:
            fRatioW = 1;
            fRatioH = 0.5;
            break;
        
        case Mantis_18_169:
            fRatioW = 1;
            fRatioH = 1;
            break;  

        case Mantis_18_329x2:
            fRatioW = 1;
            fRatioH = 0.25;
            break;   
                 
        default:
            fRatioW = 1;
            fRatioH = 1;
            break;
        }
    }

#ifndef AQ_EMBEDED
    static json ParseFromFile(string strFile)
    {
        json jRoot;
        std::fstream fIn(strFile, ios::in);
        if (fIn)
        {
            try
            {
                fIn >> jRoot;
            }
            catch (json::exception &e)
            {
            }
            fIn.close();
        }
        return jRoot;
    }

    static float JsonGetFloat(json j, string strKey, float def=0.0)
    {
        float fValue = def;

        json jval = j[strKey];
        if (jval.is_null())
        {
            return fValue;
        }

        try
        {
            jval.get_to(fValue);
        }
        catch(json::exception &e)
        {
            e.what();
        }
        return fValue;
    }

    static int JsonGetInt(json j, string strKey, int def=0)
    {
        int nValue = def;

        json jval = j[strKey];
        if (jval.is_null())
        {
            return nValue;
        }
        try
        {
            jval.get_to(nValue);
        }
        catch (json::exception &e)
        {
            e.what();
        }
        return nValue;
    }

    static int16_t JsonGetInt16(json j, string strKey, int def=0)
    {
        int16_t nValue = def;

        json jval = j[strKey];
        if (jval.is_null())
        {
            return nValue;
        }
        try
        {
            jval.get_to(nValue);
        }
        catch (json::exception &e)
        {
            e.what();
        }
        return nValue;
    }

    static uint64_t JsonGetUint64(json j, string strKey, uint64_t def=0)
    {
        int64_t nValue = def;

        json jval = j[strKey];
        if (jval.is_null())
        {
            return nValue;
        }
        try
        {
            jval.get_to(nValue);
        }
        catch (json::exception &e)
        {
            e.what();            
        }
        return nValue;
    }

    static string JsonGetString(json j, string strKey, string def="")
    {
        string strValue = def;

        json jval = j[strKey];
        if (jval.is_null())
        {
            return strValue;
        }
        try
        {
            jval.get_to(strValue);
        }
        catch (json::exception &e)
        {
            e.what();
        }
        return strValue;
    }
    
    static bool JsonGetBool(json j, string strKey, bool def=false)
    {
        bool bRet = def;

        json jval = j[strKey];
        if (jval.is_null())
        {
            return bRet;
        }
        try
        {
            jval.get_to(bRet);
        }
        catch (json::exception &e)
        {
            e.what();
        }
        return bRet;
    }
#endif


#ifdef _WIN32
    static wstring STR_A2W(const string& strA)
    {
        wstring strW;
        int nLength = MultiByteToWideChar(::GetACP(), 0, strA.c_str(), -1, NULL, 0);
        wchar_t* pTemp = new wchar_t[nLength];
        if (pTemp)
        {
            MultiByteToWideChar(::GetACP(), 0, strA.c_str(), -1, pTemp, nLength);
            strW = pTemp;
            delete[] pTemp;
        }
        return strW;
    }

    static string STR_W2A(const wstring& strW)
    {
        string strA;
        int nLength = WideCharToMultiByte(::GetACP(), 0, strW.c_str(), -1, NULL, 0, NULL, NULL);
        char* pTemp = new char[nLength];
        if (pTemp)
        {
            WideCharToMultiByte(::GetACP(), 0, strW.c_str(), -1, pTemp, nLength, NULL, NULL);
            strA = pTemp;
            delete[] pTemp;
        }
        return strA;
    }
#endif
}

#endif