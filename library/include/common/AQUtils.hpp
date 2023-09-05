#ifndef __AQUTILS_H__
#define __AQUTILS_H__

#include <string>
#include <iostream>
#include <sys/stat.h>
#include <stdarg.h>
#include <algorithm>
#include <vector>
#include <ctime>
#include <chrono>
#include <time.h>
#include <assert.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <dirent.h>
#include <unistd.h>
#endif // _WIN32


namespace AQutils
{
    static bool preUseDir(std::string dir_path)
    {
#ifdef _WIN32
        struct _stat st;
        if (_stat(dir_path.c_str(), &st) < 0)
        {
            return 0 == _mkdir(dir_path.c_str());
        }
#else
        if (access(dir_path.c_str(), R_OK) != 0)
        {
            int ret = mkdir(dir_path.c_str(), 0755);
            return ret == 0;
        }
#endif
        return true;
    }

    static std::string replace(std::string str, std::string old_val, std::string new_val)
    {
        std::vector<std::string> vec;
        for(const auto& c : str)
        {
            std::string tem = " ";
            tem[0] = c;
            vec.push_back(tem);
        }

        std::string ret;
        for(auto& s : vec)
        {
            if(s == old_val)
            {
                ret += new_val;
            }
            else
            {
                ret += s;
            }
        }
        return ret;
    }

    static std::vector<std::string> split(const std::string& str, const char delimiter)
    {
        std::vector<std::string> vec;
        std::string s;
        for(const auto c : str)
        {
            if(c == delimiter)
            {
                vec.push_back(s);
                s.clear();
            }
            else
            {
                s.push_back(c);
            }
        }
        return vec;
    }

    static uint64_t toTimeStamp(std::time_t time)
    {
		std::chrono::system_clock::time_point tp = std::chrono::system_clock::from_time_t(time);
        return (uint64_t)(std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()).count());
    }

    static uint64_t currentTime()
    {
		auto time_now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        return time_now.count();
    }

    static std::tm* toTm(uint64_t timestamp)
    {
        const std::chrono::system_clock::duration duration = std::chrono::milliseconds(timestamp);
		const std::chrono::time_point<std::chrono::system_clock> tp(duration);
		time_t time = std::chrono::system_clock::to_time_t(tp);
		struct tm* local_time = localtime(&time);
        return local_time;
    }

    static uint32_t to_uint(uint64_t timestamp_diff)
    {
        return timestamp_diff & 0xffffffff;
    }

    static void sleep(uint64_t secs)
    {
#ifdef _WIN32
        Sleep(secs * 1000);
#else
        usleep(secs * 1000000);
#endif
    }

    static string TimestampToString(uint64_t ulTimestamp)
    {
        std::tm* pTm = toTm(ulTimestamp);
        
        char aText[32] = {0};
        sprintf(aText, "%04d-%02d-%02d %02d:%02d:%02d", pTm->tm_year + 1900, pTm->tm_mon + 1, pTm->tm_mday, pTm->tm_hour, pTm->tm_min, pTm->tm_sec);
        return aText;
    }

    static string TimestampToString2(uint64_t ulTimestamp)
    {
        std::tm* pTm = toTm(ulTimestamp);
        
        char aText[32] = {0};
        sprintf(aText, "%04d%02d%02d%02d%02d%02d", pTm->tm_year + 1900, pTm->tm_mon + 1, pTm->tm_mday, pTm->tm_hour, pTm->tm_min, pTm->tm_sec);
        return aText;
    }
} 

#endif