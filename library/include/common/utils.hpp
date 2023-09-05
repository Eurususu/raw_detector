#ifndef UTILS_H
#define UTILS_H


#include <chrono>
#include <thread>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdarg.h>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <dirent.h>
#include <unistd.h>
#endif

using namespace std;

#define __GetTimeBlock \
    time_t timep;      \
    time(&timep);      \
    tm &t = *(tm *)localtime(&timep);

namespace anktech
{
    //time
    static long long TimestampNow()
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    }
    static std::string TimeNow()
    {
        char time_string[20];
        __GetTimeBlock;

        sprintf(time_string, "%02d:%02d:%02d", t.tm_hour, t.tm_min, t.tm_sec);
        return time_string;
    }

    static std::string DateNow()
    {
        char time_string[20];
        __GetTimeBlock;
        sprintf(time_string, "%04d-%02d-%02d", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday);
        return time_string;
    }

    static void Sleep(int ms)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }

    static std::string Format(const char *fmt, ...)
    {
        va_list vl;
        va_start(vl, fmt);
        char buffer[10000];
        vsprintf(buffer, fmt, vl);
        return buffer;
    }

    static bool FileExists(const std::string &path)
    {
#ifdef _WIN32
        struct _stat st;
        return (_stat(path.c_str(), &st) >= 0);
#else
        return access(path.c_str(), R_OK) == 0;
#endif
    }

    static bool mkdir(const std::string &path)
    {

#ifdef _WIN32
        return (_mkdir(path.c_str()) == 0 || GetLastError() == ERROR_ALREADY_EXISTS);
#else
        return ::mkdir(path.c_str(), 0755) == 0;
#endif
    }

    static bool chdir(const std::string& path)
    {
#ifdef _WIN32
        return _chdir(path.c_str()) == 0;
#else
        return ::chdir(path.c_str()) == 0;
#endif
    }

    static bool mkdirs(const std::string &path)
    {
        if (path.empty())
            return false;
        if (FileExists(path))
            return true;

        std::string _path = path;
        char *dir_ptr = (char *)_path.c_str();
        char *iter_ptr = dir_ptr;

        bool keep_going = *iter_ptr != 0;
        while (keep_going)
        {
            if (*iter_ptr == 0)
                keep_going = false;

            if (((*iter_ptr == '/' || *iter_ptr == '\\') && iter_ptr != dir_ptr) || *iter_ptr == 0)
            {
                char old = *iter_ptr;
                *iter_ptr = 0;
                if (!FileExists(dir_ptr))
                {
                    if (!mkdir(dir_ptr))
                    {
                        if (!FileExists(dir_ptr))
                        {
                            return false;
                        }
                    }
                }
                *iter_ptr = old;
            }
            iter_ptr++;
        }
        return true;
    }

    static bool mkmdir(const std::string& strPath)
    {
        if (FileExists(strPath))
        {
            return true;
        }
        std::string strDir = strPath;

        int nLen = strDir.length();
        if (strDir[nLen - 1] == '\\' || strDir[nLen - 1] == '/')
        {
            strDir[nLen - 1] = '\0';
        }
        if (mkdir(strDir))
        {
            return true;
        }
        int nfind1 = strDir.rfind('\\');
        int nfind2 = strDir.rfind('/');
        int nfind = max(nfind1, nfind2);
        if (nfind == strDir.npos)
        {
            return false;
        }
        std::string strSubDir = strDir.substr(0, nfind);
        if (!mkmdir(strSubDir))
        {
            return false;
        }
        return mkdir(strDir);
    }

    static bool rmdir(const std::string& strDir)
    {

#ifdef WIN32
        DWORD dwStatus = 0;

        WIN32_FIND_DATAA fData;
        std::string strPath = strDir + "/*.*";
        HANDLE hFind = FindFirstFileA(strPath.c_str(), &fData);
        if (hFind != INVALID_HANDLE_VALUE)
        {
            do
            {
                if (strcmp(fData.cFileName, ".") == 0 || strcmp(fData.cFileName, "..") == 0)
                {
                    continue;
                }
                char szFile[512] = { 0 };
                sprintf(szFile, "%s/%s", strDir.c_str(), fData.cFileName);
                if (fData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
                {
                    rmdir(szFile);
                }
                else
                {
                    DeleteFileA(szFile);
                }

            } while (FindNextFileA(hFind, &fData));
            
            ::FindClose(hFind);
        }
        _rmdir(strDir.c_str());

        return true;
#else

        std::string strExecute = "rm -rf " + strDir;
        system(strExecute.c_str());
        return true;
#endif
    }


    static bool IsEmptyDir(const std::string& strDir)
    {
#ifdef WIN32
        DWORD dwStatus = 0;

        WIN32_FIND_DATAA fData;
        std::string strPath = strDir + "/*.*";
        HANDLE hFind = FindFirstFileA(strPath.c_str(), &fData);
        if (hFind != INVALID_HANDLE_VALUE)
        {
            do
            {
                if (strcmp(fData.cFileName, ".") != 0 && strcmp(fData.cFileName, "..") != 0)
                {
                    return false;
                }

            } while (FindNextFileA(hFind, &fData));
            ::FindClose(hFind);
        }
        return true;
#else
        DIR* d_info = opendir(strDir.c_str());
        return (!d_info || readdir(d_info) ==nullptr);
#endif
    }

    static FILE *fopen_mkdirs(const std::string &path, const std::string &mode)
    {
        FILE *f = fopen(path.c_str(), mode.c_str());
        if (f)
            return f;

        int p = path.rfind('/');
        if (p == -1)
            return nullptr;

        std::string directory = path.substr(0, p);
        if (!mkdirs(directory))
            return nullptr;
        return fopen(path.c_str(), mode.c_str());
    }

    static std::string GetFileName(const std::string &path, bool bIncludeSuffix)
    {
        if (path.empty())
            return "";
        int p = path.rfind('/');
        p += 1;

        //include suffix
        if (bIncludeSuffix)
            return path.substr(p);

        int u = path.rfind('.');
        if (u == -1)
            return path.substr(p);

        if (u <= p)
            u = path.size();
        return path.substr(p, u - p);
    }

    static void splitStr(const std::string &inputStr, const std::string &key, std::vector<std::string> &outStrVec)
    {
        if (inputStr == "")
        {
            return;
        }
        size_t pos = inputStr.find(key);
        size_t oldpos = 0;
        if (pos != inputStr.npos)
        {
            std::string tmp = inputStr.substr(0, pos);
            outStrVec.push_back(tmp);
        }
        while (1)
        {
            if (pos == inputStr.npos)
            {
                break;
            }
            oldpos = pos;
            size_t newpos = inputStr.find(key, pos + key.length());
            std::string tmp = inputStr.substr(pos + key.length(), newpos - pos - key.length());
            outStrVec.push_back(tmp);
            pos = newpos;
        }
        size_t tmplen = 0;
        if (outStrVec.size() > 0)
        {
            tmplen = outStrVec.at(outStrVec.size() - 1).length();
        }
        if (oldpos + tmplen < inputStr.length() - 1)
        {
            std::string tmp = inputStr.substr(oldpos + key.length());
            outStrVec.push_back(tmp);
        }
    }

    static int toInt(const std::string &in)
    {
        int re = 0;
        sscanf(in.c_str(), "%d", &re);
        return re;
    }

    static float toFloat(const std::string &in)
    {
        float re = 0;
        sscanf(in.c_str(), "%f", &re);
        return re;
    }
} // namespace anktech
#endif