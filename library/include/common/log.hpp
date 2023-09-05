#ifndef LOG_H
#define LOG_H

#include <thread>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <functional>
#include <atomic>
#include <list>
#include <string>
#include <memory>
#include <map>
#include "utils.hpp"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

typedef bool (*LoggerListener)(const char *file, int line, int level, const char *message);



#define LWITEFILE   (0x01)
#define LPRINTTIME  (0x02)
#define LPRINTLINE  (0x04)

#define LTRASH      (0)                                     // 用于大量的垃圾打印,不打印时间,不写文件
#define LDEBUG      (0x10 |LWITEFILE|LPRINTTIME|LPRINTLINE)            // 调试信息
#define LINFOS      (0x20 |LWITEFILE)                       // info with no time
#define LINFO       (0x30 |LWITEFILE|LPRINTTIME|LPRINTLINE)            // info
#define LWARNING    (0x40 |LWITEFILE|LPRINTTIME|LPRINTLINE) // 告警信息
#define LERROR      (0x50 |LWITEFILE|LPRINTTIME|LPRINTLINE) // 错误信息
#define LFATAL      (0x60 |LWITEFILE|LPRINTTIME|LPRINTLINE) // 严重错误
#define LUNIQUE     (0x70 |LWITEFILE|LPRINTTIME)            // 去重信息
#define LDEFULT     LDEBUG


#define TRACED(...) if (anktech::Logger::inst().level<=LTRASH)   anktech::__log_func(NULL, __FILE__, __LINE__, LTRASH, __VA_ARGS__)
#define DBGING(...) if (anktech::Logger::inst().level<=LDEBUG)   anktech::__log_func(NULL, __FILE__, __LINE__, LDEBUG, __VA_ARGS__)

#define INFOS(...)  if (anktech::Logger::inst().level<=LINFOS)   anktech::__log_func(NULL, __FILE__, __LINE__, LINFOS, __VA_ARGS__)
#define INFO(...)   if (anktech::Logger::inst().level<=LINFO)    anktech::__log_func(NULL, __FILE__, __LINE__, LINFO, __VA_ARGS__)
#define INFOW(...)  if (anktech::Logger::inst().level<=LWARNING) anktech::__log_func(NULL, __FILE__, __LINE__, LWARNING, __VA_ARGS__)
#define INFOE(...)  if (anktech::Logger::inst().level<=LERROR)   anktech::__log_func(NULL, __FILE__, __LINE__, LERROR, __VA_ARGS__)
#define INFOF(...)  if (anktech::Logger::inst().level<=LFATAL)   anktech::__log_func(NULL, __FILE__, __LINE__, LFATAL, __VA_ARGS__)
#define INFOU(key, ...) if (anktech::Logger::inst().level<=LUNIQUE) anktech::__log_func(key, __FILE__, __LINE__, LUNIQUE, __VA_ARGS__)

namespace anktech
{
    class Logger
    {
    public:
        std::mutex logger_lock_;
        std::string logger_directory;
        LoggerListener logger_listener = nullptr;
        volatile bool has_logger = true;
        std::list<std::string> cache_;
        std::thread flush_thread_;
        int level = LDEFULT;
        bool async_write = false;

        void write(const std::string &line)
        {
            if (line.empty() || logger_directory.empty())
            {
                return;
            }
            if (!async_write)
            {
                flush(line, "");
                return;
            }
            {
                lock_guard<mutex> l(logger_lock_);
                cache_.emplace_back(line);
            }
        }
        std::shared_ptr<FILE> open(const string& tag)        
        {
            std::shared_ptr<FILE> hfile;
            auto now = anktech::DateNow();
            string file = anktech::Format("%s%s%s.txt", logger_directory.c_str(), tag.c_str(), now.c_str());
            hfile.reset(anktech::fopen_mkdirs(file, "a+"), [](FILE* p){ if(p) fclose(p);});
            return hfile;
        }

        void flush(const string& line, const string& tag)
        {
            std::shared_ptr<FILE> handler = open(tag);
            if (handler)
            {
                fwrite(line.c_str(), 1, line.length(), handler.get());
            }
        }

        bool flush()
        {
            std::list<std::string> local_;
            {
                std::lock_guard<std::mutex> l(logger_lock_);
                std::swap(local_, cache_);
            }

            if (!local_.empty())
            {
                std::shared_ptr<FILE> handler = open("");
                if (handler)
                {
                    for (auto &line : local_)
                    {
						fwrite(line.c_str(), 1, line.length(), handler.get());
                    }
                }
                return true;
            }
            return false;
        }

        void startThread()
        {
			async_write = true;
            flush_thread_ = thread(std::bind(&Logger::flush_job, this));    
        }
        void stopThread()
        {
			async_write = false;
            if (flush_thread_.joinable())
            {
                flush_thread_.join();
            }
        }

        void flush_job()
        {
            while (async_write)
            {
                if (!flush())
                {
                    anktech::Sleep(500);
                }
            }
            flush();
        }

        void setLoggerSaveDirectory(const std::string &loggerDirectory)
        {
            logger_directory = loggerDirectory;
            if (logger_directory.empty())
                logger_directory = ".";
            if (logger_directory.back() != '/')
            {
                logger_directory.push_back('/');
            }
            mkdirs(loggerDirectory);
        }

        void close()
        {
            stopThread();
        }

        virtual ~Logger()
        {
            close();
        }

        static Logger& inst()
        {
            static Logger l;
            return l;
        }
    };

    static map<int,string>& level_map()
    {
        static map<int,string> m = {
            { LTRASH, "trush" },
            { LDEBUG, "debug" },
            { LINFOS, "infos" },
            { LINFO, "info" },
            { LWARNING, "warn" },
            { LERROR, "error" },
            { LFATAL, "fatal" },
            { LUNIQUE, "info..." }
        };
        return m;
    }
    static const char *GetStringLevel(int nLevel)
    {
        map<int,string>& ml = level_map();
        auto it = ml.find(nLevel);
        if (it != ml.end())
        {
            return it->second.c_str();
        }
        return "unknow";
    }

    static int get_level(const char* level)
    {
        if (!level || !level[0])
        {
            return LDEFULT;
        }      
        map<int,string>& ml = level_map();
        for (auto& l : ml)
        {
#ifdef _WIN32
            if (_stricmp(l.second.c_str(), level) == 0)
#else
            if (strcasecmp(l.second.c_str(), level) == 0)
#endif
            {
                return l.first;
            }
        }
        return LDEFULT;
    }

    static void print(const char* logo, int level)
    {
#ifdef _WIN32
        OutputDebugStringA(logo);
#else
        if (level == LFATAL || level == LERROR)
        {
            fprintf(stderr, "%s", logo);
        }
        else
        {
            fprintf(stdout, "%s", logo);
        }
#endif
    }

    static void shead(char* head, int size, const char* file, int line, int level, bool color)
    {
        int n = 0;
        if (level & LPRINTTIME)
        {
            n += sprintf(head, "[%s]", anktech::TimeNow().c_str());
        }

        if (color)
        {
            if (level == LFATAL || level == LERROR)
            {
                n += snprintf(head + n, size - n, "[\033[31m%s\033[0m]", GetStringLevel(level));
            }
            else if (level == LWARNING)
            {
                n += snprintf(head + n, size - n, "[\033[33m%s\033[0m]", GetStringLevel(level));
            }
            else
            {
                n += snprintf(head + n, size - n, "[\033[32m%s\033[0m]", GetStringLevel(level));
            }
        }
        else
        {
            n += snprintf(head + n, size - n, "%s", GetStringLevel(level));
        }

        if (level & LPRINTLINE)
        {
            std::string filename = anktech::GetFileName(file, true);
            n += snprintf(head + n, size - n, "[%s:%d]:", filename.c_str(), line);
        }
    }


    static shared_ptr<char> sfmt(const char* file, int line, int level, bool color, const char* logo, int logolen)
    {
        if (!logo || logolen <= 0)
        {
            return nullptr;
        }

        char head[256] = { 0 };
        shead(head, 256, file, line, level, color);
        int nSize = 256 + logolen;

        shared_ptr<char> ptr(new(std::nothrow) char[nSize], [](char* p) {delete[]p; });
        if (ptr)
        {
            memset(ptr.get(), 0, nSize);
            snprintf(ptr.get(), nSize, "%s%s\n", head, logo);
        }
        return ptr;
    }

    static void __log_func(const char* key, const char *file, int line, int level, const char *fmt, ...)
    {
        static string s_unique_key;
        if (!Logger::inst().has_logger || level < Logger::inst().level)
        {
            return;
        }

        if (key && *key != 0)
        {
            if (s_unique_key == key) return;
            s_unique_key = key;
        }
        else
        {
            s_unique_key.clear();
        }

        char buff[1024] = { 0 };
        int nSize = sizeof(buff);
        shared_ptr<char> pLogo(buff, [](char* p){});
        do {
            va_list ap;
            va_start(ap, fmt);
            int n = vsnprintf(pLogo.get(), nSize, fmt, ap);
            va_end(ap);
            if (n < nSize - 1)
            {
                nSize = n;
                break;
            }
            nSize = n / 4 * 4 + 8;
            pLogo.reset(new(std::nothrow) char[nSize], [](char* p) {delete[]p; });
        } while (pLogo);

        if (!pLogo)
        {
            return;
        }
        if (Logger::inst().logger_listener && Logger::inst().logger_listener(file, line, level, pLogo.get()))
        {
            return;
        }

        shared_ptr<char> pBuffer;
        shared_ptr<char> pColor;
        
#ifdef _WIN32
        if (!pBuffer) pBuffer = sfmt(file, line, level, false, pLogo.get(), nSize);
        if (pBuffer) print(pBuffer.get(), level);
#else
        if (!pColor) pColor = sfmt(file, line, level, true, pLogo.get(), nSize);
        if (pColor) print(pColor.get(), level);
#endif

        if (level & LWITEFILE)
        {
            if (!pBuffer) pBuffer = sfmt(file, line, level, false, pLogo.get(), nSize);
            if (pBuffer)
            {
                Logger::inst().write(pBuffer.get());
                if (level == LERROR)
                {
                    Logger::inst().flush(pBuffer.get(), "error_");
                }
            }
        }
        if (level == LFATAL)
        {
            Logger::inst().flush();
            fflush(stdout);
            abort();
        }
    }
    static void __log_path(const char* path, int level=LDEBUG)
    {
        Logger::inst().setLoggerSaveDirectory(path);
        Logger::inst().level = level;
    }
    static void __log_async_start() {
        Logger::inst().stopThread();
        Logger::inst().startThread();
    }

} // namespace anktech
#endif