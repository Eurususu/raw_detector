#ifndef PTR_QUEUE_HPP
#define PTR_QUEUE_HPP

#include <condition_variable>
#include <mutex>
#include <list>
#include <iostream>


class IPtrQueue {
public:
    virtual ~IPtrQueue(){};
    virtual void stopQueue() = 0;
    virtual void startQueue() = 0;
    virtual void setMaxSize(int n) = 0;
    virtual bool isPopable() = 0;
    virtual bool isPushable() = 0;
    virtual void togglePushable(bool enable) = 0;
    virtual void togglePopable(bool enable) = 0;
};


template<typename T>
class PtrQueue : public IPtrQueue {
public:
    virtual ~PtrQueue() {
        stopQueue();
    }
    void setName(const std::string &id) {
        qid = id;
    }
    void push(std::shared_ptr<T> data) {
        if (data) {
            if (!_enable_push) return ;
            {
                std::unique_lock<std::mutex> l(_mtx);
                _list.push_back(data);
                if (_max_size > 0 && _list.size() > _max_size && _need_warning) {
                    _list.pop_front();
                    _need_warning = false;
                } else if (_max_size > 0 && _list.size()  < _max_size / 2 && !_need_warning) {
                    _need_warning = true;
                }
                
            }
            _cv.notify_one();
        }
    }

    void push_wait(std::shared_ptr<T> data) {
        if (data) {
            {
                std::unique_lock<std::mutex> l(_mtx);
                if (!_enable_push) return ;
                _cv.wait(l, [&]{return  _list.size() != _max_size || !_enable_push;});
                if (!_enable_push) {
                    return;
                }
                _list.push_back(data);
            }
            _cv.notify_all();
        }
    }

    std::shared_ptr<T> pop() {
        std::shared_ptr<T> ptr;
        {
            std::unique_lock<std::mutex> l(_mtx);
            _cv.wait(l, [&]{ return !_enable_pop || !_enable_push || !_list.empty();});
            if (!_enable_push && _list.empty()) _enable_pop = false;
            if (!_enable_pop) return ptr;
            ptr = _list.front();
            _list.pop_front();
        }
        _cv.notify_all();
        return ptr;
    }

    // peek and pop does not support multi-thread pop
    std::shared_ptr<T> peek() {
        std::shared_ptr<T> ptr;
        {
            std::unique_lock<std::mutex> l(_mtx);
            _cv.wait(l, [&]{ return !_enable_pop || !_enable_push || !_list.empty();});
            if (!_enable_push && _list.empty()) _enable_pop = false;
            if (!_enable_pop) return ptr;
            ptr = _list.front();
        }
        return ptr;
    }

    void togglePushable(bool enable) {
        std::unique_lock<std::mutex> l(_mtx);
        _enable_push = enable;
        _cv.notify_all();
    }

    void togglePopable(bool enable) {
        std::unique_lock<std::mutex> l(_mtx);
        _enable_pop = enable;
        _cv.notify_all();
    }

    void setMaxSize(int n) {
        std::unique_lock<std::mutex> l(_mtx);
        _max_size = n;
    }

    void startQueue() {
        togglePushable(true);
        togglePopable(true);
    }

    void stopQueue() {
        togglePushable(false);
        togglePopable(false);
        _list.clear();

    }

    bool isPopable() {
        return _enable_pop;
    }
    bool isPushable() {
        return _enable_push;
    }
    bool isEmpty() {
        return _list.empty();
    }

    bool isFull() {
        return _max_size > 0 && _list.size() >= _max_size;
    }
    int getSize() {
        return _list.size();
    }

protected:
	std::list<std::shared_ptr<T>> _list;
    std::mutex _mtx;
    std::condition_variable _cv;
    bool _enable_push{false}, _enable_pop{false};
    int _max_size{0};
    bool _need_warning{true};
    std::string qid;
};

#endif  // PTR_QUEUE_HPP
