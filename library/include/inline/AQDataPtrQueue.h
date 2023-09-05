#pragma once

#include <condition_variable>
#include <mutex>
#include <list>
#include <atomic>

using namespace std;

template<class dtype>
class AQDataPtrQueue
{
public:
	virtual ~AQDataPtrQueue() {
		DestroyQueue();
	}
	virtual void Push(shared_ptr<dtype> data) {
		if (m_bEnablePush && data && !m_bEnd) {
			{
				lock_guard<mutex> l(m_mtxData);
				m_list.push_back(data);
				if (m_nMaxSize>0 && m_list.size() > m_nMaxSize) {
					m_list.pop_front();
				}
			}
			m_condData.notify_one();
		}
	}

	virtual void EnablePush(bool bEnable) {
		m_bEnablePush = bEnable;
	}

	virtual shared_ptr<dtype> Pop() {
		shared_ptr<dtype> ptr; 
		{
			lock_guard<mutex> l(m_mtxData);
			if (!m_list.empty()) {
				ptr = m_list.front();
				m_list.pop_front();
			}
		}
		return ptr;
	}

	virtual void UnPop(shared_ptr<dtype> ptr) {
		if (m_bEnablePush && ptr && !m_bEnd) {
			lock_guard<mutex> l(m_mtxData);
			m_list.push_front(ptr);
		}
	}

	virtual void Clear() {
		lock_guard<mutex> l(m_mtxData);
		m_list.clear();
	}

	virtual shared_ptr<dtype> Previous() {
		shared_ptr<dtype> ptr;
		{
			lock_guard<mutex> l(m_mtxData);
			ptr = m_ptrPreData;
		}
		return ptr;
	}

	virtual shared_ptr<dtype> PopWait(int nMsec = -1) {
		shared_ptr<dtype> ptr = Pop();
		if (ptr)
		{
			return ptr;
		}
		if (!m_bEnd)
		{
			unique_lock<mutex> lock(m_mtxData);
			if (nMsec > 0) {
				m_condData.wait_for(lock, chrono::milliseconds(nMsec));
			}
			else {
				m_condData.wait(lock);
			}

			if (!m_list.empty()) {
				ptr = m_list.front();
				m_list.pop_front();
			}

			lock.unlock();
		}
		return ptr;
	}

	virtual void Wakeup() {
		m_condData.notify_one();
	}

	virtual void EndWait() {
		m_bEnd = true;
		m_condData.notify_all();
	}

	virtual void SetMaxSize(uint32_t nSize) {
		m_nMaxSize = nSize;
	}
	virtual size_t Size() {
		lock_guard<mutex> l(m_mtxData);
		return m_list.size();
	}
	virtual bool IsEmpty() {
		lock_guard<mutex> l(m_mtxData);
		return m_list.empty();
	}
	virtual void DestroyQueue() {
		Clear();
		EndWait();
	}
	virtual void StartQueue() {
		Clear();
		m_bEnd = false;
		m_bEnablePush = true;
	}

protected:
	list<shared_ptr<dtype>> m_list;
	mutex m_mtxData;
	condition_variable m_condData;
	atomic_bool m_bEnablePush{true};
	shared_ptr<dtype> m_ptrPreData;
	atomic_bool m_bEnd{false};
	atomic_uint32_t m_nMaxSize{0};
};
