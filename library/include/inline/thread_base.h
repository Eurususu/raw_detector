#pragma once

#include <thread>
#include <atomic>

using namespace std;

class threadBase
{
public:
	virtual ~threadBase(){
		stopThread();
	}
	virtual void startThread() {
		m_bWantToStop = false;
		try {
			m_thread = thread([](threadBase* pThis) {
				pThis->ThreadProcMain();
				}, this);
		}
		catch (std::exception& excp) {
			printf("---> threadBase catch exception: %s\n", excp.what());
			m_bWantToStop = true;
		}
	}
	virtual void stopThread() {
		m_bWantToStop = true;
		if (this_thread::get_id() != m_thread.get_id() && m_thread.joinable()) {
			m_thread.join();
		}
	}
	
protected:
	virtual void ThreadProcMain() = 0;

	thread m_thread;
	atomic_bool m_bWantToStop{false};
};
