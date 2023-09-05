/**********************************************************
 * Author        : Peng Xuejian
 * Email         : pengxj@ksitri.com
 * Last modified : 2020-08-28 10:51
 * Filename      : CPUQueue.h
 * Description   : 
 * *******************************************************/
#ifndef __CPU_QUEUE_H__
#define __CPU_QUEUE_H__

#include "Queue.h"

using namespace std;
template<class T>
class CCPUQueue : public CQueue<T> {
public:
	CCPUQueue(uint32_t iDefaultElemNum, uint32_t iQueueMaxNum, bool bDropMode, bool bRepeatMode)//iQueueMaxNum must be bigger than 1
	:
	CQueue<T>(iDefaultElemNum, iQueueMaxNum, bDropMode, bRepeatMode)
	{
		CQueue<T>::m_pBuffer = (T**)calloc(iQueueMaxNum, sizeof(T*));
		for(uint32_t i = 0; i < iQueueMaxNum; i++){
			CQueue<T>::m_pBuffer[i] = new T[iDefaultElemNum];
		}

	}
	~CCPUQueue()
	{
		for(uint32_t i = 0; i < CQueue<T>::m_iQueueMaxNum; i++){
			delete[] CQueue<T>::m_pBuffer[i];
		}
		free(CQueue<T>::m_pBuffer);
	}
private:
	virtual void ReAllocElem(uint32_t iBufIndex, uint32_t iNewSize)
	{
		delete[] CQueue<T>::m_pBuffer[iBufIndex];
		CQueue<T>::m_pBuffer[iBufIndex] = new T[iNewSize];
		CQueue<T>::m_vElemSize[iBufIndex] = iNewSize;
	}
};

#endif
