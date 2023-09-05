/**********************************************************
 * Author        : Peng Xuejian
 * Email         : pengxj@ksitri.com
 * Last modified : 2019-11-28 12:10
 * Filename      : CUDACtxMan.h
 * Description   : 
 * *******************************************************/
#ifndef __CUDA_CTX_MAN__
#define __CUDA_CTX_MAN__

#include <cuda.h>
#include <vector>
#include <stdio.h>
#include <assert.h>
#include "common/log.hpp"
using namespace std;

class CCUDACtxMan
{
public:
	static bool Init()
	{
		if(g_cCtxs.empty()){
			//Init CUDA
			if(cuInit(0) != CUDA_SUCCESS){
				INFO("Fail to Init cuda\n"); 
				return false;
			}
			int iGpuCount = 0;
			if(cuDeviceGetCount(&iGpuCount) != CUDA_SUCCESS){
				INFO("Fail to Get Device count\n"); 
				return false;
			}
			g_cCtxs.resize(iGpuCount, 0);
			for(int iGpuID = 0; iGpuID < iGpuCount; iGpuID++){
				CUdevice cDevice = 0;
				if(cuDeviceGet(&cDevice, iGpuID) != CUDA_SUCCESS){
					INFO("Fail to Get Device %d\n", iGpuID); 
					return false;
				}
				if(cuCtxCreate(&g_cCtxs[iGpuID], 0, cDevice) != CUDA_SUCCESS){
					INFO("Fail to create Context\n"); 
					return false;
				}
			}
		}
		return true;
	}
	static void Uninit()
	{
		for(int i = 0; i < (int)g_cCtxs.size(); i++){
			if(g_cCtxs[i] != 0 && cuCtxDestroy(g_cCtxs[i]) != CUDA_SUCCESS){
				INFO("Fail to destroy cuda context of GPU: %d\n", i);
				continue;
			}
		}
		g_cCtxs.clear();
	}
	static CUcontext GetCtx(int iGpuID)
	{
		assert(iGpuID < (int)g_cCtxs.size());
		return g_cCtxs[iGpuID];
	}
	static int GetGpuID(CUcontext cContext)
	{
		for(int i = 0; i < (int)g_cCtxs.size(); i++){
			if(g_cCtxs[i] == cContext) return i;
		}
		return -1;
	}
	static int GPUCount()
	{
		return g_cCtxs.size();
	}
private:
	CCUDACtxMan()
	{
	}
	~CCUDACtxMan()
	{
	}
private:
	static vector<CUcontext> g_cCtxs;	
};

#endif
