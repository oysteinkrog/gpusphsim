#ifndef __SimCudaHelper_h__
#define __SimCudaHelper_h__


#include "Config.h"
// Enable CUDA-GL interop for particle rendering
#define SPHSIMLIB_3D_SUPPORT

// Basic CUDA includes - always needed
#if !defined(__CUDACC__)
#include <cuda.h>
#include <builtin_types.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#endif

#if !defined(__CUDACC__)
#ifdef SPHSIMLIB_3D_SUPPORT

// Use Windows OpenGL headers instead of GLEW
// We only need basic GL types (GLuint, etc.) for CUDA-GL interop
#ifdef _WIN32
#include <windows.h>
#include <GL/gl.h>
#endif
#include <cuda_gl_interop.h>

#endif
#endif

#if !defined(__CUDACC__)
#ifdef SPHSIMLIB_3D_SUPPORT
#ifdef SPHSIMLIB_D3D9_SUPPORT
#ifdef _WIN32
#include <d3dx9.h>

// includes, cuda D3D interop
#include <cuda_d3d9_interop.h>
#endif
#endif
#endif
#endif

namespace SimLib
{
	class SimCudaHelper
	{
	public:
		SimCudaHelper();
		~SimCudaHelper();

		void Initialize(int cudaDevice);

#if !defined(__CUDACC__)
#ifdef SPHSIMLIB_3D_SUPPORT
		void InitializeGL(int cudaDevice);
#ifdef SPHSIMLIB_D3D9_SUPPORT
		void InitializeD3D9(int cudaDevice, IDirect3DDevice9 *pDxDevice);
#endif

		// CUDA REGISTER
		static cudaError_t RegisterGLBuffer(GLuint vbo);
		static cudaError_t UnregisterGLBuffer(GLuint vbo);

#ifdef SPHSIMLIB_D3D9_SUPPORT
#ifdef _WIN32
		static cudaError_t RegisterD3D9Buffer(IDirect3DResource9 * pResource);
		static cudaError_t UnregisterD3D9Buffer(IDirect3DResource9 * pResource);
#endif
		// CUDA MAPPING
		static cudaError_t MapBuffer(void **devPtr, IDirect3DResource9* pResource);
		static cudaError_t UnmapBuffer(void **devPtr, IDirect3DResource9* pResource);
#endif

		static cudaError_t MapBuffer(void **devPtr, GLuint bufObj);
		static cudaError_t UnmapBuffer(void **devPtr, GLuint bufObj);

#endif
#endif

		int PrintDevices(int deviceSelected);
		
		bool IsFermi();
	private:

		int Init(int cudaDevice);
		void CheckError(const char *msg);
		void CheckError(cudaError_t err, const char *msg);

		cudaDeviceProp* mDeviceProp;
	};
}


#endif