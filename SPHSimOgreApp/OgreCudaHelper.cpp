// Updated for Ogre 14.x - Uses GL3Plus render system (D3D9 is deprecated)

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <algorithm>

#include <cuda_runtime_api.h>

#include "OgreCudaHelper.h"
#include "OgreGLBufferHelper.h"

// Note: GL buffer ID extraction is done via OgreGLBufferHelper.cpp to avoid
// GL header conflicts with CUDA headers.

using namespace Ogre;

namespace OgreSim
{
	OgreCudaHelper::OgreCudaHelper(OgreSim::Config *config, SimLib::SimCudaHelper *simCudaHelper)
	{
		mSnowConfig = config;
		mSimCudaHelper = simCudaHelper;
		mRenderingMode = Unknown;
	}

	OgreCudaHelper::~OgreCudaHelper()
	{
	}

	void OgreCudaHelper::Initialize()
	{
		int cudaDevice = mSnowConfig->generalSettings.cudadevice;

		Ogre::RenderSystem* renderSystem = Ogre::Root::getSingleton().getRenderSystem();
		const String& rsName = renderSystem->getName();

		if (rsName.find("GL3Plus") != String::npos || rsName.find("OpenGL 3+") != String::npos)
		{
			mRenderingMode = GL3Plus;
		}
		else if (rsName.find("OpenGL") != String::npos)
		{
			mRenderingMode = GL;
		}
		else if (rsName.find("Direct3D11") != String::npos)
		{
			mRenderingMode = Unknown;
		}
		else
		{
			mRenderingMode = Unknown;
		}

		// Use GL interop initialization for OpenGL render systems
		if (mRenderingMode == GL3Plus || mRenderingMode == GL)
		{
			mSimCudaHelper->InitializeGL(cudaDevice);
			LogManager::getSingleton().logMessage("OgreCudaHelper: Using render system '" + rsName +
				"' with CUDA-GL interop initialization");
		}
		else
		{
			// Fallback to standard CUDA initialization for non-GL render systems
			mSimCudaHelper->Initialize(cudaDevice);
			LogManager::getSingleton().logMessage("OgreCudaHelper: Using render system '" + rsName +
				"' with standard CUDA initialization");
		}
	}

	unsigned int OgreCudaHelper::getGLBufferId(Ogre::HardwareVertexBufferSharedPtr hardwareBuffer)
	{
		// Use the separate helper function that includes GL3Plus headers
		// This avoids conflicts between GL and CUDA headers
		unsigned int bufferId = OgreSim::GetGLBufferId(hardwareBuffer);

		if (bufferId == 0)
		{
			static bool warnedOnce = false;
			if (!warnedOnce)
			{
				LogManager::getSingleton().logMessage("OgreCudaHelper: Failed to get GL buffer ID. "
					"CUDA-GL interop may not work correctly.");
				warnedOnce = true;
			}
		}

		return bufferId;
	}

	void OgreCudaHelper::RegisterHardwareBuffer(Ogre::HardwareVertexBufferSharedPtr hardwareBuffer)
	{
		if (mRenderingMode == GL3Plus || mRenderingMode == GL)
		{
			unsigned int bufferId = getGLBufferId(hardwareBuffer);
			if (bufferId != 0)
			{
				cudaError_t err = SimLib::SimCudaHelper::RegisterGLBuffer(bufferId);
				if (err != cudaSuccess)
				{
					LogManager::getSingleton().logMessage("OgreCudaHelper: Failed to register GL buffer: " +
						String(cudaGetErrorString(err)));
				}
			}
		}
	}

	void OgreCudaHelper::UnregisterHardwareBuffer(Ogre::HardwareVertexBufferSharedPtr hardwareBuffer)
	{
		if (mRenderingMode == GL3Plus || mRenderingMode == GL)
		{
			unsigned int bufferId = getGLBufferId(hardwareBuffer);
			if (bufferId != 0)
			{
				cudaError_t err = SimLib::SimCudaHelper::UnregisterGLBuffer(bufferId);
				if (err != cudaSuccess)
				{
					LogManager::getSingleton().logMessage("OgreCudaHelper: Failed to unregister GL buffer: " +
						String(cudaGetErrorString(err)));
				}
			}
		}
	}

	void OgreCudaHelper::MapBuffer(void **devPtr, Ogre::HardwareVertexBufferSharedPtr hardwareBuffer)
	{
		*devPtr = nullptr;

		if (mRenderingMode == GL3Plus || mRenderingMode == GL)
		{
			unsigned int bufferId = getGLBufferId(hardwareBuffer);
			if (bufferId != 0)
			{
				cudaError_t err = SimLib::SimCudaHelper::MapBuffer(devPtr, bufferId);
				if (err != cudaSuccess)
				{
					LogManager::getSingleton().logMessage("OgreCudaHelper: Failed to map GL buffer: " +
						String(cudaGetErrorString(err)));
					*devPtr = nullptr;
				}
			}
		}
	}

	void OgreCudaHelper::UnmapBuffer(void **devPtr, Ogre::HardwareVertexBufferSharedPtr hardwareBuffer)
	{
		if (mRenderingMode == GL3Plus || mRenderingMode == GL)
		{
			unsigned int bufferId = getGLBufferId(hardwareBuffer);
			if (bufferId != 0)
			{
				cudaError_t err = SimLib::SimCudaHelper::UnmapBuffer(devPtr, bufferId);
				if (err != cudaSuccess)
				{
					LogManager::getSingleton().logMessage("OgreCudaHelper: Failed to unmap GL buffer: " +
						String(cudaGetErrorString(err)));
				}
			}
		}
		*devPtr = nullptr;
	}
}
