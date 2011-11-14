#ifndef __OgreCudaHelper_h__
#define __OgreCudaHelper_h__


#define SPHSIMLIB_3D_SUPPORT
#include "OgreSimConfig.h"

#include <OgreHardwareBufferManager.h>
#include <OgreHardwareVertexBuffer.h>
#include <RenderSystems\GL\OgreGLRenderSystem.h>
#include <RenderSystems\GL\OgreGLHardwareVertexBuffer.h>
#include <RenderSystems\Direct3D9\OgreD3D9RenderSystem.h>
#include <RenderSystems\Direct3D9\OgreD3D9HardwareVertexBuffer.h>

#include <SimCudaHelper.h>

namespace OgreSim
{

	class OgreCudaHelper
	{
	public:
		OgreCudaHelper::OgreCudaHelper(OgreSim::Config *config, SimLib::SimCudaHelper *simCudaHelper);
		~OgreCudaHelper();

		void Initialize();

		// CUDA REGISTER
		void RegisterHardwareBuffer(Ogre::HardwareVertexBufferSharedPtr hardwareBuffer);
		void UnregisterHardwareBuffer(Ogre::HardwareVertexBufferSharedPtr hardwareBuffer);

		// CUDA MAPPING
		void MapBuffer(void **devPtr, Ogre::HardwareVertexBufferSharedPtr bufObj);
		void UnmapBuffer(void **devPtr, Ogre::HardwareVertexBufferSharedPtr bufObj);

	private:
		OgreSim::Config *mSnowConfig;
		SimLib::SimCudaHelper *mSimCudaHelper;

		enum RenderingMode
		{
			GL = 0,
			D3D9 = 1,
		};
		RenderingMode mRenderingMode;

	};
}


#endif
