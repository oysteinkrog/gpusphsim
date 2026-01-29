#ifndef __OgreCudaHelper_h__
#define __OgreCudaHelper_h__

// Updated for Ogre 14.x - Uses GL3Plus render system (D3D9 is deprecated)

#define SPHSIMLIB_3D_SUPPORT
#include "OgreSimConfig.h"

#include <Ogre.h>
#include <OgreHardwareBufferManager.h>
#include <OgreHardwareVertexBuffer.h>

// For Ogre 14.x, we use GL3Plus (D3D9 support removed as deprecated in Ogre 14)
// GL3Plus render system headers are included in .cpp file to avoid GL header conflicts
#include <SimCudaHelper.h>

namespace OgreSim
{

	class OgreCudaHelper
	{
	public:
		OgreCudaHelper(OgreSim::Config *config, SimLib::SimCudaHelper *simCudaHelper);
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
			GL3Plus = 0,
			GL = 1,
			Unknown = 2
		};
		RenderingMode mRenderingMode;

		// Helper to get GL buffer ID from Ogre hardware buffer
		unsigned int getGLBufferId(Ogre::HardwareVertexBufferSharedPtr hardwareBuffer);

	};
}


#endif
