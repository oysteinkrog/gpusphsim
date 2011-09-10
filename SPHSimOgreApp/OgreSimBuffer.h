#ifndef __OgreSimBuffer_h__
#define __OgreSimBuffer_h__

#include "Ogre.h"
#include "SimBuffer.h"
#include "OgreCudaHelper.h"
#include "OgreSimRenderable.h"

namespace OgreSim
{
	class OgreSimBuffer : public SimLib::SimBuffer
	{
	public:
		OgreSimBuffer::OgreSimBuffer(OgreSim::OgreSimRenderable *particlesMesh, OgreSim::OgreCudaHelper *OgreCudaHelper);
		~OgreSimBuffer();

		void SetOgreVertexBuffer(Ogre::HardwareVertexBufferSharedPtr  ogreVertexBuffer);

		virtual void MapBuffer();
		virtual void UnmapBuffer();

		virtual void Alloc(size_t size);
		virtual void Memset(int val);
		virtual void Free();
		virtual size_t GetSize();

	private:
		OgreSim::OgreSimRenderable *mParticlesMesh;
		OgreSim::OgreCudaHelper *mOgreCudaHelper;
		Ogre::HardwareVertexBufferSharedPtr mOgreVertexBuffer;
	};

}

#endif