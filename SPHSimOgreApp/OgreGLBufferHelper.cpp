// OgreGLBufferHelper.cpp
// Separate translation unit to extract GL buffer ID from Ogre buffers
// This file includes GL3Plus headers which conflict with CUDA headers
// so it must be compiled separately

#include "OgreGLBufferHelper.h"

#include <OgreRoot.h>
#include <OgreRenderSystem.h>
#include <OgreHardwareBuffer.h>
#include <OgreLogManager.h>

// Include GL3Plus specific headers for buffer access
// These headers bring in OpenGL definitions that conflict with CUDA
#include <OgreGL3PlusHardwareBuffer.h>

namespace OgreSim
{
    unsigned int GetGLBufferId(Ogre::HardwareVertexBufferSharedPtr buffer)
    {
        if (!buffer)
            return 0;

        // Check if we're using a GL render system
        Ogre::RenderSystem* rs = Ogre::Root::getSingleton().getRenderSystem();
        if (!rs)
            return 0;

        const Ogre::String& rsName = rs->getName();
        if (rsName.find("OpenGL") == Ogre::String::npos && rsName.find("GL3Plus") == Ogre::String::npos)
            return 0;

        // Get the GL3Plus implementation and extract the buffer ID
        const Ogre::GL3PlusHardwareBuffer* glBuffer = buffer->_getImpl<Ogre::GL3PlusHardwareBuffer>();
        if (glBuffer)
            return glBuffer->getGLBufferId();

        return 0;
    }
}
