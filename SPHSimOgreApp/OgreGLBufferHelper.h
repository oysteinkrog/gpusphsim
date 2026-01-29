// OgreGLBufferHelper.h
// Separate helper to extract GL buffer ID from Ogre buffers
// This avoids GL header conflicts with CUDA headers

#ifndef __OgreGLBufferHelper_h__
#define __OgreGLBufferHelper_h__

#include <OgreHardwareVertexBuffer.h>

namespace OgreSim
{
    // Get the OpenGL buffer ID from an Ogre hardware buffer
    // Returns 0 if the buffer is not a GL buffer or extraction fails
    unsigned int GetGLBufferId(Ogre::HardwareVertexBufferSharedPtr buffer);
}

#endif
