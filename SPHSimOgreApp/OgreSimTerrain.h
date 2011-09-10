#ifndef __OgreSimTerrain_h_
#define __OgreSimTerrain_h_

#include "OgreSimConfig.h"
#include "OgreCudaHelper.h"

#include <OgreCamera.h>
#include <OgreEntity.h>
#include <OgreLogManager.h>
#include <OgreRoot.h>
#include <OgreViewport.h>
#include <OgreSceneManager.h>
#include <OgreRenderWindow.h>
#include <OgreConfigFile.h>

#include <OISEvents.h>
#include <OISInputManager.h>
#include <OISKeyboard.h>
#include <OISMouse.h>

#include <SdkTrays.h>
#include <SdkCameraMan.h>
#include <Terrain\OgreTerrain.h>
#include <Terrain\OgreTerrainGroup.h>
#include <Terrain\OgreTerrainQuadTreeNode.h>
#include <Terrain\OgreTerrainMaterialGeneratorA.h>
//#include <Terrain\OgreTerrainPaging.h>

#include <OgreHardwareVertexBuffer.h>

namespace OgreSim
{

	class OgreSimTerrain : public Ogre::FrameListener//, public OIS::KeyListener, public OIS::MouseListener
	{
	public:
		OgreSimTerrain(OgreSim::Config *config);
		~OgreSimTerrain(void);

		void createScene(Ogre::SceneManager* mSceneMgr, Ogre::Light* terrainLight);
		void destroyScene(Ogre::RenderWindow* renderWindow, Ogre::SceneManager* mSceneMgr);

		Ogre::TerrainMaterialGeneratorA::SM2Profile* getMaterialProfile();

		bool frameRenderingQueued(const Ogre::FrameEvent& evt);
		bool keyPressed (const OIS::KeyEvent &e);

		void SaveTerrains(bool onlyIfModified);
		void dumpTextures();

		Ogre::Terrain* getTerrain();

		float* getTerrainHeightData();
		Ogre::Vector4* getTerrainNormalData();
		int getTerrainSize();
		Ogre::Real getTerrainWorldSize();

		Ogre::Vector3 mTerrainPos;
		Ogre::Terrain* mTerrain;
		Ogre::TerrainGroup* mTerrainGroup;

	protected:
		void defineTerrain(long x, long y, bool flat = false);
		void initBlendMaps(Ogre::Terrain* terrain);

		Ogre::Terrain* createTerrain();

	private:
		Ogre::Vector4* convertNormalsToFloats(Ogre::PixelBox* terrainNormals, bool compressed);
		Ogre::ManualObject* createDebugNormals(Ogre::SceneManager* mSceneMgr);

		OgreSim::Config *mSnowConfig;

		Ogre::uint mTerrainSize;
		Ogre::Real mTerrainWorldSize;
		Ogre::Real mTerrainWorldScale;

		//Terrain stuff
		Ogre::uint8 mLayerEdit;
		Ogre::Real mUpdateCountDown;
		Ogre::Real mUpdateRate;

		bool mTerrainsImported;

		Ogre::ManualObject* mDebugNormalsManualObject;
		Ogre::SceneNode* mDebugNormalsNode;
		Ogre::SceneManager* mSceneMgr;
		Ogre::TerrainGlobalOptions* mTerrainGlobals;

	};

}
#endif // #ifndef __OgreSimTerrain_h_