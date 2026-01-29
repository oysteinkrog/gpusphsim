#ifndef __OgreSimFluid_h_
#define __OgreSimFluid_h_

#include "OgreSimConfig.h"
#include "OgreCudaHelper.h"

#include <Ogre.h>
#include <OgreInput.h>
#include <OgreTrays.h>
#include <OgreCameraMan.h>

#include <Terrain/OgreTerrain.h>
#include <Terrain/OgreTerrainGroup.h>
#include <Terrain/OgreTerrainQuadTreeNode.h>
#include <Terrain/OgreTerrainMaterialGeneratorA.h>

#include "OgreSimTerrain.h"
#include "OgreCudaHelper.h"
#include "OgreSimBuffer.h"

#define SPHSIMLIB_3D_SUPPORT
#include "SimulationSystem.h"


namespace OgreSim
{
	class OgreSimFluid : public Ogre::FrameListener
	{
	public:
		OgreSimFluid(OgreSim::Config *config);
		~OgreSimFluid(void);

		void createScene(Ogre::RenderWindow* renderWindow, Ogre::SceneManager* mSceneMgr, OgreSimTerrain* terrain, Ogre::Light* terrainLight);
		void destroyScene(Ogre::RenderWindow* renderWindow, Ogre::SceneManager* mSceneMgr);

		bool frameRenderingQueued(const Ogre::FrameEvent& evt);
		bool frameStarted(const Ogre::FrameEvent &evt);
		bool frameEnded(const Ogre::FrameEvent &evt);

		// Updated for Ogre 14.x SDL2 input
		bool keyPressed(const OgreBites::KeyboardEvent& evt, bool ctrlDown, bool shiftDown);

		Ogre::SceneNode* mParticlesNode;
		OgreSimRenderable *mParticlesEntity;

	protected:

	private:
		Ogre::RenderWindow* mRenderWindow;
		void setParticleMaterial(Ogre::String particleMaterial);
		void configureTerrain(OgreSimTerrain* terrain);

		void SetScene(int scene);
		void FillTestData(int scene, SimLib::Sim::ParticleData &hParticles);

		OgreSim::Config *mSnowConfig;
		OgreSim::OgreCudaHelper* mOgreCudaHelper;
		SimLib::SimCudaHelper* mSimCudaHelper;

		int lastScene;
		bool mProgress;

		SimLib::SimulationSystem* mParticleSystem;
		int mNumParticles;
		int mVolumeSize;

		Ogre::Entity* sphereEntity;
		Ogre::SceneNode* sphereNode;

		Ogre::Vector3 spherePosition;
		Ogre::Vector3 sphereVelocity;
		Ogre::Vector3 sphereAccel;


		Ogre::ManualObject* mFluidGridObject;
		Ogre::SceneNode* mFluidGridNode;

	};
}
#endif // #ifndef __OgreSimFluid_h_
