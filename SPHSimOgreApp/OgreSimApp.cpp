#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "OgreSimApp.h"
// OgreBites provides SDLK_* keycodes via OgreInput.h

using namespace Ogre;
using namespace OgreBites;

namespace OgreSim
{

	//-------------------------------------------------------------------------------------
	SnowApplication::SnowApplication(void)
		: mSimulationPaused(false)
		, mScreenCaptureFrame(0)
		, mScreenCapture(false)
		, mDestroyed(false)
		, mOgreSimTerrain(nullptr)
		, mOgreSimFluid(nullptr)
		, mShadowMode(SHADOWS_NONE)
		, mShadowsMenu(nullptr)
	{
	}

	//-------------------------------------------------------------------------------------
	SnowApplication::~SnowApplication(void)
	{
	}

	void SnowApplication::setupResources(void)
	{
		BaseApplication::setupResources();
	}

	//-------------------------------------------------------------------------------------
	void SnowApplication::destroyScene(void)
	{
		if(mDestroyed) return;

		mDestroyed = true;

		// Destroy fluid
		if (mOgreSimFluid)
		{
			mOgreSimFluid->destroyScene(mWindow, mSceneMgr);
			delete mOgreSimFluid;
			mOgreSimFluid = nullptr;
		}

		// Destroy terrain
		if (mOgreSimTerrain)
		{
			mOgreSimTerrain->destroyScene(mWindow, mSceneMgr);
			delete mOgreSimTerrain;
			mOgreSimTerrain = nullptr;
		}
	}

	//-------------------------------------------------------------------------------------
	void SnowApplication::createScene(void)
	{
		mDestroyed = false;

		srand(time(NULL));

		mOgreSimTerrain = new OgreSimTerrain(mSnowConfig);
		mOgreSimFluid = new OgreSimFluid(mSnowConfig);

		MaterialManager::getSingleton().setDefaultTextureFiltering(TFO_ANISOTROPIC);
		MaterialManager::getSingleton().setDefaultAnisotropy(7);

		// Setup lighting
		Vector3 lightdir(0, -0.85, 0);
		lightdir.normalise();

		// PRIMARY LIGHT - In Ogre 14.x, lights must be attached to SceneNodes
		Light* primaryLight = mSceneMgr->createLight("PrimaryLight");
		primaryLight->setType(Light::LT_DIRECTIONAL);
		primaryLight->setDiffuseColour(ColourValue::White);
		primaryLight->setSpecularColour(ColourValue(0.4, 0.4, 0.4));

		// Create a scene node for the light and set direction via the node
		SceneNode* lightNode = mSceneMgr->getRootSceneNode()->createChildSceneNode();
		lightNode->attachObject(primaryLight);
		lightNode->setDirection(lightdir);

		mSceneMgr->setAmbientLight(ColourValue(0.2, 0.2, 0.2));

		mWindow->getViewport(0)->setBackgroundColour(mSnowConfig->sceneSettings.backgroundColor);

		// Set a skybox
		mSceneMgr->setSkyBox(mSnowConfig->sceneSettings.skyBoxMaterial.length()>0, mSnowConfig->sceneSettings.skyBoxMaterial);

		// Create terrain
		if(mSnowConfig->terrainSettings.enabled)
			mOgreSimTerrain->createScene(mSceneMgr, primaryLight);

		// Create fluid
		mOgreSimFluid->createScene(mWindow, mSceneMgr, mOgreSimTerrain, primaryLight);

		// Place camera
		Vector3 cameraPos = mSnowConfig->sceneSettings.cameraPosition;
		if(mSnowConfig->sceneSettings.cameraRelativeToFluid && mSnowConfig->fluidSettings.enabled)
			cameraPos += mOgreSimFluid->mParticlesNode->getPosition();

		mCamera->getParentSceneNode()->setPosition(cameraPos);
		mCamera->getParentSceneNode()->setOrientation(mSnowConfig->sceneSettings.cameraOrientation);
		mCamera->setNearClipDistance(0.1);
		mCamera->setFarClipDistance(50000);
		if (mRoot->getRenderSystem()->getCapabilities()->hasCapability(RSC_INFINITE_FAR_PLANE))
		{
			mCamera->setFarClipDistance(0);   // enable infinite far clip distance if we can
		}

		return;
	}

	void SnowApplication::createFrameListener()
	{
		BaseApplication::createFrameListener();

		setupControls();
	}

	bool SnowApplication::frameStarted(const FrameEvent &evt)
	{
		if(mDestroyed) return false;

		mOgreSimFluid->frameStarted(evt);
		return BaseApplication::frameStarted(evt);
	}

	bool SnowApplication::frameEnded(const FrameEvent &evt)
	{
		if(mDestroyed) return false;

		mOgreSimFluid->frameEnded(evt);
		return BaseApplication::frameEnded(evt);

	}

	bool SnowApplication::frameRenderingQueued(const Ogre::FrameEvent& evt)
	{
		if(mDestroyed) return false;

		if(mScreenCapture)
		{
			char tmp[1000]={0};
			sprintf(tmp,"scr%i.png",mScreenCaptureFrame++);
			mWindow->writeContentsToFile(tmp);
		}

		if(!mSimulationPaused) {
			mOgreSimTerrain->frameRenderingQueued(evt);
			mOgreSimFluid->frameRenderingQueued(evt);
		}

		return BaseApplication::frameRenderingQueued(evt);  // don't forget the parent updates!
	}


	bool SnowApplication::keyPressed(const OgreBites::KeyboardEvent& evt)
	{
		// toggle visibility of help dialog
		if (evt.keysym.sym == 'h' || evt.keysym.sym == SDLK_F1)
		{
			if (!mTrayMgr->isDialogVisible()) mTrayMgr->showOkDialog("Help", "");
			else mTrayMgr->closeDialog();
		}

		// don't process any more keys if dialog is up
		if (mTrayMgr->isDialogVisible()) return true;


		switch (evt.keysym.sym)
		{
		case SDLK_PRINTSCREEN:
			// take a screenshot
			{
				mWindow->writeContentsToTimestampedFile("screenshot", ".png");
			}
			break;
		case 'f':
			{
				mTrayMgr->toggleAdvancedFrameStats();
			}
			break;
		case 'r':
			// cycle polygon rendering mode
			{
				Ogre::String newVal;
				Ogre::PolygonMode pm;

				switch (mCamera->getPolygonMode())
				{
				case Ogre::PM_SOLID:
					newVal = "Wireframe";
					pm = Ogre::PM_WIREFRAME;
					break;
				case Ogre::PM_WIREFRAME:
					newVal = "Points";
					pm = Ogre::PM_POINTS;
					break;
				default:
					newVal = "Solid";
					pm = Ogre::PM_SOLID;
				}

				mCamera->setPolygonMode(pm);
			}
			break;
		case SDLK_F5:
			{
				// refresh all textures
				Ogre::TextureManager::getSingleton().reloadAll();
			}
			break;

		case 's':
			// CTRL-S to save
			if (evt.keysym.mod & KMOD_CTRL)
			{
				Ogre::LogManager::getSingleton().logMessage("Saving terrain");
				mOgreSimTerrain->SaveTerrains(false);
			}
			break;
		case SDLK_F9:
			mScreenCapture = !mScreenCapture;
			if(mScreenCapture)
				mScreenCaptureFrame = 0;
			break;
		case SDLK_F10:
			// dump
			{
				mOgreSimTerrain->dumpTextures();
			}
			break;
		case SDLK_F11:
			// dump
			{
				mOgreSimTerrain->getTerrain();
			}
			break;
		case 'p':
			// pause
			{
				mSimulationPaused = !mSimulationPaused;
			}
			break;
		}


		mOgreSimTerrain->keyPressed(evt);

		// Pass modifier key states to fluid (use OgreBites KMOD_ flags)
		bool ctrlDown = (evt.keysym.mod & KMOD_CTRL) != 0;
		bool shiftDown = (evt.keysym.mod & KMOD_SHIFT) != 0;
		mOgreSimFluid->keyPressed(evt, ctrlDown, shiftDown);

		return BaseApplication::keyPressed(evt);
	}


	/*-----------------------------------------------------------------------------
	| Extends setupView to change some initial camera settings for this sample.
	-----------------------------------------------------------------------------*/
	void SnowApplication::createCamera()
	{
		BaseApplication::createCamera();

	}


	void SnowApplication::setupControls()
	{
		mCameraMan->setTopSpeed(500);

		setDragLook(true);

		mTrayMgr->showCursor();

		// make room for the controls
		mTrayMgr->showFrameStats(TL_TOPRIGHT);
		mTrayMgr->toggleAdvancedFrameStats();
	}


	void SnowApplication::itemSelected(SelectMenu* menu)
	{
		if (menu == mShadowsMenu)
		{
			mShadowMode = (ShadowMode)mShadowsMenu->getSelectionIndex();
			changeShadows();
		}
	}

	void SnowApplication::checkBoxToggled(CheckBox* box)
	{
	}

	void SnowApplication::windowClosed(Ogre::RenderWindow* rw)
	{
		BaseApplication::windowClosed(rw);
	}


	MaterialPtr SnowApplication::buildDepthShadowMaterial(const String& textureName)
	{
		String matName = "DepthShadows/" + textureName;

		MaterialPtr ret = MaterialManager::getSingleton().getByName(matName);
		if (!ret)
		{
			MaterialPtr baseMat = MaterialManager::getSingleton().getByName("Ogre/shadow/depth/integrated/pssm");
			ret = baseMat->clone(matName);
			Pass* p = ret->getTechnique(0)->getPass(0);
			p->getTextureUnitState("diffuse")->setTextureName(textureName);

			Vector4 splitPoints;
			const PSSMShadowCameraSetup::SplitPointList& splitPointList =
				static_cast<PSSMShadowCameraSetup*>(mPSSMSetup.get())->getSplitPoints();
			for (int i = 0; i < 3; ++i)
			{
				splitPoints[i] = splitPointList[i];
			}
			p->getFragmentProgramParameters()->setNamedConstant("pssmSplitPoints", splitPoints);


		}

		return ret;
	}

	void SnowApplication::changeShadows()
	{
		configureShadows(mShadowMode != SHADOWS_NONE, mShadowMode == SHADOWS_DEPTH);
	}

	void SnowApplication::configureShadows(bool enabled, bool depthShadows)
	{
		TerrainMaterialGeneratorA::SM2Profile* matProfile = mOgreSimTerrain->getMaterialProfile();

		matProfile->setReceiveDynamicShadowsEnabled(enabled);
	#ifdef SHADOWS_IN_LOW_LOD_MATERIAL
		matProfile->setReceiveDynamicShadowsLowLod(true);
	#else
		matProfile->setReceiveDynamicShadowsLowLod(false);
	#endif

		// Default materials
		for (EntityList::iterator i = mHouseList.begin(); i != mHouseList.end(); ++i)
		{
			(*i)->setMaterialName("Examples/TudorHouse");
		}

		if (enabled)
		{
			// General scene setup
			mSceneMgr->setShadowTechnique(SHADOWTYPE_TEXTURE_ADDITIVE_INTEGRATED);
			mSceneMgr->setShadowFarDistance(3000);

			// 3 textures per directional light (PSSM)
			mSceneMgr->setShadowTextureCountPerLightType(Ogre::Light::LT_DIRECTIONAL, 3);

			if (!mPSSMSetup)
			{
				// shadow camera setup
				PSSMShadowCameraSetup* pssmSetup = new PSSMShadowCameraSetup();
				pssmSetup->setSplitPadding(mCamera->getNearClipDistance());
				pssmSetup->calculateSplitPoints(3, mCamera->getNearClipDistance(), mSceneMgr->getShadowFarDistance());
				pssmSetup->setOptimalAdjustFactor(0, 2);
				pssmSetup->setOptimalAdjustFactor(1, 1);
				pssmSetup->setOptimalAdjustFactor(2, 0.5);

				mPSSMSetup.reset(pssmSetup);

			}
			mSceneMgr->setShadowCameraSetup(mPSSMSetup);

			if (depthShadows)
			{
				mSceneMgr->setShadowTextureCount(3);
				mSceneMgr->setShadowTextureConfig(0, 2048, 2048, PF_FLOAT32_R);
				mSceneMgr->setShadowTextureConfig(1, 1024, 1024, PF_FLOAT32_R);
				mSceneMgr->setShadowTextureConfig(2, 1024, 1024, PF_FLOAT32_R);
				mSceneMgr->setShadowTextureSelfShadow(true);
				mSceneMgr->setShadowCasterRenderBackFaces(true);
				mSceneMgr->setShadowTextureCasterMaterial(MaterialManager::getSingleton().getByName("PSSM/shadow_caster"));

				MaterialPtr houseMat = buildDepthShadowMaterial("fw12b.jpg");
				for (EntityList::iterator i = mHouseList.begin(); i != mHouseList.end(); ++i)
				{
					(*i)->setMaterial(houseMat);
				}

			}
			else
			{
				mSceneMgr->setShadowTextureCount(3);
				mSceneMgr->setShadowTextureConfig(0, 2048, 2048, PF_X8B8G8R8);
				mSceneMgr->setShadowTextureConfig(1, 1024, 1024, PF_X8B8G8R8);
				mSceneMgr->setShadowTextureConfig(2, 1024, 1024, PF_X8B8G8R8);
				mSceneMgr->setShadowTextureSelfShadow(false);
				mSceneMgr->setShadowCasterRenderBackFaces(false);
				mSceneMgr->setShadowTextureCasterMaterial(MaterialPtr());
			}

			// setReceiveDynamicShadowsDepth removed in Ogre 14
			matProfile->setReceiveDynamicShadowsPSSM(static_cast<PSSMShadowCameraSetup*>(mPSSMSetup.get()));

		}
		else
		{
			mSceneMgr->setShadowTechnique(SHADOWTYPE_NONE);
		}


	}


}
