#include "OgreSimFluid.h"
// OgreBites provides SDLK_* keycodes via OgreInput.h

using namespace Ogre;
using namespace OgreBites;

namespace OgreSim
{
	OgreSimFluid::OgreSimFluid(OgreSim::Config *snowConfig)
	: mSnowConfig(snowConfig)
	, mParticleSystem(NULL)
	, mProgress(true)
	{
		mSimCudaHelper = new SimLib::SimCudaHelper();
		mOgreCudaHelper = new OgreSim::OgreCudaHelper(mSnowConfig, mSimCudaHelper);
		mOgreCudaHelper->Initialize();
	}


	OgreSimFluid::~OgreSimFluid()
	{
	}


	void OgreSimFluid::destroyScene(Ogre::RenderWindow* renderWindow, Ogre::SceneManager* mSceneMgr)
	{
		if(mParticleSystem)
		{
			delete mParticlesEntity;
			delete mParticleSystem;
		}
		mRenderWindow = NULL;
	}

	void OgreSimFluid::configureTerrain(OgreSimTerrain* terrain)
	{
		if(terrain == NULL) return;

		Vector3 terrainPosition = terrain->mTerrainPos;
		//Vector3 terrainPosition = mParticlesNode->getPosition();
		int terrainSize = terrain->getTerrainSize();
		Real terrainWorldSize = terrain->getTerrainWorldSize();
		float* terrainHeightData = terrain->getTerrainHeightData();
		Vector4* terrainNormalData = terrain->getTerrainNormalData();

		if(terrainHeightData != 0 && terrainNormalData != 0)
			mParticleSystem->SetTerrainData(make_float3(terrainPosition.x,terrainPosition.y,terrainPosition.z), terrainHeightData, (float4*)terrainNormalData, terrainSize, terrainWorldSize);

		//OGRE_FREE(terrainHeightData, MEMCATEGORY_GENERAL);
		//OGRE_FREE(terrainNormalData, MEMCATEGORY_GENERAL);

	}

	void OgreSimFluid::setParticleMaterial(Ogre::String particleMaterial)
	{
		// Set a material for particles (Ogre 14.x uses MaterialPtr, not string)
		Ogre::MaterialPtr mat = MaterialManager::getSingleton().getByName(particleMaterial);
		if (mat)
		{
			mParticlesEntity->setMaterial(mat);
		}
		// Configure the particle shader (set ball size)
		Ogre::Technique *technique = mParticlesEntity->getMaterial()->getTechnique(0);
		Ogre::Pass *pass = technique->getPass(0);
		GpuProgramParametersSharedPtr params = pass->getVertexProgramParameters();

		if(params)
		{
			if (params->_findNamedConstantDefinition("pointRadius"))
				params->setNamedConstant( "pointRadius", mParticlesNode->getScale().x*(mParticleSystem->GetParticleSize()) );

			if (params->_findNamedConstantDefinition("pointScale"))
				params->setNamedConstant( "pointScale", (float)(mRenderWindow->getWidth() / tanf(90.0*0.5f*(float)Math::PI/180.0f )));
		}

	}
	void OgreSimFluid::createScene(Ogre::RenderWindow* renderWindow, Ogre::SceneManager* mSceneMgr, OgreSimTerrain* terrain, Ogre::Light* terrainLight)
	{
		mRenderWindow = renderWindow;
		if(mSnowConfig->fluidSettings.enabled)
		{

			mParticleSystem = new SimLib::SimulationSystem(mSnowConfig->fluidSettings.simpleSPH, mSimCudaHelper, false);

			mParticlesEntity = new OgreSimRenderable(mOgreCudaHelper, 1024);

			// Add particles to scene
			mParticlesNode = mSceneMgr->getRootSceneNode()->createChildSceneNode();
			mParticlesNode->attachObject(mParticlesEntity);
			mParticlesNode->setPosition(mSnowConfig->sceneSettings.fluidPosition);

			if(mSnowConfig->terrainSettings.enabled && mSnowConfig->fluidSettings.terrainCollisions)
				configureTerrain(terrain);

			mParticleSystem->SetFluidPosition(make_float3(mParticlesNode->getPosition().x, mParticlesNode->getPosition().y, mParticlesNode->getPosition().z));
			mParticleSystem->SetExternalBuffer(SimLib::Sim::BufferPosition,  mParticlesEntity->GetCudaBufferPosition());
			mParticleSystem->SetExternalBuffer(SimLib::Sim::BufferColor,  mParticlesEntity->GetCudaBufferColor());
			mParticleSystem->Init();


			// Ogre 14.x: Use range-based iteration for ConfigFile
			Ogre::ConfigFile* cfg = mSnowConfig->getCfg();
			auto settings = cfg->getSettingsBySection();
			auto it = settings.find("FluidParams");
			if (it != settings.end())
			{
				for (const auto& setting : it->second)
				{
					const String& name = setting.first;
					const String& value = setting.second;
					float val = StringConverter::parseReal(value);

					if(!StringUtil::startsWith(name, "//"))
						mParticleSystem->GetSettings()->SetValue(name, val);
				}
			}

			mVolumeSize = mParticleSystem->GetSettings()->GetValue("Grid World Size");
			mNumParticles = mParticleSystem->GetSettings()->GetValue("Particles Number");

			mParticlesEntity->Resize(mNumParticles);
			setParticleMaterial(mSnowConfig->generalSettings.fluidShader);


			// create material for fluid cube/grid
			Ogre::MaterialPtr gridMaterial = MaterialManager::getSingleton().create("FluidGridMaterial", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
			gridMaterial->setReceiveShadows(false);
			//gridMaterial->createTechnique()->createPass();
			gridMaterial->getTechnique(0)->setLightingEnabled(false);
			gridMaterial->getTechnique(0)->getPass(0)->setDiffuse(0, 0, 1, 0);
			gridMaterial->getTechnique(0)->getPass(0)->setAmbient(0, 0, 1);
			gridMaterial->getTechnique(0)->getPass(0)->setSelfIllumination(0, 0, 1);
			gridMaterial->load();

			// Draw cube of the fluid grid/simulation volume
			mFluidGridObject = mSceneMgr->createManualObject("ParticlesGrid");
			mFluidGridObject->begin("FluidGridMaterial", RenderOperation::OT_LINE_LIST, Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
			Vector3 min = Vector3(0,0,0);
			Vector3 max = min + Vector3(mVolumeSize,mVolumeSize,mVolumeSize);
			mFluidGridObject->position ( min.x, min.y, min.z );	mFluidGridObject->position ( max.x, min.y, min.z );
			mFluidGridObject->position ( min.x, max.y, min.z );	mFluidGridObject->position ( max.x, max.y, min.z );
			mFluidGridObject->position ( min.x, min.y, min.z );	mFluidGridObject->position ( min.x, max.y, min.z );
			mFluidGridObject->position ( max.x, min.y, min.z );	mFluidGridObject->position ( max.x, max.y, min.z );
			mFluidGridObject->position ( min.x, min.y, max.z );	mFluidGridObject->position ( max.x, min.y, max.z );
			mFluidGridObject->position ( min.x, max.y, max.z );	mFluidGridObject->position ( max.x, max.y, max.z );
			mFluidGridObject->position ( min.x, min.y, max.z );	mFluidGridObject->position ( min.x, max.y, max.z );
			mFluidGridObject->position ( max.x, min.y, max.z );	mFluidGridObject->position ( max.x, max.y, max.z );
			mFluidGridObject->position ( min.x, min.y, max.z );	mFluidGridObject->position ( min.x, min.y, min.z );
			mFluidGridObject->position ( min.x, max.y, max.z );	mFluidGridObject->position ( min.x, max.y, min.z );
			mFluidGridObject->position ( max.x, min.y, max.z );	mFluidGridObject->position ( max.x, min.y, min.z );
			mFluidGridObject->position ( max.x, max.y, max.z );	mFluidGridObject->position ( max.x, max.y, min.z );
			mFluidGridObject->end();

			mFluidGridNode = mSceneMgr->getRootSceneNode()->createChildSceneNode();
			mFluidGridNode->setPosition(mParticlesNode->getPosition());

			if(!mSnowConfig->fluidSettings.showFluidGrid)
				mFluidGridNode->detachObject(mFluidGridObject);
			else
				mFluidGridNode->attachObject(mFluidGridObject);

		}

		// set a scene
		SetScene(mSnowConfig->sceneSettings.fluidScene);
	}

	bool OgreSimFluid::frameStarted(const FrameEvent &evt)
	{
		return true;
	}

	bool OgreSimFluid::frameEnded(const FrameEvent &evt)
	{
		return true;
	}

	bool OgreSimFluid::frameRenderingQueued(const Ogre::FrameEvent& evt)
	{
		if(mParticleSystem)
		{
			mParticleSystem->Simulate(mProgress, mSnowConfig->fluidSettings.gridWallCollisions);
		}
		return true;
	}


	void OgreSimFluid::SetScene(int scene)
	{
		if(!mParticleSystem) return;

		lastScene = scene;

		mParticleSystem->SetScene(scene);
	}


	bool OgreSimFluid::keyPressed(const OgreBites::KeyboardEvent& evt, bool ctrlDown, bool shiftDown)
	{
		switch (evt.keysym.sym)
		{
		case '1':
			SetScene(1);
			break;
		case '2':
			SetScene(2);
			break;
		case '3':
			SetScene(3);
			break;
		case '4':
			SetScene(4);
			break;
		case '5':
			SetScene(5);
			break;
		case '6':
			SetScene(6);
			break;
		case '7':
			SetScene(7);
			break;
		case '8':
			SetScene(8);
			break;
		case '9':
			SetScene(9);
			break;


		// show fluid grid
		case 'g':
			{
				if(!mSnowConfig->fluidSettings.enabled)break;

				mSnowConfig->fluidSettings.showFluidGrid = !mSnowConfig->fluidSettings.showFluidGrid;

				if(!mSnowConfig->fluidSettings.showFluidGrid)
					mFluidGridNode->detachObject(mFluidGridObject);
				else
					mFluidGridNode->attachObject(mFluidGridObject);
			}
			break;
		case 'o':
			{
				mProgress = !mProgress;
			}
			break;
		case SDLK_LEFT:
			{
				Vector3 pos = mParticlesNode->getPosition();
				pos.x += 10;
				mParticlesNode->setPosition(pos);
				mFluidGridNode->setPosition(pos);
				mParticleSystem->SetFluidPosition(make_float3(mParticlesNode->getPosition().x, mParticlesNode->getPosition().y, mParticlesNode->getPosition().z));
			}
			break;
		case SDLK_RIGHT:
			{
				Vector3 pos = mParticlesNode->getPosition();
				pos.x -= 10;
				mParticlesNode->setPosition(pos);
				mFluidGridNode->setPosition(pos);
				mParticleSystem->SetFluidPosition(make_float3(mParticlesNode->getPosition().x, mParticlesNode->getPosition().y, mParticlesNode->getPosition().z));
			}
			break;
		case SDLK_UP:
			{
				Vector3 pos = mParticlesNode->getPosition();
				if (shiftDown)
					pos.y += 10;
				else
					pos.z += 10;
				mParticlesNode->setPosition(pos);
				mFluidGridNode->setPosition(pos);
				mParticleSystem->SetFluidPosition(make_float3(mParticlesNode->getPosition().x, mParticlesNode->getPosition().y, mParticlesNode->getPosition().z));
			}
			break;
		case SDLK_DOWN:
			{
				Vector3 pos = mParticlesNode->getPosition();
				if (shiftDown)
					pos.y -= 10;
				else
					pos.z -= 10;
				mParticlesNode->setPosition(pos);
				mFluidGridNode->setPosition(pos);
				mParticleSystem->SetFluidPosition(make_float3(mParticlesNode->getPosition().x, mParticlesNode->getPosition().y, mParticlesNode->getPosition().z));
			}
			break;


		case SDLK_PAGEUP:
			{
				if(!mSnowConfig->fluidSettings.enabled)break;


				if (shiftDown)
					mNumParticles *=2;
				else
					mNumParticles += 1000;

				mParticleSystem->SetNumParticles(mNumParticles);
				setParticleMaterial(mSnowConfig->generalSettings.fluidShader);

				SetScene(lastScene);

				//mParticlesEntity->setMaterial("shader/ParticleBall");
			}
			break;

		case SDLK_PAGEDOWN:
			{
				if(!mSnowConfig->fluidSettings.enabled)break;

				if (shiftDown)
					mNumParticles /=2;
				else
					mNumParticles -= 1000;

				mParticleSystem->SetNumParticles(mNumParticles);
				setParticleMaterial(mSnowConfig->generalSettings.fluidShader);

				SetScene(lastScene);

				//mParticlesEntity->setMaterial("shader/ParticleBall");
			}
			break;

		case SDLK_KP_PLUS:
			{
				float timestep = mParticleSystem->GetSettings()->GetValue("Timestep");
				if (shiftDown)
					timestep += 0.0001f;
				else
					timestep += 0.00001f;
				mParticleSystem->GetSettings()->SetValue("Timestep", timestep);
			}
			break;

		case SDLK_KP_MINUS:
			{
				float timestep = mParticleSystem->GetSettings()->GetValue("Timestep");

				if (shiftDown)
					timestep -= 0.0001f;
				else
					timestep -= 0.00001f;
				mParticleSystem->GetSettings()->SetValue("Timestep", timestep);
			}
			break;

		}
		return true;
	}




}
