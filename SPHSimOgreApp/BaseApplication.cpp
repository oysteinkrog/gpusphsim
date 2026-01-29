/*
-----------------------------------------------------------------------------
Filename:    BaseApplication.cpp
-----------------------------------------------------------------------------
Updated for Ogre 14.x - Uses ApplicationContext with SDL2 input
-----------------------------------------------------------------------------
*/

#include "OgreCudaHelper.h"
#include "BaseApplication.h"
// OgreBites provides SDLK_* keycodes via OgreInput.h

#if OGRE_PLATFORM == OGRE_PLATFORM_WIN32
#include "../res/resource.h"
#endif

using namespace OgreBites;

//-------------------------------------------------------------------------------------
BaseApplication::BaseApplication(void)
	: ApplicationContext("OgreSim")
	, mRoot(nullptr)
	, mCamera(nullptr)
	, mSceneMgr(nullptr)
	, mWindow(nullptr)
	, mTrayMgr(nullptr)
	, mCameraMan(nullptr)
	, mDetailsPanel(nullptr)
	, mCursorWasVisible(false)
	, mShutDown(false)
	, mDragLook(false)
	, mSnowConfig(nullptr)
{
}

//-------------------------------------------------------------------------------------
BaseApplication::~BaseApplication(void)
{
	delete mSnowConfig;
}

//-------------------------------------------------------------------------------------
void BaseApplication::go(void)
{
	initApp();

	if (mRoot->getRenderSystem() != nullptr)
	{
		// Use a custom render loop that properly pumps SDL events
		// mRoot->startRendering() doesn't pump SDL events in ApplicationContext
		while (!mShutDown)
		{
			// Poll SDL events (required for window interaction)
			pollEvents();

			// Render one frame
			if (!mRoot->renderOneFrame())
				break;
		}
	}

	closeApp();
}

//-------------------------------------------------------------------------------------
void BaseApplication::setup(void)
{
	// Load config first (before calling base setup)
	mSnowConfig = new OgreSim::Config();

	// Call base class setup - this creates the window and initializes Ogre
	ApplicationContext::setup();

	// Add input listener
	addInputListener(this);

	// Get pointers from ApplicationContext
	mRoot = getRoot();
	mWindow = getRenderWindow();


	// Set logging level
	Ogre::LogManager::getSingleton().setMinLogLevel(
		mSnowConfig->generalSettings.logLevel == Ogre::LL_LOW ? Ogre::LML_TRIVIAL :
		mSnowConfig->generalSettings.logLevel == Ogre::LL_NORMAL ? Ogre::LML_NORMAL :
		Ogre::LML_CRITICAL);

#if OGRE_PLATFORM == OGRE_PLATFORM_WIN32
	// Set window icon
	HWND hwnd;
	mWindow->getCustomAttribute("WINDOW", (void*)&hwnd);
	LONG iconID = (LONG)LoadIcon(GetModuleHandle(0), MAKEINTRESOURCE(IDI_APPICON));
	SetClassLongPtr(hwnd, GCLP_HICON, iconID);
#endif

	// Setup resources
	setupResources();

	// Choose scene manager
	chooseSceneManager();

	// Create camera
	createCamera();

	// Create viewports
	createViewports();

	// Set default mipmap level
	Ogre::TextureManager::getSingleton().setDefaultNumMipmaps(5);

	// Load resources
	loadResources();

	// Create frame listener (TrayManager, etc.)
	createFrameListener();

	// Create the scene
	createScene();
}

//-------------------------------------------------------------------------------------
void BaseApplication::shutdown(void)
{
	destroyScene();

	if (mCameraMan)
	{
		delete mCameraMan;
		mCameraMan = nullptr;
	}

	if (mTrayMgr)
	{
		delete mTrayMgr;
		mTrayMgr = nullptr;
	}

	ApplicationContext::shutdown();
}

//-------------------------------------------------------------------------------------
void BaseApplication::chooseSceneManager(void)
{
	// Get the SceneManager, in this case a generic one
	mSceneMgr = mRoot->createSceneManager();

	// Register the SceneManager with RTShaderSystem (required for Ogre 14.x GL3Plus)
	if (Ogre::RTShader::ShaderGenerator::getSingletonPtr())
	{
		Ogre::RTShader::ShaderGenerator::getSingleton().addSceneManager(mSceneMgr);
	}
}

//-------------------------------------------------------------------------------------
void BaseApplication::createCamera(void)
{
	// Create the camera
	mCamera = mSceneMgr->createCamera("PlayerCam");

	// In Ogre 14.x, cameras must be attached to scene nodes for positioning
	Ogre::SceneNode* camNode = mSceneMgr->getRootSceneNode()->createChildSceneNode();
	camNode->attachObject(mCamera);

	// Position it at 500 in Z direction
	camNode->setPosition(Ogre::Vector3(0, 0, 80));
	// Look back along -Z
	camNode->lookAt(Ogre::Vector3(0, 0, -300), Ogre::Node::TS_WORLD);
	mCamera->setNearClipDistance(5);
	mCamera->setAutoAspectRatio(true);

	// Create a default camera controller
	mCameraMan = new OgreBites::CameraMan(camNode);
}

//-------------------------------------------------------------------------------------
void BaseApplication::createFrameListener(void)
{
	Ogre::LogManager::getSingletonPtr()->logMessage("*** Initializing TrayManager ***");

	// Create TrayManager
	mTrayMgr = new OgreBites::TrayManager("InterfaceName", mWindow, this);
	mTrayMgr->showFrameStats(OgreBites::TL_BOTTOMLEFT);
	mTrayMgr->hideCursor();

	if (!mSnowConfig->generalSettings.showOgreGui)
		mTrayMgr->hideAll();

	// Create a params panel for displaying sample details
	Ogre::StringVector items;
	items.push_back("cam.pX");
	items.push_back("cam.pY");
	items.push_back("cam.pZ");
	items.push_back("");
	items.push_back("cam.oW");
	items.push_back("cam.oX");
	items.push_back("cam.oY");
	items.push_back("cam.oZ");
	items.push_back("");
	items.push_back("Filtering");
	items.push_back("Poly Mode");

	mDetailsPanel = mTrayMgr->createParamsPanel(OgreBites::TL_NONE, "DetailsPanel", 200, items);
	mDetailsPanel->setParamValue(9, "Bilinear");
	mDetailsPanel->setParamValue(10, "Solid");
	mDetailsPanel->hide();

	mRoot->addFrameListener(this);
}

//-------------------------------------------------------------------------------------
void BaseApplication::destroyScene(void)
{
}

//-------------------------------------------------------------------------------------
void BaseApplication::createViewports(void)
{
	// Create one viewport, entire window
	Ogre::Viewport* vp = mWindow->addViewport(mCamera);
	vp->setBackgroundColour(Ogre::ColourValue(0, 0, 0));

	// Alter the camera aspect ratio to match the viewport
	mCamera->setAspectRatio(
		Ogre::Real(vp->getActualWidth()) / Ogre::Real(vp->getActualHeight()));
}

//-------------------------------------------------------------------------------------
void BaseApplication::setupResources(void)
{
	// Load resource paths from config file
	Ogre::ConfigFile cf;
	cf.load("resources.cfg");

	// Go through all sections & settings in the file (Ogre 14.x style)
	auto sections = cf.getSettingsBySection();
	for (const auto& sec : sections)
	{
		const Ogre::String& secName = sec.first;
		const Ogre::ConfigFile::SettingsMultiMap& settings = sec.second;

		for (const auto& setting : settings)
		{
			const Ogre::String& typeName = setting.first;
			const Ogre::String& archName = setting.second;
			Ogre::ResourceGroupManager::getSingleton().addResourceLocation(
				archName, typeName, secName);
		}
	}
}

//-------------------------------------------------------------------------------------
void BaseApplication::loadResources(void)
{
	Ogre::ResourceGroupManager::getSingleton().initialiseAllResourceGroups();
}

//-------------------------------------------------------------------------------------
bool BaseApplication::frameStarted(const Ogre::FrameEvent& evt)
{
	return true;
}

//-------------------------------------------------------------------------------------
bool BaseApplication::frameEnded(const Ogre::FrameEvent& evt)
{
	return true;
}

//-------------------------------------------------------------------------------------
bool BaseApplication::frameRenderingQueued(const Ogre::FrameEvent& evt)
{
	if (mWindow->isClosed())
		return false;

	if (mShutDown)
		return false;

	// Update TrayManager
	mTrayMgr->frameRendered(evt);

	if (!mTrayMgr->isDialogVisible())
	{
		mCameraMan->frameRendered(evt);   // if dialog isn't up, then update the camera
		if (mDetailsPanel->isVisible())   // if details panel is visible, then update its contents
		{
			mDetailsPanel->setParamValue(0, Ogre::StringConverter::toString(mCamera->getDerivedPosition().x));
			mDetailsPanel->setParamValue(1, Ogre::StringConverter::toString(mCamera->getDerivedPosition().y));
			mDetailsPanel->setParamValue(2, Ogre::StringConverter::toString(mCamera->getDerivedPosition().z));
			mDetailsPanel->setParamValue(4, Ogre::StringConverter::toString(mCamera->getDerivedOrientation().w));
			mDetailsPanel->setParamValue(5, Ogre::StringConverter::toString(mCamera->getDerivedOrientation().x));
			mDetailsPanel->setParamValue(6, Ogre::StringConverter::toString(mCamera->getDerivedOrientation().y));
			mDetailsPanel->setParamValue(7, Ogre::StringConverter::toString(mCamera->getDerivedOrientation().z));
		}
	}

	return true;
}

//-------------------------------------------------------------------------------------
bool BaseApplication::keyPressed(const OgreBites::KeyboardEvent& evt)
{
	// Track key state
	mKeysDown.insert(evt.keysym.sym);

	if (mTrayMgr->isDialogVisible()) return true;   // don't process any more keys if dialog is up

	if (evt.keysym.sym == 'f')   // toggle visibility of advanced frame stats
	{
		mTrayMgr->toggleAdvancedFrameStats();
	}
	else if (evt.keysym.sym == 't')   // cycle texture filtering mode
	{
		Ogre::String newVal;
		Ogre::TextureFilterOptions tfo;
		unsigned int aniso;

		switch (mDetailsPanel->getParamValue(9)[0])
		{
		case 'B':
			newVal = "Trilinear";
			tfo = Ogre::TFO_TRILINEAR;
			aniso = 1;
			break;
		case 'T':
			newVal = "Anisotropic";
			tfo = Ogre::TFO_ANISOTROPIC;
			aniso = 8;
			break;
		case 'A':
			newVal = "None";
			tfo = Ogre::TFO_NONE;
			aniso = 1;
			break;
		default:
			newVal = "Bilinear";
			tfo = Ogre::TFO_BILINEAR;
			aniso = 1;
		}

		Ogre::MaterialManager::getSingleton().setDefaultTextureFiltering(tfo);
		Ogre::MaterialManager::getSingleton().setDefaultAnisotropy(aniso);
		mDetailsPanel->setParamValue(9, newVal);
	}
	else if (evt.keysym.sym == 'r')   // cycle polygon rendering mode
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
		mDetailsPanel->setParamValue(10, newVal);
	}
	else if (evt.keysym.sym == SDLK_F5)   // refresh all textures
	{
		Ogre::TextureManager::getSingleton().reloadAll();
	}
	else if (evt.keysym.sym == SDLK_PRINTSCREEN)   // take a screenshot
	{
		mWindow->writeContentsToTimestampedFile("screenshot", ".jpg");
	}
	else if (evt.keysym.sym == SDLK_ESCAPE)
	{
		mShutDown = true;
	}

	mCameraMan->keyPressed(evt);
	return true;
}

//-------------------------------------------------------------------------------------
bool BaseApplication::keyReleased(const OgreBites::KeyboardEvent& evt)
{
	mKeysDown.erase(evt.keysym.sym);
	mCameraMan->keyReleased(evt);
	return true;
}

//-------------------------------------------------------------------------------------
bool BaseApplication::mouseMoved(const OgreBites::MouseMotionEvent& evt)
{
	if (mShutDown) return true;
	if (mTrayMgr->mouseMoved(evt)) return true;
	mCameraMan->mouseMoved(evt);
	return true;
}

//-------------------------------------------------------------------------------------
bool BaseApplication::mousePressed(const OgreBites::MouseButtonEvent& evt)
{
	if (mShutDown) return true;
	if (mTrayMgr->mousePressed(evt)) return true;

	if (mDragLook && evt.button == OgreBites::BUTTON_LEFT)
	{
		mCameraMan->setStyle(OgreBites::CS_FREELOOK);
		mTrayMgr->hideCursor();
	}

	mCameraMan->mousePressed(evt);
	return true;
}

//-------------------------------------------------------------------------------------
bool BaseApplication::mouseReleased(const OgreBites::MouseButtonEvent& evt)
{
	if (mShutDown) return true;
	if (mTrayMgr->mouseReleased(evt)) return true;

	if (mDragLook && evt.button == OgreBites::BUTTON_LEFT)
	{
		mCameraMan->setStyle(OgreBites::CS_MANUAL);
		mTrayMgr->showCursor();
	}

	mCameraMan->mouseReleased(evt);
	return true;
}

//-------------------------------------------------------------------------------------
bool BaseApplication::mouseWheelRolled(const OgreBites::MouseWheelEvent& evt)
{
	mCameraMan->mouseWheelRolled(evt);
	return true;
}

//-------------------------------------------------------------------------------------
void BaseApplication::setDragLook(bool enabled)
{
	if (enabled)
	{
		mCameraMan->setStyle(OgreBites::CS_MANUAL);
		mTrayMgr->showCursor();
		mDragLook = true;
	}
	else
	{
		mCameraMan->setStyle(OgreBites::CS_FREELOOK);
		mTrayMgr->hideCursor();
		mDragLook = false;
	}
}

//-------------------------------------------------------------------------------------
bool BaseApplication::isKeyDown(OgreBites::Keycode key) const
{
	return mKeysDown.find(key) != mKeysDown.end();
}

//-------------------------------------------------------------------------------------
void BaseApplication::windowResized(Ogre::RenderWindow* rw)
{
	unsigned int width, height;
	int left, top;
	rw->getMetrics(width, height, left, top);

	// Update camera aspect ratio
	if (mCamera)
	{
		mCamera->setAspectRatio(Ogre::Real(width) / Ogre::Real(height));
	}
}

//-------------------------------------------------------------------------------------
void BaseApplication::windowClosed(Ogre::RenderWindow* rw)
{
	if (rw == mWindow)
	{
		mShutDown = true;
	}
}
