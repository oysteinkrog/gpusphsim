/*
-----------------------------------------------------------------------------
Filename:    BaseApplication.h
-----------------------------------------------------------------------------
Updated for Ogre 14.x - Uses ApplicationContext with SDL2 input
-----------------------------------------------------------------------------
*/
#ifndef __BaseApplication_h_
#define __BaseApplication_h_

#include <Ogre.h>
#include <OgreApplicationContext.h>
#include <OgreInput.h>
#include <OgreTrays.h>
#include <OgreCameraMan.h>
#include <OgreRTShaderSystem.h>

#include "OgreSimConfig.h"

class BaseApplication : public OgreBites::ApplicationContext,
                        public OgreBites::InputListener,
                        public OgreBites::TrayListener
{
public:
	BaseApplication(void);
	virtual ~BaseApplication(void);

	void go(void);

protected:
	// ApplicationContext overrides
	void setup() override;
	void shutdown() override;
	bool frameRenderingQueued(const Ogre::FrameEvent& evt) override;
	bool frameStarted(const Ogre::FrameEvent& evt) override;
	bool frameEnded(const Ogre::FrameEvent& evt) override;

	// Setup methods
	virtual void chooseSceneManager(void);
	virtual void createCamera(void);
	virtual void createScene(void) = 0; // Override me!
	virtual void destroyScene(void);
	virtual void createViewports(void);
	virtual void setupResources(void);
	virtual void loadResources(void);
	virtual void createFrameListener(void);

	virtual void setDragLook(bool enabled);

	// InputListener overrides (SDL2-based input)
	bool keyPressed(const OgreBites::KeyboardEvent& evt) override;
	bool keyReleased(const OgreBites::KeyboardEvent& evt) override;
	bool mouseMoved(const OgreBites::MouseMotionEvent& evt) override;
	bool mousePressed(const OgreBites::MouseButtonEvent& evt) override;
	bool mouseReleased(const OgreBites::MouseButtonEvent& evt) override;
	bool mouseWheelRolled(const OgreBites::MouseWheelEvent& evt) override;

	// Window events
	virtual void windowResized(Ogre::RenderWindow* rw);
	virtual void windowClosed(Ogre::RenderWindow* rw);

	Ogre::Root* mRoot;
	Ogre::Camera* mCamera;
	Ogre::SceneManager* mSceneMgr;
	Ogre::RenderWindow* mWindow;

	// OgreBites
	OgreBites::TrayManager* mTrayMgr;
	OgreBites::CameraMan* mCameraMan;       // basic camera controller
	OgreBites::ParamsPanel* mDetailsPanel;  // sample details panel
	bool mCursorWasVisible;                 // was cursor visible before dialog appeared
	bool mShutDown;
	bool mDragLook;

	OgreSim::Config* mSnowConfig;

	// Helper to check if a key is currently pressed
	bool isKeyDown(OgreBites::Keycode key) const;

private:
	std::set<OgreBites::Keycode> mKeysDown;
};

#endif // #ifndef __BaseApplication_h_
