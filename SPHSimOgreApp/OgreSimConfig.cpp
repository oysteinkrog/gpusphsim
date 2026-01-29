// Updated for Ogre 14.x - Uses range-based iteration for ConfigFile

#include "OgreSimConfig.h"

using namespace Ogre;

namespace OgreSim {

	Config::Config(const std::string& configFileName)
		: mCfg(NULL)
	{
		Ogre::LogManager::getSingleton().logMessage("*** Loading OgreSnow configuration ***");

		mCfg = new Ogre::ConfigFile();
		// Load directly from filesystem since resource groups aren't set up yet
		mCfg->load(configFileName);

		loadConfig();
	}

	Config::~Config()
	{
		delete mCfg;
	}

	void Config::loadConfig()
	{
		// defaults
		generalSettings.cudadevice = 0;
		generalSettings.logLevel = LoggingLevel::LL_LOW;
		generalSettings.showOgreConfigDialog = true;
		generalSettings.showOgreGui = true;

		// Ogre 14.x: Use getSettingsBySection() with range-based iteration
		auto sections = mCfg->getSettingsBySection();
		auto generalIt = sections.find("General");
		if (generalIt != sections.end())
		{
			for (const auto& setting : generalIt->second)
			{
				String name = setting.first;
				Ogre::StringUtil::toLowerCase(name);
				const String& value = setting.second;

				if(name == "cudadevice")
				{
					generalSettings.cudadevice = StringConverter::parseUnsignedInt(value);
				}
				else if(name == "loglevel")
				{
					generalSettings.logLevel = (Ogre::LoggingLevel)StringConverter::parseUnsignedInt(value);
				}
				else if(name == "showogreconfigdialog")
				{
					generalSettings.showOgreConfigDialog = StringConverter::parseBool(value);
				}
				else if(name == "showogregui")
				{
					generalSettings.showOgreGui = StringConverter::parseBool(value);
				}
				else if(name == "fluidshader")
				{
					generalSettings.fluidShader = value;
				}
			}
		}

		loadSceneConfig();
		loadFluidConfig();
		loadTerrainConfig();
	}

	void Config::loadSceneConfig()
	{
		// defaults
		sceneSettings.cameraPosition = Ogre::Vector3(0,0,0);
		sceneSettings.fluidPosition = Ogre::Vector3(0,0,0);
		sceneSettings.terrainPosition = Ogre::Vector3(0,0,0);
		sceneSettings.cameraOrientation = Ogre::Quaternion(0,0,0,0);
		sceneSettings.fluidScene = 9;

		auto sections = mCfg->getSettingsBySection();
		auto sceneIt = sections.find("Scene");
		if (sceneIt != sections.end())
		{
			for (const auto& setting : sceneIt->second)
			{
				String name = setting.first;
				Ogre::StringUtil::toLowerCase(name);
				const String& value = setting.second;

				if(name == "skyboxmaterial")
				{
					sceneSettings.skyBoxMaterial = value;
				}
				else if(name == "cameratelativetofluid")
				{
					sceneSettings.cameraRelativeToFluid = StringConverter::parseBool(value);
				}
				else if(name == "cameraposition")
				{
					sceneSettings.cameraPosition = StringConverter::parseVector3(value);
				}
				else if(name == "cameraorientation")
				{
					sceneSettings.cameraOrientation = StringConverter::parseQuaternion(value);
				}
				else if(name == "fluidposition")
				{
					sceneSettings.fluidPosition = StringConverter::parseVector3(value);
				}
				else if(name == "terrainposition")
				{
					sceneSettings.terrainPosition = StringConverter::parseVector3(value);
				}
				else if(name == "backgroundcolor")
				{
					sceneSettings.backgroundColor = StringConverter::parseColourValue(value);
				}
				else if(name == "fluidgridcolor")
				{
					sceneSettings.fluidGridColor = StringConverter::parseColourValue(value);
				}
				else if(name == "fluidscene")
				{
					sceneSettings.fluidScene = StringConverter::parseUnsignedInt(value);
				}
			}
		}
	}

	void Config::loadFluidConfig()
	{
		// defaults
		fluidSettings.simpleSPH = true;
		fluidSettings.enabled = false;
		fluidSettings.showFluidGrid = true;

		auto sections = mCfg->getSettingsBySection();
		auto fluidIt = sections.find("Fluid");
		if (fluidIt != sections.end())
		{
			for (const auto& setting : fluidIt->second)
			{
				String name = setting.first;
				Ogre::StringUtil::toLowerCase(name);
				const String& value = setting.second;

				if(name == "simplesph")
				{
					fluidSettings.simpleSPH = StringConverter::parseBool(value);
				}
				if(name == "enabled")
				{
					fluidSettings.enabled = StringConverter::parseBool(value);
				}
				if(name == "enablekerneltiming")
				{
					fluidSettings.enableKernelTiming = StringConverter::parseBool(value);
				}
				else if(name == "showfluidgrid")
				{
					fluidSettings.showFluidGrid = StringConverter::parseBool(value);
				}
				else if(name == "gridwallcollisions")
				{
					fluidSettings.gridWallCollisions = StringConverter::parseBool(value);
				}
				else if(name == "terraincollisions")
				{
					fluidSettings.terrainCollisions = StringConverter::parseBool(value);
				}
			}
		}
	}

	void Config::loadTerrainConfig()
	{
		// defaults
		terrainSettings.enabled = false;
		terrainSettings.showDebugNormals = false;
		terrainSettings.flat = true;
		terrainSettings.worldSize = 2250.0f;
		terrainSettings.worldScale = 376.0f;
		terrainSettings.size = 4097;

		auto sections = mCfg->getSettingsBySection();
		auto terrainIt = sections.find("Terrain");
		if (terrainIt != sections.end())
		{
			for (const auto& setting : terrainIt->second)
			{
				String name = setting.first;
				Ogre::StringUtil::toLowerCase(name);
				const String& value = setting.second;

				if(name == "enabled")
				{
					terrainSettings.enabled = StringConverter::parseBool(value);
				}
				else if(name == "showdebugnormals")
				{
					terrainSettings.showDebugNormals = StringConverter::parseBool(value);
				}
				else if(name == "flat")
				{
					terrainSettings.flat = StringConverter::parseBool(value);
				}
				else if(name == "size")
				{
					terrainSettings.size = StringConverter::parseUnsignedInt(value);
				}
				else if(name == "worldsize")
				{
					terrainSettings.worldSize = StringConverter::parseReal(value);
				}
				else if(name == "worldscale")
				{
					terrainSettings.worldScale = StringConverter::parseReal(value);
				}
				else if(name == "normalsdatafile")
				{
					terrainSettings.normalsDataFile = value;
				}
				else if(name == "heightdatafile")
				{
					terrainSettings.heightDataFile = value;
				}
			}
		}

		Ogre::String setting;

		for(int i = 0; i< 10; i++)
		{
			setting = "textureBlendFile"+Ogre::StringConverter::toString(i);
			Ogre::String textureBlendFile = mCfg->getSetting(setting, "Terrain");
			terrainSettings.textureBlendFileList.push_back(textureBlendFile);

			setting = "textureLayerNormalHeightFile"+Ogre::StringConverter::toString(i);
			Ogre::String textureLayerNormalHeightFile = mCfg->getSetting(setting, "Terrain");

			setting = "textureLayerDiffSpecFile"+Ogre::StringConverter::toString(i);
			Ogre::String textureLayerDiffSpecFile = mCfg->getSetting(setting, "Terrain");

			setting = "textureLayerSize"+Ogre::StringConverter::toString(i);
			Ogre::Real textureLayerSize = StringConverter::parseReal(mCfg->getSetting(setting, "Terrain"));

			if(textureLayerNormalHeightFile.length() > 0 || textureLayerDiffSpecFile.length() > 0 || textureLayerSize > 0)
			{
				terrainSettings.textureLayerNormalHeightFileList.resize(i+1);
				terrainSettings.textureLayerDiffSpecFileList.resize(i+1);
				terrainSettings.textureLayerSize.resize(i+1);

				if(textureLayerSize == 0)
					textureLayerSize = terrainSettings.worldSize;

				terrainSettings.textureLayerNormalHeightFileList[i] = textureLayerNormalHeightFile;
				terrainSettings.textureLayerDiffSpecFileList[i] = textureLayerDiffSpecFile;
				terrainSettings.textureLayerSize[i] = textureLayerSize;
			}

		}
	}
}
