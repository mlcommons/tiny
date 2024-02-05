//
// Created by Steve Reckamp on 1/2/24.
//

#ifndef RESOURCEMANAGER_HPP
#define RESOURCEMANAGER_HPP

#include "tx_api.h"
#include "Audio/WaveSink.hpp"
#include "IO/FileSystem.hpp"
#include "Tasks/TaskRunner.hpp"
#include "Test/DeviceUnderTest.hpp"
#include "ResourceManager.h"

class ResourceManager
{
public:
  static IO::FileSystem &GetFileSystem();
  static Audio::WaveSink &GetWaveSink();
  static Tasks::TaskRunner &GetTaskRunner();
  static Test::DeviceUnderTest &GetDeviceUnderTest();
private:
  friend void ::InitializeResourceManager(TX_BYTE_POOL *);
  static TX_MUTEX singleton_mutex;
  static ResourceManager *instance;
  static ResourceManager *GetSingleton();
  static void InitSingleton(TX_BYTE_POOL &byte_pool);
  Tasks::TaskRunner *task_runner;
  Audio::WaveSink *wave_sink;
  IO::FileSystem *file_system;
  Test::DeviceUnderTest *dut;

  ResourceManager(TX_BYTE_POOL &byte_pool);
};

#endif //RESOURCEMANAGER_HPP
