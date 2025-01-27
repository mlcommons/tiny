#include "Audio/HeadphoneWaveSink.hpp"
#include "Audio/RawI2SWaveSink.hpp"
#include "ResourceManager.hpp"
#include "usart.h"

void InitializeResourceManager(TX_BYTE_POOL *byte_pool)
{
  static const char *name = "ResourceManager mutex";
  tx_mutex_create(&ResourceManager::singleton_mutex, (char *)name, 0);
  ResourceManager::InitSingleton(*byte_pool);
}

void SetFxMedia(FX_MEDIA *media)
{
  ResourceManager::GetFileSystem().SetMedia(media);
}

ResourceManager *ResourceManager::instance = (ResourceManager *)TX_NULL;
TX_MUTEX ResourceManager::singleton_mutex = { 0 };

ResourceManager *ResourceManager::GetSingleton()
{
  return instance;
};

IO::FileSystem &ResourceManager::GetFileSystem()
{
  return *GetSingleton()->file_system;
};

Audio::WaveSink &ResourceManager::GetWaveSink()
{
  return *GetSingleton()->wave_sink;
};

Tasks::TaskRunner &ResourceManager::GetTaskRunner()
{
  return *GetSingleton()->task_runner;
};

Test::DeviceUnderTest &ResourceManager::GetDeviceUnderTest()
{
  return *GetSingleton()->dut;
};

void ResourceManager::InitSingleton(TX_BYTE_POOL &byte_pool)
{
  tx_mutex_get(&singleton_mutex, TX_WAIT_FOREVER);
  if(instance == TX_NULL)
  {
    instance = new ResourceManager(byte_pool);
  }
  tx_mutex_put(&singleton_mutex);
}

ResourceManager::ResourceManager(TX_BYTE_POOL &byte_pool): task_runner(new Tasks::TaskRunner(byte_pool)),
#ifdef HEADPHONE_PLAYBACK
                                                           wave_sink(new Audio::HeadphoneWaveSink(*task_runner, byte_pool)),
#else
                                                           wave_sink(new Audio::RawI2SWaveSink(*task_runner, byte_pool)),
#endif
                                                           file_system(new IO::FileSystem(*task_runner)),
                                                           dut(new Test::DeviceUnderTest(*task_runner, new IO::Uart(byte_pool, &huart3)))
{
}
