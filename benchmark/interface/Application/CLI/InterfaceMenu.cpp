#include "InterfaceMenu.hpp"
#include "../IDataSource.hpp"
#include "../ResourceManager.hpp"

void CLI_Init(TX_BYTE_POOL *byte_pool, UART_HandleTypeDef *huart)
{
  static const char *name = "InterfaceMenu mutex";
  tx_mutex_create(&CLI::InterfaceMenu::singleton_mutex, (char *)name, 0);
  CLI::InterfaceMenu::InitSingleton(*byte_pool, new IO::Uart(*byte_pool, huart), ResourceManager::GetFileSystem(), ResourceManager::GetWaveSink(), ResourceManager::GetDeviceUnderTest());
}

void CLI_Run()
{
  CLI::InterfaceMenu::GetSingleton().Run();
}

namespace CLI
{
  InterfaceMenu *InterfaceMenu::instance = (InterfaceMenu *)TX_NULL;
  TX_MUTEX InterfaceMenu::singleton_mutex = { 0 };
  const menu_command_t InterfaceMenu::menu_struct[] = { {"dut", PassthroughWrapper},
                                                        {"name", NameWrapper},
                                                        {"ls", ListWrapper},
                                                        {"play", PlayWrapper},
                                                        {"", DefaultWrapper} };

  void InterfaceMenu::ListWrapper(const std::string &args)
  {
    instance->List(args);
  }

  void InterfaceMenu::NameWrapper(const std::string &args)
  {
    instance->Name(args);
  }

  void InterfaceMenu::PlayWrapper(const std::string &args)
  {
    instance->Play(args);
  }

  void InterfaceMenu::DefaultWrapper(const std::string &args)
  {
    instance->Default(args);
  }

  void InterfaceMenu::PassthroughWrapper(const std::string &args)
  {
    instance->Passthrough(args);
  }

  void InterfaceMenu::InitSingleton(TX_BYTE_POOL &byte_pool, IO::Uart *uart, IO::FileSystem &file_system, Audio::WaveSink &player, Test::DeviceUnderTest &dut)
  {
    tx_mutex_get(&singleton_mutex, TX_WAIT_FOREVER);
    if(instance == TX_NULL)
    {
      instance = new InterfaceMenu(byte_pool, *uart, file_system, player, dut);
    }
    tx_mutex_put(&singleton_mutex);
  }

  InterfaceMenu &InterfaceMenu::GetSingleton()
  {
    return *instance;
  }

  InterfaceMenu::InterfaceMenu(TX_BYTE_POOL &byte_pool, IO::Uart &uart, IO::FileSystem &file_system, Audio::WaveSink &player, Test::DeviceUnderTest &dut):
            Menu(byte_pool, uart, menu_struct),
            file_system(file_system),
            player(player),
            dut(dut)
  {

  }

  void InterfaceMenu::List(const std::string &args)
  {
    file_system.ListDirectory(args, &queue);
    SendResponse();
    SendEnd();
  }

  void InterfaceMenu::Name(const std::string &args)
  {
    SendString("tinyML Enhanced Interface Board");
    SendEndLine();
    SendEnd();
  }

  void InterfaceMenu::Passthrough(const std::string &args)
  {
    static const std::string prefix("[dut]: ");
    SendString("m-dut-passthrough(");
    SendString(args);
    SendString(")");
    SendEndLine();
    SendEnd();
    dut.SendCommand(args, &queue);
    SendResponse(&prefix);
  }

  void InterfaceMenu::Play(const std::string &args)
  {
    IDataSource *source = file_system.OpenFile(args);
    Audio::WaveSource wav(*source);
    std::string *msg = new std::string("Playing ");
    msg->append(source->GetName());
    msg->append("\n");
    SendString(msg->c_str());
    player.Play(wav);
    SendEnd();
    delete source;
  }
}
