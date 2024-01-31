#ifndef CLI_INTERFACEMENU_HPP
#define CLI_INTERFACEMENU_HPP

#include "Menu.hpp"
#include "InterfaceMenu.h"
#include "../IO/FileSystem.hpp"
#include "../Audio/WaveSink.hpp"
#include "../Test/DeviceUnderTest.hpp"

struct menu_command_s;

namespace CLI
{
  class InterfaceMenu: public Menu
  {
  public:
    static InterfaceMenu &GetSingleton();
  private:
    friend void ::CLI_Init(TX_BYTE_POOL *byte_pool, UART_HandleTypeDef *huart);
    static const menu_command_t menu_struct[];
    static TX_MUTEX singleton_mutex;
    static InterfaceMenu *instance;
    static void InitSingleton(TX_BYTE_POOL &byte_pool, IO::Uart *uart, IO::FileSystem &file_system, Audio::WaveSink &player, Test::DeviceUnderTest &dut);
    static void ListWrapper(const char *args);
    static void NameWrapper(const char *args);
    static void PassthroughWrapper(const char *args);
    static void PlayWrapper(const char *args);
    static void DefaultWrapper(const char *args);
    InterfaceMenu(TX_BYTE_POOL &byte_pool, IO::Uart &uart, IO::FileSystem &file_system, Audio::WaveSink &player, Test::DeviceUnderTest &dut);
    void List(const char *args);
    void Name(const char *args);
    void Passthrough(const char *args);
    void Play(const char *args);
    IO::FileSystem &file_system;
    Audio::WaveSink &player;
    Test::DeviceUnderTest &dut;
  };
}

#endif //CLI_INTERFACEMENU_HPP
