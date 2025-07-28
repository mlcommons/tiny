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
    /**
     * The application singleton
     * @return the menu instance
     */
    static InterfaceMenu &GetSingleton();
    void RecordOneDetection();
    void DutycycleStart();
    void DutycycleStop();


    // Wrapper function declarations
    static void ListWrapper(const std::string &args);
    static void NameWrapper(const std::string &args);
    static void PassthroughWrapper(const std::string &args);
    static void PlayWrapper(const std::string &args);
    static void SetBaudWrapper(const std::string &args);
    static void CheckBaudWrapper(const std::string &args);
    static void RecDetsWrapper(const std::string &args);
    static void PrintDetsWrapper(const std::string &args);
    static void PrintDutycycleWrapper(const std::string &args);
    static void DefaultWrapper(const std::string &args);
    static bool InstanceInitialized();

    // Declarations for SetBaud and CheckBaud
    void SetBaud(const std::string &args);
    void CheckBaud(const std::string &args);

  private:
    /**
     * Allow CLI_Init access to private constructor to create the singleton.
     */
    friend void ::CLI_Init(TX_BYTE_POOL *byte_pool, UART_HandleTypeDef *huart);
    static const menu_command_t menu_struct[];
    static TX_MUTEX singleton_mutex;
    static InterfaceMenu *instance;
    static void InitSingleton(TX_BYTE_POOL &byte_pool, IO::Uart *uart, IO::FileSystem &file_system, Audio::WaveSink &player, Test::DeviceUnderTest &dut);

    InterfaceMenu(TX_BYTE_POOL &byte_pool, IO::Uart &uart, IO::FileSystem &file_system, Audio::WaveSink &player, Test::DeviceUnderTest &dut);
    void List(const std::string &args);
    void Name(const std::string &args);
    void Passthrough(const std::string &args);
    void Play(const std::string &args);
    void RecordDetections(const std::string &args);
    void PrintDetections(const std::string &args);
    void PrintDutycycle(const std::string &args);

    IO::FileSystem &file_system;
    Audio::WaveSink &player;
    Test::DeviceUnderTest &dut;
  };
}

void Record_WW_Detection();

#endif // CLI_INTERFACEMENU_HPP
