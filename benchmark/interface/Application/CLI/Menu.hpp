#ifndef CLI_MENU_HPP
#define CLI_MENU_HPP

#include "tx_api.h"
#include <string>
#include "../IO/Uart.hpp"

namespace CLI
{
  typedef struct {
    std::string command;
    void (*action)(const char *);
  } menu_command_t;

  class Menu
  {
  public:
    void Run();
  protected:
    Menu(TX_BYTE_POOL &byte_pool, IO::Uart &uart, const menu_command_t *commands);
    void HandleCommand(const std::string &buffer);
    void Default(const char *args);
    void SendString(const char *string);
    void SendEndLine();
    void SendEnd();
    void SendResponse(const char *prefix=(const char *)TX_NULL);
    TX_QUEUE queue;
  private:
    IO::Uart &uart;
    const menu_command_t *commands;
  };
}

#endif //CLI_MENU_HPP
