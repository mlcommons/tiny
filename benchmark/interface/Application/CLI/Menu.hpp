#ifndef CLI_MENU_HPP
#define CLI_MENU_HPP

#include "tx_api.h"
#include <string>
#include "../IO/Uart.hpp"

namespace CLI
{
  typedef struct {
    std::string command;
    void (*action)(const std::string &);
  } menu_command_t;

  class Menu
  {
  public:
    void Run();
  protected:
    Menu(TX_BYTE_POOL &byte_pool, IO::Uart &uart, const menu_command_t *commands);
    void HandleCommand(const std::string &buffer);
    void Default(const std::string &args);
    void SendString(const std::string &string);
    void SendString(const char *string);
    void SendEndLine();
    void SendEnd();
    void SendResponse(const std::string *prefix=(const std::string *)TX_NULL);
    TX_QUEUE queue;
  private:
    IO::Uart &uart;
    const menu_command_t *commands;
  };
}

#endif //CLI_MENU_HPP
