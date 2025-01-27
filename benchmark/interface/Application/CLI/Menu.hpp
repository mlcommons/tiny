#ifndef CLI_MENU_HPP
#define CLI_MENU_HPP

#include "tx_api.h"
#include <string>
#include "../IO/Uart.hpp"

namespace CLI
{
  /***
   * Struct to define a particular command
   */
  typedef struct {
    /**
     * The string to match to identify this command
     */
    std::string command;
    /**
     * The function that is called with the buffer as an argument excluding the command and space after it
     */
    void (*action)(const std::string &);
  } menu_command_t;

  /**
   * Class defining the menu mechanics
   */
  class Menu
  {
  public:
    /**
     * Read from the host UART and process the command
     */
    void Run();
  protected:
    /**
     * A CLI Menu
     * This is protected because it should not be instantiated, it should be extended first.
     *
     * @param byte_pool The data pool to create a queue in
     * @param uart The host uart.
     * @param commands A list of structs that define the commands to run for the application
     */
    Menu(TX_BYTE_POOL &byte_pool, IO::Uart &uart, const menu_command_t *commands);

    /**
     * Process a command based on those sent to the commands argument of the Menu
     *
     * Matches the command to the commands struct, and passes the arguments to that command function.
     *
     * If no command is found Default(buffer) is executed.
     *
     * @param buffer The string received from the host
     */
    void HandleCommand(const std::string &buffer);

    /**
     * The default behavior of the menu which just returns m-ready to the host
     *
     * @param args are ignored
     */
    void Default(const std::string &args);

    /**
     * Sends the string to the host
     *
     * @param string The string to send
     */
    void SendString(const std::string &string);

    /**
     * Sends the string to the host
     *
     * @param string the const char* to send
     */
    void SendString(const char *string);

    /**
     * Sends the end line characters to the application
     */
    void SendEndLine();

    /**
     * Sends the end-of-command (m-ready\r\n) to the host
     */
    void SendEnd();

    /**
     * Reads the response from the queue of strings and sends them to the host.
     *
     * A TX_NULL response from the application will end the command.  This function will delete the string when it is
     * done sending it.
     *
     * @param prefix string sent before each line received
     */
    void SendResponse(const std::string *prefix=(const std::string *)TX_NULL);

    /**
     * Queue to send responses from the task to the host
     */
    TX_QUEUE queue;
  private:
    IO::Uart &uart;
    const menu_command_t *commands;
  };
}

#endif //CLI_MENU_HPP
