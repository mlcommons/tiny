#ifndef TEST_DEVICEUNDERTEST_HPP
#define TEST_DEVICEUNDERTEST_HPP

#include "tx_api.h"
#include <string>
#include "../IO/Uart.hpp"
#include "../Tasks/TaskRunner.hpp"

namespace Test
{
  class SendCommandTask;

  /**
   * Device Under Test Abstraction
   */
  class DeviceUnderTest
  {
  public:
    /**
     * Constructor
     * @param runner TaskRunner that executes the submitted Task::ITasks
     * @param uart The UART connected to the DUT
     */
    DeviceUnderTest(Tasks::TaskRunner &runner, IO::Uart *uart);
    /**
     * Submit a Task::ITask to send a command to DUT and then forward the response to a queue
     * @param command The command to send
     * @param queue The queue to send the response to
     */
    void SendCommand(const std::string &command, TX_QUEUE *queue = (TX_QUEUE *) TX_NULL);
  private:
    friend class SendCommandTask;
    Tasks::TaskRunner &runner;
    IO::Uart &uart;
    void IndirectSendCommand(const std::string &command, TX_QUEUE *queue = (TX_QUEUE *) TX_NULL);
  };
}

#endif
