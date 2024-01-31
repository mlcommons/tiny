#ifndef TEST_DEVICEUNDERTEST_HPP
#define TEST_DEVICEUNDERTEST_HPP

#include "tx_api.h"
#include <string>
#include "../IO/Uart.hpp"
#include "../Tasks/TaskRunner.hpp"

namespace Test
{
  class SendCommandTask;
  class DeviceUnderTest
  {
  public:
    DeviceUnderTest(Tasks::TaskRunner &runner, IO::Uart *uart);
    void SendCommand(const char *command, TX_QUEUE *queue = (TX_QUEUE *) TX_NULL);
  private:
    friend class SendCommandTask;
    Tasks::TaskRunner &runner;
    IO::Uart &uart;
    void AsyncSendCommand(const char *command, TX_QUEUE *queue = (TX_QUEUE *) TX_NULL);
  };
}

#endif
