#include "DeviceUnderTest.hpp"
#include "usart.h"
#include "../Tasks/ITask.hpp"

namespace Test
{
  class IDUTTask : public Tasks::ITask
  {
  public:
    IDUTTask(DeviceUnderTest &dut) : Tasks::ITask(TX_TRUE), dut(dut) {}
    virtual void Run() = 0;
  protected:
    DeviceUnderTest &dut;
  };

  class SendCommandTask : public IDUTTask
  {
  public:
    SendCommandTask(DeviceUnderTest &dut, const CHAR *command, TX_QUEUE *queue) :
                        IDUTTask(dut), command(command), queue(queue)
    { }

    void Run()
    {
      dut.AsyncSendCommand(command, queue);
    }

  private:
    const CHAR *command;
    TX_QUEUE *queue;
  };


  DeviceUnderTest::DeviceUnderTest(Tasks::TaskRunner &runner, IO::Uart *uart) : runner(runner), uart(*uart)
  {
  }

  void DeviceUnderTest::SendCommand(const char *command, TX_QUEUE *queue)
  {
    SendCommandTask *task = new SendCommandTask(*this, command, queue);
    runner.Submit(task);
  }

  void DeviceUnderTest::AsyncSendCommand(const char *command, TX_QUEUE *queue)
  {
    const std::string *line = (std::string *)TX_NULL;
    uart.SendString(command);
    uart.SendString("%");
    do
    {
      line = uart.ReadUntil("\r\n");
      tx_queue_send(queue, &line, TX_WAIT_FOREVER);
    }while(line == TX_NULL || line->compare("m-ready") != 0);
    VOID *tx_msg = TX_NULL;
    tx_queue_send(queue, &tx_msg, TX_WAIT_FOREVER);
  }
}
