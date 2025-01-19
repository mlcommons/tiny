#include "DeviceUnderTest.hpp"
#include "usart.h"
#include "../Tasks/ITask.hpp"

namespace Test
{
  /**
   * Send a command to the DUT
   */
class SendCommandTask : public Tasks::IIndirectTask<DeviceUnderTest>
  {
  public:
    /**
     * Constructor
     * @param dut The dut to execute the task on
     * @param command The command to send
     * @param queue The queue to send the response to
     */
    SendCommandTask(DeviceUnderTest &dut, const std::string &command, TX_QUEUE *queue) :
                    IIndirectTask(dut, TX_TRUE), command(command), queue(queue)
    { }

    void Run()
    {
      actor.IndirectSendCommand(command, queue);
    }

  private:
    const std::string command;
    TX_QUEUE *queue;
  };


  DeviceUnderTest::DeviceUnderTest(Tasks::TaskRunner &runner, IO::Uart *uart) : runner(runner), uart(*uart)
  {
  }

  void DeviceUnderTest::SendCommand(const std::string &command, TX_QUEUE *queue)
  {
    SendCommandTask *task = new SendCommandTask(*this, command, queue);
    runner.Submit(task);
  }

  /**
   * Send a command to DUT and then forward the response to a queue
   * @param command command to send
   * @param queue The queue to send the response to
   */
  void DeviceUnderTest::IndirectSendCommand(const std::string &command, TX_QUEUE *queue)
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
