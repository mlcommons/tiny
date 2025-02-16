#include <cstring>
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


  DeviceUnderTest::DeviceUnderTest(Tasks::TaskRunner &runner, IO::Uart *uart) : runner(runner), uart(*uart), wwdet_timestamp_idx(0)
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

  void DeviceUnderTest::RecordDetection()
  {
	  uint32_t current_time_us = __HAL_TIM_GET_COUNTER(&htim16);

	  // don't record two detections with the same timestamp and
	  // don't overrun the end of the allocated array
	  if( wwdet_timestamps[wwdet_timestamp_idx-1] != current_time_us &&
	      wwdet_timestamp_idx < MAX_TIMESTAMPS )
	  {
		  wwdet_timestamps[wwdet_timestamp_idx++] = current_time_us;
	  }
  }
  uint32_t *DeviceUnderTest::GetDetections()
  {
	  return wwdet_timestamps;
  }

  uint32_t DeviceUnderTest::GetNumDetections()
  {
  	  return wwdet_timestamp_idx;
  }

  void DeviceUnderTest::ClearDetections()
  {
	std::memset(wwdet_timestamps, 0, sizeof(wwdet_timestamps));
	wwdet_timestamp_idx = 0;
  }

  void DeviceUnderTest::StartRecordingDetections()
  {
	ClearDetections();
	dut_state=RecordingDetections;
  }

}

