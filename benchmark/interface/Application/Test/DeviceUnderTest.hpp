#ifndef TEST_DEVICEUNDERTEST_HPP
#define TEST_DEVICEUNDERTEST_HPP

#include "tx_api.h"
#include <string>
#include <stdint.h>
#include "../IO/Uart.hpp"
#include "../Tasks/TaskRunner.hpp"
#include "stm32h5xx_hal_tim.h"
#include "tim.h"

#define MAX_TIMESTAMPS 10000

namespace Test
{
  typedef enum dut_state_enum {
	  Idle,
	  RecordingDetections,
  } dut_state_t;
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
	void RecordDetection();
	void ClearDetections();
	void StartRecordingDetections();
	uint32_t *GetDetections();
	uint32_t GetNumDetections();

  private:
    friend class SendCommandTask;
    Tasks::TaskRunner &runner;
    IO::Uart &uart;
    dut_state_t dut_state=Idle;
    void IndirectSendCommand(const std::string &command, TX_QUEUE *queue = (TX_QUEUE *) TX_NULL);
	uint32_t wwdet_timestamps[MAX_TIMESTAMPS] = {0};
	uint32_t wwdet_timestamp_idx = 0;
  };
}

#endif
