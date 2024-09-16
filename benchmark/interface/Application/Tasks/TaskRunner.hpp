#ifndef BENCHMARK_INTERFACE_TASKRUNNER_HPP
#define BENCHMARK_INTERFACE_TASKRUNNER_HPP

#include "tx_api.h"
#include "ITask.hpp"

namespace Tasks
{
  class TaskRunner
  {
  public:
    TaskRunner(TX_BYTE_POOL &byte_pool);
    UCHAR Submit(ITask *task);
    UCHAR Poll();
  private:
    TX_BYTE_POOL &byte_pool;
    UCHAR ready;
    TX_QUEUE task_queue;
  };
}

#endif //BENCHMARK_INTERFACE_TASKRUNNER_HPP
