#ifndef BENCHMARK_INTERFACE_TASKRUNNER_HPP
#define BENCHMARK_INTERFACE_TASKRUNNER_HPP

#include "tx_api.h"
#include "ITask.hpp"

namespace Tasks
{
  /**
   * Runs the supplied tasks in order of submission
   */
  class TaskRunner
  {
  public:
    /**
     * Constructor
     * @param byte_pool Memory to create the task queue
     */
    TaskRunner(TX_BYTE_POOL &byte_pool);
    /**
     * Add a new task to run
     * @param task The task to run
     * @return TX_TRUE if successfully added to the queue
     */
    UCHAR Submit(ITask *task);
    /**
     * Execute the next task in the queue
     * @return TX_TRUE if the task is successfully run
     */
    UCHAR Poll();
  private:
    TX_BYTE_POOL &byte_pool;
    UCHAR ready;
    TX_QUEUE task_queue;
  };
}

#endif //BENCHMARK_INTERFACE_TASKRUNNER_HPP
