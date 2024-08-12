//
// Created by Steve Reckamp on 1/3/24.
//

#ifndef TASKS_ITASK_HPP
#define TASKS_ITASK_HPP

#include "tx_api.h"

namespace Tasks
{
  class ITask
  {
  public:
    ITask(UCHAR self_destruct): self_destruct(self_destruct)
    {
      const char *name = "";
      tx_semaphore_create(&semaphore, (char *)name, 0);
    }
    void Execute()
    {
      Run();
      tx_semaphore_ceiling_put(&semaphore, 1);
    }
    virtual void Wait()
    {
      tx_semaphore_get(&semaphore, TX_WAIT_FOREVER);
    }
    virtual ~ITask()
    {
      tx_semaphore_delete(&semaphore);
    }
    UCHAR IsSelfDestruct() { return self_destruct; }
  protected:
    virtual void Run() = 0;
  private:
    UCHAR self_destruct;
    TX_SEMAPHORE semaphore;
  };
}

#endif //BENCHMARK_INTERFACE_ITASK_HPP
