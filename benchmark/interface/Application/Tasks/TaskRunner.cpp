//
// Created by Steve Reckamp on 1/2/24.
//

#include "../ResourceManager.hpp"
#include "TaskRunner.hpp"
#include "TaskRunner.h"

#define ULONG_SIZEOF(object)            ((sizeof(object) + sizeof(ULONG) - 1)/sizeof(ULONG))
#define TASK_MESSAGE_SIZE               (ULONG_SIZEOF(ITask *) == 1 ? TX_1_ULONG :\
                                         (ULONG_SIZEOF(ITask *) == 2 ? TX_2_ULONG :\
                                          (ULONG_SIZEOF(ITask *) == 4 ? TX_4_ULONG :\
                                           (ULONG_SIZEOF(ITask *) == 8 ? TX_8_ULONG :\
                                            (ULONG_SIZEOF(ITask *) == 16 ? TX_16_ULONG :\
                                             -1)))))
#define TASK_QUEUE_SIZE                 5 * TASK_MESSAGE_SIZE

void PollTasks()
{
  ResourceManager::GetTaskRunner().Poll();
}

namespace Tasks
{
  TaskRunner::TaskRunner(TX_BYTE_POOL &byte_pool) : byte_pool(byte_pool), ready(TX_FALSE)
  {
    static char name[] = "Task Queue";
    UINT ret;
    VOID *pointer;

    ret = tx_byte_allocate(&byte_pool, &pointer, TASK_QUEUE_SIZE, TX_NO_WAIT);
    if (ret == FX_SUCCESS)
    {
      ret = tx_queue_create(&task_queue, name, TASK_MESSAGE_SIZE, pointer, TASK_QUEUE_SIZE);
      if (ret == FX_SUCCESS)
      {
        ready = TX_TRUE;
      }
    }
  }

  UCHAR TaskRunner::Submit(ITask *task)
  {
    return ready == TX_TRUE && tx_queue_send(&task_queue, &task, TX_NO_WAIT) == TX_SUCCESS
          ? TX_TRUE
          : TX_FALSE;
  }

  UCHAR TaskRunner::Poll()
  {
    ITask *task;
    if(tx_queue_receive(&task_queue, &task, 500 * TX_TIMER_TICKS_PER_SECOND / 1000) == TX_SUCCESS)
    {
      task->Execute();
      if(task->IsSelfDestruct() == TX_TRUE)
        delete task;
      return TX_TRUE;
    }
    return TX_FALSE;
  }
}
