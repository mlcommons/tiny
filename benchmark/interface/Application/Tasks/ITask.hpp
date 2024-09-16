#ifndef TASKS_ITASK_HPP
#define TASKS_ITASK_HPP

#include "tx_api.h"

namespace Tasks
{
  /**
   * Task interface
   */
  class ITask
  {
  public:
    /**
     * Execute the task
     */
    void Execute()
    {
      Run();
      tx_semaphore_ceiling_put(&semaphore, 1);
    }

    /**
     * Wait fpr a task to complete
     */
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
    /**
     * Constructor (Protected to prevent instantiation)
     * @param self_destruct If true, the task can be deleted when completed.
     */
    ITask(UCHAR self_destruct): self_destruct(self_destruct)
    {
      const char *name = "";
      tx_semaphore_create(&semaphore, (char *)name, 0);
    }
    /**
     * Implementation of the task
     */
    virtual void Run() = 0;
  private:
    UCHAR self_destruct;
    TX_SEMAPHORE semaphore;
  };

  /**
   * Implementation of ITask that stores an instance of an actor class to be used to execute the task
   * @tparam T
   */
  template<class T>
  class IIndirectTask: public ITask
  {
  protected:
    /**
     * Constructor (Protected to prevent instantiation)
     * @param actor The instance of the object to execute the task
     * @param self_destruct If TX_TRUE, the task will be deleted when completed.
     */
    IIndirectTask(T &actor, UCHAR self_destruct): ITask(self_destruct), actor(actor)
    {
    }
    T &actor;
  };
}

#endif //BENCHMARK_INTERFACE_ITASK_HPP
