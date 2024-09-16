#ifndef TASKS_TASK_RUNNER_H
#define TASKS_TASK_RUNNER_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Runs the next task in the queue
 *
 * Returns when the task completes
 */
void PollTasks();

#ifdef __cplusplus
}
#endif

#endif
