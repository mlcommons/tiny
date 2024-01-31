#include "WaveSink.hpp"
#include "../Tasks/ITask.hpp"

#define PLAY_BUFFER_BYTES   8 * 1024

namespace Audio
{
  class PlayWaveTask: public Tasks::ITask
  {
  public:
    PlayWaveTask(WaveSink &player, WaveSource &source):
        ITask(TX_FALSE), player(player), source(source)
        { }

    void Run()
    {
      result = player.AsyncPlay(source);
    }

    PlayerResult GetResult()
    {
      Wait();
      return result;
    }
  private:
    WaveSink &player;
    WaveSource &source;
    PlayerResult result;
  };

  WaveSink::WaveSink(Tasks::TaskRunner &runner, TX_BYTE_POOL &byte_pool): runner(runner), size(PLAY_BUFFER_BYTES)
  {
    tx_byte_allocate(&byte_pool, (void **)&play_buffer, size, TX_NO_WAIT);
  }

  PlayerResult WaveSink::Play(WaveSource &source)
  {
    PlayWaveTask *task = new PlayWaveTask(*this, source);
    runner.Submit(task);
    PlayerResult result = task->GetResult();
    delete task;
    return result;
  }

  PlayerResult WaveSink::AsyncPlay(WaveSource &source)
  {
    PlayerState state = GetState();
    if(state == RESET)
    {
      state = Initialize();
    }
    if(state != STOPPED)
    {
      return ERROR;
    }

    if(source.Open() == TX_TRUE)
    {
      Configure(source);
      source.Seek(0);

      INT active_buffer = 0;
      ULONG next_bytes = source.ReadData(play_buffer, size);
      SetActiveBuffer(-1);

      PlayerResult status = Play((UCHAR *)play_buffer, size);

      while(status == SUCCESS && next_bytes > 0)
      {
        active_buffer = WaitForActiveBuffer();

        next_bytes = source.ReadData(&play_buffer[active_buffer * size/2], size / 2);

        SetActiveBuffer(-1);
      }
      Stop();
      source.Close();
    }

    return SUCCESS;
  }
}
