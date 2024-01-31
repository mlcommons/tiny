#include "FileSystem.hpp"
#include "FxFile.hpp"
#include "MemoryReader.hpp"
#include "../ResourceManager.hpp"
#include "../Tasks/ITask.hpp"

namespace IO
{
  class IFileTask : public Tasks::ITask
  {
  public:
    IFileTask(FileSystem &fs, UCHAR self_destruct) : Tasks::ITask(self_destruct), fs(fs) {}
    virtual void Run() = 0;
  protected:
    FileSystem &fs;
  };

  class ListDirectoryTask : public IFileTask
  {
  public:
    ListDirectoryTask(FileSystem &fs,
                      const CHAR *dir_name,
                      TX_QUEUE *queue,
                      CHAR show_directory,
                      CHAR show_hidden) :
                        IFileTask(fs, TX_TRUE), dir_name(dir_name), queue(queue),
                        show_directory(show_directory),
                        show_hidden(show_hidden)
    { }

    void Run()
    {
      fs.AsyncListDirectory(dir_name, queue, show_directory, show_hidden);
    }

  private:
    std::string dir_name;
    TX_QUEUE *queue;
    CHAR show_directory;
    CHAR show_hidden;
  };

  class OpenFileTask : public IFileTask
  {
  public:
    OpenFileTask(FileSystem &fs,
                 const CHAR *file_name) :
                      IFileTask(fs, TX_FALSE), file_name(file_name)
    { }

    void Run()
    {
      result = fs.AsyncOpenFile(file_name);
    }

    IDataSource *GetResult()
    {
      Wait();
      return result;
    }
  private:
    std::string file_name;
    IDataSource *result;
  };

  FileSystem::FileSystem(Tasks::TaskRunner &runner) : runner(runner), media((FX_MEDIA*)TX_NULL)
  {
  }

  void FileSystem::SetMedia(FX_MEDIA *media)
  {
    this->media = media;
  }

  void FileSystem::ListDirectory(const char *directory, TX_QUEUE *queue, bool show_directory, bool show_hidden)
  {
    ListDirectoryTask *task = new ListDirectoryTask(*this, directory, queue, show_directory, show_hidden);
    runner.Submit(task);
  }

  IDataSource *FileSystem::OpenFile(const char *file_name)
  {
    OpenFileTask *task = new OpenFileTask(*this, file_name);
    runner.Submit(task);
    IDataSource *result = task->GetResult();
    delete task;
    return result;
  }

  void FileSystem::AsyncListDirectory(const std::string &directory, TX_QUEUE *queue, bool show_directory, bool show_hidden)
  {
    UINT sd_status = FX_SUCCESS;

    CHAR file_name[FX_MAX_LONG_NAME_LEN];
    UINT attributes, year, month, day, hour, minute, second;
    ULONG size;
    for (sd_status = fx_directory_first_full_entry_find(media, file_name, &attributes, &size,
                                                        &year, &month, &day, &hour, &minute, &second);
         sd_status == FX_SUCCESS;
         sd_status = fx_directory_next_full_entry_find(media, file_name, &attributes, &size,
                                                       &year, &month, &day, &hour, &minute, &second))
    {
      if((show_hidden || (attributes & FX_HIDDEN) == 0)
         && (show_directory || (attributes & FX_DIRECTORY) == 0)
         && (attributes & FX_SYSTEM) == 0)
      {
        if(queue != TX_NULL)
        {
          std::string *file = new std::string(file_name);
          if((attributes & FX_DIRECTORY) != 0)
          {
            file->append("/");
          }
          tx_queue_send(queue, &file, TX_WAIT_FOREVER);
        }
      }
    }
    if(queue != TX_NULL)
    {
      VOID *tx_msg = TX_NULL;
      tx_queue_send(queue, &tx_msg, TX_WAIT_FOREVER);
    }
  }

  IDataSource *FileSystem::AsyncOpenFile(std::string file_name)
  {
    return file_name.length() > 0
              ? (IDataSource *) new IO::FxFile(media, file_name)
              : (IDataSource *) new IO::MemoryReader(0x08080000, (180 * 1024));
  }
}
