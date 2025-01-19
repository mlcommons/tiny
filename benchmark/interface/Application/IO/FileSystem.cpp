#include "FileSystem.hpp"
#include "FxFile.hpp"
#include "MemoryReader.hpp"
#include "../ResourceManager.hpp"
#include "../Tasks/ITask.hpp"

namespace IO
{
  /**
   * List the contents of a directory
   */
  class ListDirectoryTask : public Tasks::IIndirectTask<FileSystem>
  {
  public:
    /**
     * Constructor
     * @param fs the file system to operate on
     * @param dir_name Directory path to list the contents of
     * @param queue The queue to send the response to
     * @param show_directory Show directories in the output (default TX_FALSE)
     * @param show_hidden Show hidden files in the output (default TX_FALSE)
     */
    ListDirectoryTask(FileSystem &fs,
                      const std::string &dir_name,
                      TX_QUEUE *queue,
                      CHAR show_directory,
                      CHAR show_hidden) :
                    	  Tasks::IIndirectTask<FileSystem>(fs, TX_TRUE),
						  dir_name(dir_name),
						  queue(queue),
						  show_directory(show_directory),
						  show_hidden(show_hidden)
    { }

    void Run()
    {
      actor.IndirectListDirectory(dir_name, queue, show_directory, show_hidden);
    }

  private:
    std::string dir_name;
    TX_QUEUE *queue;
    CHAR show_directory;
    CHAR show_hidden;
  };

  FileSystem::FileSystem(Tasks::TaskRunner &runner) : runner(runner), media((FX_MEDIA*)TX_NULL)
  {
  }

  void FileSystem::SetMedia(FX_MEDIA *media)
  {
    this->media = media;
  }

  void FileSystem::ListDirectory(const std::string &directory, TX_QUEUE *queue, bool show_directory, bool show_hidden)
  {
    ListDirectoryTask *task = new ListDirectoryTask(*this, directory, queue, show_directory, show_hidden);
    runner.Submit(task);
  }

  IDataSource *FileSystem::OpenFile(const std::string &file_name)
  {
    return file_name.length() > 0
           ? (IDataSource *) new IO::FxFile(media, file_name)
           : (IDataSource *) new IO::MemoryReader(0x08080000, (180 * 1024));
  }

  /**
   * Enumerate the directory and send the results line by line to the queue
   *
   * Sends a TX_NULL when the list is complete.
   *
   * @param directory Directory path to list the contents of
   * @param queue The queue to send the response to
   * @param show_directory Show directories in the output (default TX_FALSE)
   * @param show_hidden Show hidden files in the output (default TX_FALSE)
   */
  void FileSystem::IndirectListDirectory(const std::string &directory, TX_QUEUE *queue, bool show_directory, bool show_hidden)
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
}
