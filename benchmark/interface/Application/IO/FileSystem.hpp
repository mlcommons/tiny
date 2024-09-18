#ifndef FILESYSTEM_HPP
#define FILESYSTEM_HPP

#include "fx_api.h"
#include <string>
#include "../IDataSource.hpp"
#include "../Tasks/TaskRunner.hpp"

namespace IO
{
  class ListDirectoryTask;

  /**
   * File System Abstraction
   */
  class FileSystem
  {
  public:
    /**
     * Constructor
     * @param runner TaskRunner that executes the submitted Task::ITasks
     */
    FileSystem(Tasks::TaskRunner &runner);

    /**
     * Connect the FileSystem to an FX_MEDIA instance
     * @param media the FX_MEDIA instance in it
     */
    void SetMedia(FX_MEDIA *media);

    /**
     * Submit a Task::ITask to list the contents of the directory
     * @param directory Directory path to list the contents of
     * @param queue The queue to send the response to
     * @param show_directory Show directories in the output (default TX_FALSE)
     * @param show_hidden Show hidden files in the output (default TX_FALSE)
     */
    void ListDirectory(const std::string &directory, TX_QUEUE *queue = (TX_QUEUE *) TX_NULL, bool show_directory = TX_FALSE,
                       bool show_hidden = TX_FALSE);

    /**
     * Open a file
     *
     * @param file_name The name of the file. If filename is empty. it plays back the memory at 0x08080000
     * @return The file as an ::IDataSource
     */
    IDataSource *OpenFile(const std::string &file_name);
  private:
    Tasks::TaskRunner &runner;
    FX_MEDIA *media;
    void IndirectListDirectory(const std::string &directory, TX_QUEUE *queue, bool show_directory, bool show_hidden);
    friend class ListDirectoryTask;
  };
}

#endif //FILESYSTEM_HPP
