//
// Created by Steve Reckamp on 12/15/23.
//

#ifndef FILESYSTEM_HPP
#define FILESYSTEM_HPP

#include "fx_api.h"
#include <string>
#include "../IDataSource.hpp"
#include "../Tasks/TaskRunner.hpp"

namespace IO
{
  class ListDirectoryTask;
  class OpenFileTask;

  class FileSystem
  {
  public:
    FileSystem(Tasks::TaskRunner &runner);
    void SetMedia(FX_MEDIA *media);
    void ListDirectory(const std::string &directory, TX_QUEUE *queue = (TX_QUEUE *) TX_NULL, bool show_directory = TX_FALSE,
                       bool show_hidden = TX_FALSE);
    IDataSource *OpenFile(const std::string &file_name);
  private:
    Tasks::TaskRunner &runner;
    FX_MEDIA *media;
    void AsyncListDirectory(const std::string &directory, TX_QUEUE *queue, bool show_directory, bool show_hidden);
    IDataSource *AsyncOpenFile(const std::string &file_name);
    friend class ListDirectoryTask;
    friend class OpenFileTask;
  };
}

#endif //FILESYSTEM_HPP
