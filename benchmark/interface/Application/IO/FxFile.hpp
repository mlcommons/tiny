//
// Created by Steve Reckamp on 12/17/23.
//

#ifndef FXFILE_HPP
#define FXFILE_HPP

#include <string>
#include "IDataSource.hpp"
#include "fx_api.h"

namespace IO
{
  class FxFile : public ::IDataSource
  {
  public:
    FxFile(FX_MEDIA *const media, const std::string &file_name) : media(media), name(file_name)
    {}

    virtual ~FxFile()
    {}

    const std::string &GetName() const { return name; }
    UCHAR Open()
    {
      char fname[name.length() + 1];
      strncpy(fname, name.c_str(), name.length() + 1);
      return fx_file_open(media, &file, fname, FX_OPEN_FOR_READ) == FX_SUCCESS ? TX_TRUE : TX_FALSE;
    }

    UCHAR Close() { return fx_file_close(&file) == FX_SUCCESS ? TX_TRUE : TX_FALSE; }

    UCHAR Seek(ULONG position) { return fx_file_seek(&file, position) == FX_SUCCESS ? TX_TRUE : TX_FALSE; }

    ULONG ReadData(void *dest, ULONG request_size)
    {
      ULONG actual_size;
      UINT result = fx_file_read(&file, dest, request_size, &actual_size);
      return result == FX_SUCCESS ? actual_size : 0;
    }

    ULONG GetPosition() const { return file.fx_file_current_file_offset; }

    operator const CHAR *() const { return name.c_str(); }

  protected:
    FX_MEDIA *const media;
    std::string name;
    FX_FILE file;
  };
}

#endif //FXFILE_HPP
