#ifndef FXFILE_HPP
#define FXFILE_HPP

#include <string>
#include "IDataSource.hpp"
#include "fx_api.h"

namespace IO
{
  /**
   * Wrpper to read from an FX_MEDIA file
   */
  class FxFile : public ::IDataSource
  {
  public:
    /**
     * Constructor
     * @param media The FX_MEDIA system
     * @param file_name The name of the file
     */
    FxFile(FX_MEDIA *const media, const std::string &file_name) : media(media), name(file_name)
    {}

    virtual ~FxFile()
    {}

    /**
     * @return The file name
     */
    const std::string &GetName() const { return name; }

    /**
     * Open a file
     * @return The results of opening the file
     */
    UCHAR Open()
    {
      char fname[name.length() + 1];
      strncpy(fname, name.c_str(), name.length() + 1);
      return fx_file_open(media, &file, fname, FX_OPEN_FOR_READ) == FX_SUCCESS ? TX_TRUE : TX_FALSE;
    }

    /**
     * Close the file
     * @return The result of closing the file
     */
    UCHAR Close() { return fx_file_close(&file) == FX_SUCCESS ? TX_TRUE : TX_FALSE; }

    /**
     * Seek to a position in the file
     * @param position The place to move the pointer to
     * @return The result of seeking in the file
     */
    UCHAR Seek(ULONG position) { return fx_file_seek(&file, position) == FX_SUCCESS ? TX_TRUE : TX_FALSE; }

    /**
     * Read data from the file and write it to a buffer
     * @param dest The buffer to hold the data
     * @param request_size The max number of bytes to read
     * @return The actual number of bytes read.
     */
    ULONG ReadData(void *dest, ULONG request_size)
    {
      ULONG actual_size;
      UINT result = fx_file_read(&file, dest, request_size, &actual_size);
      return result == FX_SUCCESS ? actual_size : 0;
    }

    /**
     * @return The current position within the file
     */
    ULONG GetPosition() const { return file.fx_file_current_file_offset; }

    /**
     * @return the name of the file if converted to a string
     */
    operator const CHAR *() const { return name.c_str(); }

  private:
    FX_MEDIA *const media;
    std::string name;
    FX_FILE file;
  };
}

#endif //FXFILE_HPP
