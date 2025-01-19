#ifndef IO_UART_HPP
#define IO_UART_HPP

#include "tx_api.h"
#include "usart.h"
#include <string>

namespace IO
{
  /**
   * Wrapper around a UART
   */
  class Uart
  {
  public:
    /**
     * Constructor
     * @param byte_pool The memory pool to use
     * @param huart hardware UART handle
     */
    Uart(TX_BYTE_POOL &byte_pool, UART_HandleTypeDef *huart);
    ~Uart();

    /**
     * Send a string to the uart
     * @param text the text to send to the uart
     */
    void SendString(std::string text);

    /**
     * Read a terminated string
     * @param end The string the signifies the end
     * @return The received string (without the terminator)
     */
    const std::string *ReadUntil(const std::string &end);
  private:
    UART_HandleTypeDef *handle;
    TX_QUEUE queue;
    uint8_t rx_char;
  };
}

#endif //BENCHMARK_INTERFACE_UART_HPP
