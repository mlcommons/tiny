#ifndef IO_UART_HPP
#define IO_UART_HPP

#include "tx_api.h"
#include "usart.h"
#include <string>

namespace IO
{
  class Uart
  {
  public:
    Uart(TX_BYTE_POOL &byte_pool, UART_HandleTypeDef *huart);
    ~Uart();
    void SendString(std::string command);
    const std::string *ReadUntil(const std::string &end);
  private:
    UART_HandleTypeDef *handle;
    TX_QUEUE queue;
    uint8_t rx_char;
  };
}

#endif //BENCHMARK_INTERFACE_UART_HPP
