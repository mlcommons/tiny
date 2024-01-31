#include "usart.h"
#include "Uart.hpp"

#define RX_QUEUE_SIZE   50 * sizeof(ULONG)

#define UART_COUNT  4

static void ISR_UART(UART_HandleTypeDef *huart);
static uint8_t uart_idx;
static UART_HandleTypeDef *handles[UART_COUNT] = {(UART_HandleTypeDef *)TX_NULL};
static TX_QUEUE *queues[UART_COUNT] = {(TX_QUEUE *)TX_NULL};
static uint8_t *rx_chars[UART_COUNT] = {(uint8_t *)TX_NULL};

namespace IO
{
  Uart::Uart(TX_BYTE_POOL &byte_pool, UART_HandleTypeDef *huart): handle(huart)
  {
    static char name[] = "UART receive queue";
    VOID *qpointer;
    tx_byte_allocate(&byte_pool, (VOID**) &qpointer, RX_QUEUE_SIZE, TX_NO_WAIT);
    tx_queue_create(&queue, name, TX_1_ULONG, qpointer, RX_QUEUE_SIZE);
    uint8_t id = uart_idx++;
    handles[id] = huart;
    queues[id] = &queue;
    rx_chars[id] = &rx_char;
    HAL_UART_RegisterCallback(handle, HAL_UART_RX_COMPLETE_CB_ID, ISR_UART);
    HAL_UART_Receive_IT(handle, &rx_char, 1);
  }

  Uart::~Uart()
  {
    HAL_UART_UnRegisterCallback(handle, HAL_UART_RX_COMPLETE_CB_ID);
  }

  void Uart::SendString(std::string command)
  {
    UART_SendString(handle, command.c_str());
  }

  const std::string *Uart::ReadUntil(const std::string &end)
  {
    std::string *message = new std::string();
    std::string eol;
    while(1)
    {
      uint32_t rx_msg;
      tx_queue_receive(&queue, &rx_msg, TX_WAIT_FOREVER);
      uint8_t rx_byte = (uint8_t)rx_msg;

      if(end[eol.length()] == rx_byte)
      {
        eol.push_back(rx_byte);
        if(eol.length() == end.length())
          break;
      }
      else
      {
        if(eol.length() > 0)
        {
          message->append(eol);
          eol.clear();
        }
        message->push_back(rx_byte);
      }
    }
    return message;
  }
}

static void ISR_UART(UART_HandleTypeDef *huart)
{
  int i = 0;
  for(; i < uart_idx; i++)
  {
    if(handles[i] == huart)
    {
      uint32_t tx_msg = (uint32_t)0x00FF & *rx_chars[i];
      tx_queue_send(queues[i], &tx_msg, TX_NO_WAIT);
      break;
    }
  }

  /* Fire again request to get a new byte. */
  HAL_UART_Receive_IT(huart, rx_chars[i], 1);
}
