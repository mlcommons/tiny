/*
 * submitter_implemented.h
 *
 *  Created on: Sep 3, 2025
 *      Author: owen
 * \file
 * \brief Submitter implementations required to perform inference.
 * \detail All methods starting with th_ are platform-specific and to be
 * implemented by the submitter. All basic I/O, inference and timer APIs must
 * be implemented in order for the benchmark to output useful results, but some
 * auxiliary methods default to an empty implementation. These methods are
 * provided to enable submitter optimizations and are not required for
 * submission.
 */

#ifndef __SWW_REF_UTIL_SUBMITTER_H__
#define __SWW_REF_UTIL_SUBMITTER_H__

#include "stm32l4xx_hal.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

// needed for running the model and/or initializing inference setup
#include "sww_model.h"
#include "sww_model_data.h"
#include "fixed_data.h"

#include "sww_ref_util.h"

// I/O defines
#define B1_Pin GPIO_PIN_13
#define B1_GPIO_Port GPIOC
#define timestamp_Pin GPIO_PIN_13
#define timestamp_GPIO_Port GPIOF
#define Processing_Pin GPIO_PIN_9
#define Processing_GPIO_Port GPIOE
#define LD3_Pin GPIO_PIN_14
#define LD3_GPIO_Port GPIOB
#define STLK_RX_Pin GPIO_PIN_8
#define STLK_RX_GPIO_Port GPIOD
#define STLK_TX_Pin GPIO_PIN_9
#define STLK_TX_GPIO_Port GPIOD
#define USB_OverCurrent_Pin GPIO_PIN_5
#define USB_OverCurrent_GPIO_Port GPIOG
#define USB_PowerSwitchOn_Pin GPIO_PIN_6
#define USB_PowerSwitchOn_GPIO_Port GPIOG
#define STLINK_TX_Pin GPIO_PIN_7
#define STLINK_TX_GPIO_Port GPIOG
#define STLINK_RX_Pin GPIO_PIN_8
#define STLINK_RX_GPIO_Port GPIOG
#define USB_SOF_Pin GPIO_PIN_8
#define USB_SOF_GPIO_Port GPIOA
#define USB_VBUS_Pin GPIO_PIN_9
#define USB_VBUS_GPIO_Port GPIOA
#define USB_ID_Pin GPIO_PIN_10
#define USB_ID_GPIO_Port GPIOA
#define USB_DM_Pin GPIO_PIN_11
#define USB_DM_GPIO_Port GPIOA
#define USB_DP_Pin GPIO_PIN_12
#define USB_DP_GPIO_Port GPIOA
#define TMS_Pin GPIO_PIN_13
#define TMS_GPIO_Port GPIOA
#define TCK_Pin GPIO_PIN_14
#define TCK_GPIO_Port GPIOA
#define SWO_Pin GPIO_PIN_3
#define SWO_GPIO_Port GPIOB
#define LD2_Pin GPIO_PIN_7
#define LD2_GPIO_Port GPIOB
#define WW_DETECTED_Pin GPIO_PIN_8
#define WW_DETECTED_GPIO_Port GPIOB

// platform-specific defines
// used for the time-critical register reads and writes on the GPIO and timer
#define TH_VENDOR_NAME_STRING "ML Commons"
#define TH_MODEL_VERSION EE_MODEL_VERSION_SWW01
#define TH_I2S_OK HAL_OK
#define TH_GPIO_WRITE(port, pin, value) HAL_GPIO_WritePin(port, pin, value)
#define TH_TIMER16_GET() __HAL_TIM_GET_COUNTER(&htim16)
#if !defined(TH_GPIO_WRITE) || !defined(TH_TIMER16_GET) \
    || !defined(TH_VENDOR_NAME_STRING) || !defined(TH_I2S_OK)
#error Unmapped macros detected. Make sure all macros in submitter_implemented are defined.
#endif

// core API functions
/**
  * @brief initialize GPIO and necessary peripherals on DUT
  */
void th_hardware_init(void);

/**
  * @brief sets or reset a GPIO pin
  * @param port Pointer to the port address
  * @param pin Pin mask
  * @param value 0 or 1
  */
// void th_gpio_write(void *port, uint16_t pin, uint8_t value);

/**
  * @brief start the sixteen-bit timer peripheral on the device
  */
void th_timer16_start(void);

/**
  * @brief get 16-bit counter value of the timer peripheral
  * @retval timer value (16-bit uint8_t)
  */
// uint16_t th_timer16_get(void);


/**
  * @brief receive DMA data
  * @param dma_addr Pointer to the DMA
  * @param i2s_buffer Pointer to the I2S buffer
  * @param size Number of data bytes to receive
  * @retval DMA receive success code
  */
uint32_t th_dma_receive(void *dma_addr, uint8_t *i2s_buffer, uint16_t size);

/**
  * @brief stop DMA capture
  * @param dma_addr Pointer to the DMA memory address
  * @retval DMA status value
  */
uint32_t th_dma_stop(void *dma_addr);

/**
  * @brief receive from the device's UART
  * @param uart Pointer to the UART peripheral
  * @param data Pointer to the transmission data
  * @param size Number of data bytes to receive
  * @param timeout Timeout window in milliseconds
  * @retval UART receive success code
  */
uint32_t th_uart_receive(void *uart, uint8_t *data, uint16_t size,
    uint32_t timeout);

// private functions, originally from main.c
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_LPUART1_UART_Init(void);
static void MX_USART3_UART_Init(void);
static void MX_USB_OTG_FS_PCD_Init(void);
static void MX_SAI1_Init(void);
static void MX_TIM16_Init(void);
void Error_Handler(void);

// These defines and this function are to get printf() working
#ifdef __GNUC__
#define PUTCHAR_PROTOTYPE int __io_putchar(int ch)
#else
#define PUTCHAR_PROTOTYPE int fputc(int ch, FILE *f)
#endif

#endif
