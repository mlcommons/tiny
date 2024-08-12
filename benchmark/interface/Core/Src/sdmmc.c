/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    sdmmc.c
  * @brief   This file provides code for the configuration
  *          of the SDMMC instances.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "sdmmc.h"

/* USER CODE BEGIN 0 */
#include "tx_api.h"

#define SD_DETECT_Pin                       GPIO_PIN_14
#define SD_DETECT_GPIO_Port                 GPIOH
#define SD_DETECT_EXTI_IRQn                 EXTI14_IRQn

TX_SEMAPHORE    card_in_semaphore;
TX_SEMAPHORE    card_out_semaphore;

/* USER CODE END 0 */

SD_HandleTypeDef hsd1;

/* SDMMC1 init function */

void MX_SDMMC1_SD_Init(void)
{

  /* USER CODE BEGIN SDMMC1_Init 0 */

  UINT ret = TX_SUCCESS;

  ret = tx_semaphore_create(&card_in_semaphore, "SD Card Insert semaphore", 0);
  if (ret != TX_SUCCESS)
  {
    Error_Handler();
  }
  ret = tx_semaphore_create(&card_out_semaphore, "SD Card Remove semaphore", 0);
  if (ret != TX_SUCCESS)
  {
    Error_Handler();
  }

  if(SD_IsDetected() != SD_PRESENT)
    return;

  /* USER CODE END SDMMC1_Init 0 */

  /* USER CODE BEGIN SDMMC1_Init 1 */

  /* USER CODE END SDMMC1_Init 1 */
  hsd1.Instance = SDMMC1;
  hsd1.Init.ClockEdge = SDMMC_CLOCK_EDGE_RISING;
  hsd1.Init.ClockPowerSave = SDMMC_CLOCK_POWER_SAVE_DISABLE;
  hsd1.Init.BusWide = SDMMC_BUS_WIDE_4B;
  hsd1.Init.HardwareFlowControl = SDMMC_HARDWARE_FLOW_CONTROL_DISABLE;
  hsd1.Init.ClockDiv = 4;
  if (HAL_SD_Init(&hsd1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SDMMC1_Init 2 */

  /* USER CODE END SDMMC1_Init 2 */

}

void HAL_SD_MspInit(SD_HandleTypeDef* sdHandle)
{

  GPIO_InitTypeDef GPIO_InitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};
  if(sdHandle->Instance==SDMMC1)
  {
  /* USER CODE BEGIN SDMMC1_MspInit 0 */

  /* USER CODE END SDMMC1_MspInit 0 */

  /** Initializes the peripherals clock
  */
    PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_SDMMC1;
    PeriphClkInitStruct.Sdmmc1ClockSelection = RCC_SDMMC1CLKSOURCE_PLL1Q;
    if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
    {
      Error_Handler();
    }

    /* SDMMC1 clock enable */
    __HAL_RCC_SDMMC1_CLK_ENABLE();

    __HAL_RCC_GPIOD_CLK_ENABLE();
    __HAL_RCC_GPIOC_CLK_ENABLE();
    /**SDMMC1 GPIO Configuration
    PD2     ------> SDMMC1_CMD
    PC12     ------> SDMMC1_CK
    PC11     ------> SDMMC1_D3
    PC10     ------> SDMMC1_D2
    PC9     ------> SDMMC1_D1
    PC8     ------> SDMMC1_D0
    */
    GPIO_InitStruct.Pin = SDIO1_CMD_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF12_SDMMC1;
    HAL_GPIO_Init(SDIO1_CMD_GPIO_Port, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = SDIO1_CK_Pin|SDIO1_D3_Pin|SDIO1_D2_Pin|SDIO1_D1_Pin
                          |SDIO1_D0_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF12_SDMMC1;
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

    /* SDMMC1 interrupt Init */
    HAL_NVIC_SetPriority(SDMMC1_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(SDMMC1_IRQn);
  /* USER CODE BEGIN SDMMC1_MspInit 1 */

  /* USER CODE END SDMMC1_MspInit 1 */
  }
}

void HAL_SD_MspDeInit(SD_HandleTypeDef* sdHandle)
{

  if(sdHandle->Instance==SDMMC1)
  {
  /* USER CODE BEGIN SDMMC1_MspDeInit 0 */

  /* USER CODE END SDMMC1_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_SDMMC1_CLK_DISABLE();

    /**SDMMC1 GPIO Configuration
    PD2     ------> SDMMC1_CMD
    PC12     ------> SDMMC1_CK
    PC11     ------> SDMMC1_D3
    PC10     ------> SDMMC1_D2
    PC9     ------> SDMMC1_D1
    PC8     ------> SDMMC1_D0
    */
    HAL_GPIO_DeInit(SDIO1_CMD_GPIO_Port, SDIO1_CMD_Pin);

    HAL_GPIO_DeInit(GPIOC, SDIO1_CK_Pin|SDIO1_D3_Pin|SDIO1_D2_Pin|SDIO1_D1_Pin
                          |SDIO1_D0_Pin);

    /* SDMMC1 interrupt Deinit */
    HAL_NVIC_DisableIRQ(SDMMC1_IRQn);
  /* USER CODE BEGIN SDMMC1_MspDeInit 1 */

  /* USER CODE END SDMMC1_MspDeInit 1 */
  }
}

/* USER CODE BEGIN 1 */

/**
  * @brief  EXTI line detection callback.
  * @param  GPIO_Pin: Specifies the port pin connected to corresponding EXTI line.
  * @retval None
  */
void HAL_GPIO_EXTI_Falling_Callback(uint16_t GPIO_Pin)
{
  if(GPIO_Pin == SD_DETECT_Pin)
  {
    tx_semaphore_ceiling_put(&card_in_semaphore, 1);
  }
}

void HAL_GPIO_EXTI_Rising_Callback(uint16_t GPIO_Pin)
{
  if(GPIO_Pin == SD_DETECT_Pin)
  {
    tx_semaphore_ceiling_put(&card_out_semaphore, 1);
  }
}

int32_t SD_IsDetected()
{
  int32_t ret;
  /* Check SD card detect pin */
  if (HAL_GPIO_ReadPin(SD_DETECT_GPIO_Port, SD_DETECT_Pin) == GPIO_PIN_SET)
  {
    ret = SD_NOT_PRESENT;
  }
  else
  {
    ret = SD_PRESENT;
  }

  return(int32_t)ret;
}

void SD_WaitForCardInserted()
{
  while(SD_IsDetected() != SD_PRESENT)
  {
    tx_semaphore_get(&card_in_semaphore, TX_WAIT_FOREVER);

    /* for debouncing purpose we wait a bit till it settles down */
    tx_thread_sleep(TX_TIMER_TICKS_PER_SECOND / 2);
    tx_semaphore_get(&card_out_semaphore, TX_NO_WAIT);

  }
}

void SD_WaitForCardRemoved()
{
  while(SD_IsDetected() == SD_PRESENT)
  {
    tx_semaphore_get(&card_out_semaphore, TX_WAIT_FOREVER);

    /* for debouncing purpose we wait a bit till it settles down */
    tx_thread_sleep(TX_TIMER_TICKS_PER_SECOND / 2);
    tx_semaphore_get(&card_in_semaphore, TX_NO_WAIT);
  }
}

/* USER CODE END 1 */
