/*
 * exti_handlers.c
 *
 *  Created on: Feb 10, 2025
 *      Author: jeremy
 */

#include "sdmmc.h"
#include "tx_api.h"
#include <stdint.h>


extern TX_SEMAPHORE    card_in_semaphore;
extern TX_SEMAPHORE    card_out_semaphore;
/**
  * @brief  EXTI line detection callback.
  * @param  GPIO_Pin: Specifies the port pin connected to corresponding EXTI line.
  * @retval None
  */
void HAL_GPIO_EXTI_Falling_Callback(uint16_t GPIO_Pin)
{
	if(GPIO_Pin == uSD_DETECT_Pin)
	{
		tx_semaphore_ceiling_put(&card_in_semaphore, 1);
	}
	else if(GPIO_Pin == WW_DET_IN_Pin) {
		// jhdbg
		int x = 0;
		x = x + 1;
	}
}

void HAL_GPIO_EXTI_Rising_Callback(uint16_t GPIO_Pin)
{
	if(GPIO_Pin == uSD_DETECT_Pin)
	{
		tx_semaphore_ceiling_put(&card_out_semaphore, 1);
	}
}

