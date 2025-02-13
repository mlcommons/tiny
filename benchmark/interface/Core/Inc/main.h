/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32h5xx_hal.h"

#include "stm32h5xx_ll_ucpd.h"
#include "stm32h5xx_ll_bus.h"
#include "stm32h5xx_ll_cortex.h"
#include "stm32h5xx_ll_rcc.h"
#include "stm32h5xx_ll_system.h"
#include "stm32h5xx_ll_utils.h"
#include "stm32h5xx_ll_pwr.h"
#include "stm32h5xx_ll_gpio.h"
#include "stm32h5xx_ll_dma.h"

#include "stm32h5xx_ll_exti.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define SAI2_FS_A_Pin GPIO_PIN_7
#define SAI2_FS_A_GPIO_Port GPIOI
#define SAI2_SCK_A_Pin GPIO_PIN_5
#define SAI2_SCK_A_GPIO_Port GPIOI
#define ARD_D3_Pin GPIO_PIN_5
#define ARD_D3_GPIO_Port GPIOB
#define WW_DET_IN_Pin GPIO_PIN_15
#define WW_DET_IN_GPIO_Port GPIOG
#define WW_DET_IN_EXTI_IRQn EXTI15_IRQn
#define TRACED1_Pin GPIO_PIN_14
#define TRACED1_GPIO_Port GPIOG
#define SAI2_SD_B_Pin GPIO_PIN_10
#define SAI2_SD_B_GPIO_Port GPIOG
#define LCD_NWE_Pin GPIO_PIN_5
#define LCD_NWE_GPIO_Port GPIOD
#define LCD_TE_Pin GPIO_PIN_3
#define LCD_TE_GPIO_Port GPIOD
#define LCD_D3_Pin GPIO_PIN_1
#define LCD_D3_GPIO_Port GPIOD
#define LCD_BL_CTRL_Pin GPIO_PIN_3
#define LCD_BL_CTRL_GPIO_Port GPIOI
#define ARD_D13_Pin GPIO_PIN_1
#define ARD_D13_GPIO_Port GPIOI
#define TRACED0_Pin GPIO_PIN_3
#define TRACED0_GPIO_Port GPIOE
#define SAI2_MCLK_A_Pin GPIO_PIN_4
#define SAI2_MCLK_A_GPIO_Port GPIOI
#define MEMS_LED_Pin GPIO_PIN_1
#define MEMS_LED_GPIO_Port GPIOE
#define I2C1_SCL_Pin GPIO_PIN_6
#define I2C1_SCL_GPIO_Port GPIOB
#define SWO_Pin GPIO_PIN_3
#define SWO_GPIO_Port GPIOB
#define RMII_TXD1_Pin GPIO_PIN_12
#define RMII_TXD1_GPIO_Port GPIOG
#define LCD_NOE_Pin GPIO_PIN_4
#define LCD_NOE_GPIO_Port GPIOD
#define LCD_D2_Pin GPIO_PIN_0
#define LCD_D2_GPIO_Port GPIOD
#define JTCK_Pin GPIO_PIN_14
#define JTCK_GPIO_Port GPIOA
#define ARD_D12_Pin GPIO_PIN_2
#define ARD_D12_GPIO_Port GPIOI
#define LCD_RST_Pin GPIO_PIN_13
#define LCD_RST_GPIO_Port GPIOH
#define TRACED3_Pin GPIO_PIN_6
#define TRACED3_GPIO_Port GPIOE
#define SAI2_SD_A_Pin GPIO_PIN_6
#define SAI2_SD_A_GPIO_Port GPIOI
#define DETECTN_Pin GPIO_PIN_0
#define DETECTN_GPIO_Port GPIOE
#define I2C1_SDA_Pin GPIO_PIN_7
#define I2C1_SDA_GPIO_Port GPIOB
#define NJTRST_Pin GPIO_PIN_4
#define NJTRST_GPIO_Port GPIOB
#define RMII_TXD0_Pin GPIO_PIN_13
#define RMII_TXD0_GPIO_Port GPIOG
#define RMII_TX_EN_Pin GPIO_PIN_11
#define RMII_TX_EN_GPIO_Port GPIOG
#define SDIO1_CMD_Pin GPIO_PIN_2
#define SDIO1_CMD_GPIO_Port GPIOD
#define SDIO1_CK_Pin GPIO_PIN_12
#define SDIO1_CK_GPIO_Port GPIOC
#define SDIO1_D3_Pin GPIO_PIN_11
#define SDIO1_D3_GPIO_Port GPIOC
#define JTDI_Pin GPIO_PIN_15
#define JTDI_GPIO_Port GPIOA
#define USB_FS_P_Pin GPIO_PIN_12
#define USB_FS_P_GPIO_Port GPIOA
#define TRACED2_Pin GPIO_PIN_5
#define TRACED2_GPIO_Port GPIOE
#define SDIO1_D2_Pin GPIO_PIN_10
#define SDIO1_D2_GPIO_Port GPIOC
#define uSD_DETECT_Pin GPIO_PIN_14
#define uSD_DETECT_GPIO_Port GPIOH
#define uSD_DETECT_EXTI_IRQn EXTI14_IRQn
#define USB_FS_N_Pin GPIO_PIN_11
#define USB_FS_N_GPIO_Port GPIOA
#define GREEN_LED_Pin GPIO_PIN_9
#define GREEN_LED_GPIO_Port GPIOI
#define ORANGE_LED_Pin GPIO_PIN_8
#define ORANGE_LED_GPIO_Port GPIOI
#define WAKEUP_Pin GPIO_PIN_13
#define WAKEUP_GPIO_Port GPIOC
#define JTMS_Pin GPIO_PIN_13
#define JTMS_GPIO_Port GPIOA
#define STLK_VCP_TX_Pin GPIO_PIN_10
#define STLK_VCP_TX_GPIO_Port GPIOA
#define STLK_VCP_RX_Pin GPIO_PIN_9
#define STLK_VCP_RX_GPIO_Port GPIOA
#define RED_LED_Pin GPIO_PIN_1
#define RED_LED_GPIO_Port GPIOF
#define LCD_A0_RS_Pin GPIO_PIN_0
#define LCD_A0_RS_GPIO_Port GPIOF
#define AUDIO_NRST_Pin GPIO_PIN_11
#define AUDIO_NRST_GPIO_Port GPIOI
#define RMII_RX_ER_Pin GPIO_PIN_10
#define RMII_RX_ER_GPIO_Port GPIOI
#define SDIO1_D1_Pin GPIO_PIN_9
#define SDIO1_D1_GPIO_Port GPIOC
#define SDIO1_D0_Pin GPIO_PIN_8
#define SDIO1_D0_GPIO_Port GPIOC
#define ARD_D9_Pin GPIO_PIN_8
#define ARD_D9_GPIO_Port GPIOA
#define BLUE_LED_Pin GPIO_PIN_4
#define BLUE_LED_GPIO_Port GPIOF
#define STMOD_17_Pin GPIO_PIN_3
#define STMOD_17_GPIO_Port GPIOF
#define LCD_PWR_ON_Pin GPIO_PIN_6
#define LCD_PWR_ON_GPIO_Port GPIOC
#define ARD_D8_Pin GPIO_PIN_8
#define ARD_D8_GPIO_Port GPIOG
#define LCD_CTP_INT_Pin GPIO_PIN_7
#define LCD_CTP_INT_GPIO_Port GPIOG
#define LCD_CTP_RST_Pin GPIO_PIN_3
#define LCD_CTP_RST_GPIO_Port GPIOG
#define ARD_D7_Pin GPIO_PIN_5
#define ARD_D7_GPIO_Port GPIOG
#define LCD_D1_Pin GPIO_PIN_15
#define LCD_D1_GPIO_Port GPIOD
#define ARD_D4_Pin GPIO_PIN_4
#define ARD_D4_GPIO_Port GPIOG
#define LCD_D15_Pin GPIO_PIN_10
#define LCD_D15_GPIO_Port GPIOD
#define LCD_D0_Pin GPIO_PIN_14
#define LCD_D0_GPIO_Port GPIOD
#define RMII_MDC_Pin GPIO_PIN_1
#define RMII_MDC_GPIO_Port GPIOC
#define RMII_REF_CLK_Pin GPIO_PIN_1
#define RMII_REF_CLK_GPIO_Port GPIOA
#define STMOD_18_Pin GPIO_PIN_12
#define STMOD_18_GPIO_Port GPIOB
#define LCD_D14_Pin GPIO_PIN_9
#define LCD_D14_GPIO_Port GPIOD
#define RMII_MDIO_Pin GPIO_PIN_2
#define RMII_MDIO_GPIO_Port GPIOA
#define ARD_A1_Pin GPIO_PIN_4
#define ARD_A1_GPIO_Port GPIOA
#define ARD_D1_Pin GPIO_PIN_10
#define ARD_D1_GPIO_Port GPIOB
#define STMOD_11_INT_Pin GPIO_PIN_9
#define STMOD_11_INT_GPIO_Port GPIOH
#define STMOD_14_PWM_Pin GPIO_PIN_12
#define STMOD_14_PWM_GPIO_Port GPIOH
#define ARD_D11_Pin GPIO_PIN_15
#define ARD_D11_GPIO_Port GPIOB
#define LCD_D13_Pin GPIO_PIN_8
#define LCD_D13_GPIO_Port GPIOD
#define PDM_SAI1_SD3_Pin GPIO_PIN_3
#define PDM_SAI1_SD3_GPIO_Port GPIOC
#define RMII_RXD0_Pin GPIO_PIN_4
#define RMII_RXD0_GPIO_Port GPIOC
#define ARD_D10_Pin GPIO_PIN_3
#define ARD_D10_GPIO_Port GPIOA
#define ARD_A5_Pin GPIO_PIN_12
#define ARD_A5_GPIO_Port GPIOF
#define LCD_D6_Pin GPIO_PIN_9
#define LCD_D6_GPIO_Port GPIOE
#define LCD_D11_Pin GPIO_PIN_14
#define LCD_D11_GPIO_Port GPIOE
#define LCD_D12_Pin GPIO_PIN_15
#define LCD_D12_GPIO_Port GPIOE
#define ARD_D0_Pin GPIO_PIN_11
#define ARD_D0_GPIO_Port GPIOB
#define STMOD_9_MISOs_Pin GPIO_PIN_8
#define STMOD_9_MISOs_GPIO_Port GPIOH
#define ARD_D6_Pin GPIO_PIN_10
#define ARD_D6_GPIO_Port GPIOH
#define UCPD_CC2_Pin GPIO_PIN_14
#define UCPD_CC2_GPIO_Port GPIOB
#define STMOD_20_Pin GPIO_PIN_5
#define STMOD_20_GPIO_Port GPIOH
#define ARD_A2_Pin GPIO_PIN_0
#define ARD_A2_GPIO_Port GPIOA
#define RMII_RXD1_Pin GPIO_PIN_5
#define RMII_RXD1_GPIO_Port GPIOC
#define ARD_A4_Pin GPIO_PIN_6
#define ARD_A4_GPIO_Port GPIOA
#define IBUS_SENSE_Pin GPIO_PIN_13
#define IBUS_SENSE_GPIO_Port GPIOF
#define UCPD_FLT_Pin GPIO_PIN_1
#define UCPD_FLT_GPIO_Port GPIOG
#define LCD_D5_Pin GPIO_PIN_8
#define LCD_D5_GPIO_Port GPIOE
#define LCD_D8_Pin GPIO_PIN_11
#define LCD_D8_GPIO_Port GPIOE
#define LCD_D10_Pin GPIO_PIN_13
#define LCD_D10_GPIO_Port GPIOE
#define STMOD_12_Pin GPIO_PIN_6
#define STMOD_12_GPIO_Port GPIOH
#define STMOD_12_RST_Pin GPIO_PIN_7
#define STMOD_12_RST_GPIO_Port GPIOH
#define ARD_D5_Pin GPIO_PIN_11
#define ARD_D5_GPIO_Port GPIOH
#define STMOD_19_Pin GPIO_PIN_4
#define STMOD_19_GPIO_Port GPIOH
#define ARD_A3_Pin GPIO_PIN_5
#define ARD_A3_GPIO_Port GPIOA
#define RMII_CRS_DV_Pin GPIO_PIN_7
#define RMII_CRS_DV_GPIO_Port GPIOA
#define ARD_A0_Pin GPIO_PIN_0
#define ARD_A0_GPIO_Port GPIOB
#define STMOD_13_Pin GPIO_PIN_11
#define STMOD_13_GPIO_Port GPIOF
#define VBUS_SENSE_Pin GPIO_PIN_14
#define VBUS_SENSE_GPIO_Port GPIOF
#define UCPD_PWR_Pin GPIO_PIN_0
#define UCPD_PWR_GPIO_Port GPIOG
#define LCD_D7_Pin GPIO_PIN_10
#define LCD_D7_GPIO_Port GPIOE
#define LCD_D9_Pin GPIO_PIN_12
#define LCD_D9_GPIO_Port GPIOE
#define UCPD_CC1_Pin GPIO_PIN_13
#define UCPD_CC1_GPIO_Port GPIOB

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
