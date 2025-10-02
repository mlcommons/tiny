/*
 * sww_util.h
 *
 *  Created on: Jan 16, 2025
 *      Author: jeremy
 */

#ifndef INC_SWW_UTIL_H_
#define INC_SWW_UTIL_H_

// includes
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <ctype.h>
#include <math.h>

//#include "stm32l4xx_hal.h"
//#include "arm_math.h"
#include "feature_extraction.h"
#include "sww_ref_util_submitter.h"

// needed for running the model and/or initializing inference setup
//#include "sww_model.h"
//#include "sww_model_data.h"
#include "fixed_data.h"

#define EE_FW_VERSION "MLPerf Tiny Firmware V0.1.0"

/* Version 1.0 of the benchmark only supports these models */
#define EE_MODEL_VERSION_KWS01 "kws01"
#define EE_MODEL_VERSION_VWW01 "vww01"
#define EE_MODEL_VERSION_AD01 "ad01"
#define EE_MODEL_VERSION_IC01 "ic01"
#define EE_MODEL_VERSION_SWW01 "sww01"

typedef enum { EE_ARG_CLAIMED, EE_ARG_UNCLAIMED } arg_claimed_t;
typedef enum { EE_STATUS_OK = 0, EE_STATUS_ERROR } ee_status_t;

#define EE_DEVICE_NAME "dut"

#define EE_CMD_SIZE 1028u
#define EE_CMD_DELIMITER " "
#define EE_CMD_TERMINATOR '%'
#define EE_CMD_NAME "name"
#define EE_CMD_TIMESTAMP "timestamp"

#define EE_MSG_READY "m-ready\r\n"
#define EE_MSG_INIT_DONE "m-init-done\r\n"
#define EE_MSG_NAME "m-name-%s-[%s]\r\n"
#define EE_MSG_TIMESTAMP "m-lap-us-%lu\r\n"
#define EE_ERR_CMD "e-[Unknown command: %s]\r\n"

#define SWW_WINLEN_SAMPLES 1024
#define SWW_WINSTRIDE_SAMPLES 512
#define SWW_MODEL_INPUT_SIZE 1200
#define NUM_MEL_FILTERS 40
#define DETECT_THRESHOLD 115

// struct for a log we can write to without the delay of printing to UART
#define LOG_BUFFER_SIZE 8192
typedef struct {
    char buffer[LOG_BUFFER_SIZE];
    size_t current_pos;
} LogBuffer;



typedef enum {
	Idle,
	FileCapture, // capture a block of a pre-determined size, then stop
	Streaming,   // continually capture blocks of "window_step" size, then
	             // extract features and run NN
	Stopping     // command received to stop streaming detection, but not processed yet

} i2s_state_t;


void ee_print_vals_int16(const int16_t *buffer, uint32_t num_vals);
void ee_print_vals_int8(const int8_t *buffer, uint32_t num_vals);
void ee_print_bytes(const uint8_t *buffer, uint32_t num_bytes);
void ee_print_vals_float(const float *buffer, uint32_t num_vals);
void ee_log_printf(LogBuffer *log, const char *format, ...);

void ee_process_command(char *full_command);
void ee_serial_callback(char c);
void ee_timestamp(void);
void ee_set_processing_pin_high(void);
void ee_set_processing_pin_low(void);

void ee_setup_i2s_buffers();
void ee_process_chunk_and_cont_capture(void *hsai);

#endif /* INC_SWW_UTIL_H_ */
