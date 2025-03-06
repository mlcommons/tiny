/*
 * sww_util.h
 *
 *  Created on: Jan 16, 2025
 *      Author: jeremy
 */

#ifndef INC_SWW_UTIL_H_
#define INC_SWW_UTIL_H_

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include "sww_model.h"


#define EE_FW_VERSION "MLPerf Tiny Firmware V0.1.0"

/* Version 1.0 of the benchmark only supports these models */
#define EE_MODEL_VERSION_KWS01 "kws01"
#define EE_MODEL_VERSION_VWW01 "vww01"
#define EE_MODEL_VERSION_AD01 "ad01"
#define EE_MODEL_VERSION_IC01 "ic01"
#define EE_MODEL_VERSION_SWW01 "sww01"





#define TH_MODEL_VERSION EE_MODEL_VERSION_SWW01


typedef enum { EE_ARG_CLAIMED, EE_ARG_UNCLAIMED } arg_claimed_t;
typedef enum { EE_STATUS_OK = 0, EE_STATUS_ERROR } ee_status_t;

#define EE_DEVICE_NAME "dut"

#define EE_CMD_SIZE 80u
#define EE_CMD_DELIMITER " "
#define EE_CMD_TERMINATOR '%'

#define EE_CMD_NAME "name"
#define EE_CMD_TIMESTAMP "timestamp"

#define EE_MSG_READY "m-ready\r\n"
#define EE_MSG_INIT_DONE "m-init-done\r\n"
#define EE_MSG_NAME "m-name-%s-[%s]\r\n"
#define EE_MSG_TIMESTAMP "m-lap-us-%lu\r\n"

#define EE_ERR_CMD "e-[Unknown command: %s]\r\n"

#define TH_VENDOR_NAME_STRING "ML Commons"


#define SWW_WINLEN_SAMPLES 1024
#define SWW_WINSTRIDE_SAMPLES 512
#define SWW_MODEL_INPUT_SIZE 1200
#define NUM_MEL_FILTERS 40

// struct for a log we can write to without the delay of printing to UART
#define LOG_BUFFER_SIZE 4096
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


void print_vals_int16(const int16_t *buffer, uint32_t num_vals);
void print_bytes(const uint8_t *buffer, uint32_t num_bytes);
void print_vals_float(const float *buffer, uint32_t num_vals);
void log_printf(LogBuffer *log, const char *format, ...);

void process_command(char *full_command);
void ee_serial_callback(char c);

ai_error aiInit(void);
void setup_i2s_buffers();
void compute_lfbe_f32(const int16_t *pSrc, float32_t *pDst, float32_t *pTmp);


#endif /* INC_SWW_UTIL_H_ */
