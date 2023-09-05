/**********************************************************
 * Author        : Peng Xuejian
 * Email         : pengxj@ksitri.com
 * Last modified : 2020-09-01 09:45
 * Filename      : Def.h
 * Description   : 
 * *******************************************************/
#ifndef __DEF_H__
#define __DEF_H__

//#include "common/log.hpp"

#ifdef AQ_EMBEDED
#ifndef INFO
#define INFO(fmt...)   \
do {\
    fprintf(stderr, "\033[0;33m[%s]-%d: ", __FUNCTION__, __LINE__);\
    fprintf(stderr, fmt);\
    fprintf(stderr, "\033[0;39m");\
}while(0)
#endif
#endif

#ifdef NDEBUG
#define DEBUG(infor,...)
#else
#define DEBUG(infor,...) (fprintf(stderr,infor,##__VA_ARGS__))
#endif

#define VERSION_INFO "Version_1.2"

#define SENSOR_PORT_BASE 5000
#define CAMERA_PORT_BASE 6600
#define RECORDER_PORT_BASE 7000
#define RECORDER_CON_PORT 8000

#define SENSOR_DEFAULT_FRAME_RATE 25
#define MAX_SELECTED_HIGH_SENSORS 4

//Unit:milliseconds
#define CONN_TIMEOUT 50
#define SEND_TIMEOUT 120000
#define RECV_TIMEOUT 30000
#define DATACLIENT_WRITE_QUEUE_TIMEOUT 99000
#define TRANS_IN_QUEUE_TIMEOUT 100000
#define TRANS_OUT_QUEUE_TIMEOUT 101000
#define DECODER_TIMESTAMP_QUEUE_TIMEOUT 100010
#define DECODER_FRAME_QUEUE_TIMEOUT 100020

#define FRAME_HEAD_LEN (sizeof(int32_t) + sizeof(uint64_t))
#define MAX_CONN 10   //maximum number of connnection one camera can hold

#define GET_FRAME_BUF_SIZE(ENCODED_FRAME_BUF) (*((int32_t*)ENCODED_FRAME_BUF) + sizeof(int32_t))
#define GET_FRAME(ENCODED_FRAME_BUF) ( ENCODED_FRAME_BUF + sizeof(int32_t))
#define GET_FRAME_SIZE(ENCODED_FRAME_BUF) ( *((int32_t*)ENCODED_FRAME_BUF))
#define GET_FRAME_DATA(ENCODED_FRAME_BUF) ( ENCODED_FRAME_BUF + FRAME_HEAD_LEN )
#define GET_FRAME_DATA_SIZE(ENCODED_FRAME_BUF) ( *((int32_t*)ENCODED_FRAME_BUF) - sizeof(uint64_t))
#define GET_FRAME_TS(ENCODED_FRAME_BUF) (*((uint64_t*)(ENCODED_FRAME_BUF + sizeof(int32_t)))) //Millisecnonds
#define CHECK_IDR_FRAME(ENCODED_FRAME_BUF) (ENCODED_FRAME_BUF[FRAME_HEAD_LEN] 	=='\0' && \
									ENCODED_FRAME_BUF[FRAME_HEAD_LEN + 1] =='\0' && \
									ENCODED_FRAME_BUF[FRAME_HEAD_LEN + 2] =='\0' && \
									ENCODED_FRAME_BUF[FRAME_HEAD_LEN + 3] =='\1' && \
									(ENCODED_FRAME_BUF[FRAME_HEAD_LEN + 4] == 0x67 || ENCODED_FRAME_BUF[FRAME_HEAD_LEN + 4] == 0x40))

#define GET_DECODED_CPU_FRAME_WIDTH(FRAME_BUF) (*((int32_t*)FRAME_BUF))
#define GET_DECODED_CPU_FRAME_HEIGHT(FRAME_BUF) (*((int32_t*)(FRAME_BUF + sizeof(int32_t))))
#define GET_DECODED_CPU_FRAME_TS(FRAME_BUF) (*((uint64_t*)(FRAME_BUF + sizeof(int32_t) * 2)))
#define GET_DECODED_FRAME_DATA(FRAME_BUF) (FRAME_BUF + sizeof(int32_t) * 2 + sizeof(uint64_t))

//Ratio of width of two resolutions must be equal to ratio of height, currenlty it's 2
#define COMPOSE_SCALE 4.f
#endif
