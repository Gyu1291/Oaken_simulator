#hidden dim = 1024일 때의 cycle

FVRF_RD_CYCLE = 2
FVRF_WR_CYCLE = 1
FSRF_RD_CYCLE = 2
FSRF_WR_CYCLE = 1
# FIFO
FIFO_RD_CYCLE = 2
FIFO_WR_CYCLE = 1
# Processing Unit
PIPELINE_CYCLE = 7 # VPU pipeline -> VFU(MFU) pipeline -> vfu(mfu) -> VFU(MFU) pipeline->  VPU pipeline->  SFU pipeline -> sfu -> SFU pipeline
BYPASS_CYCLE = 1

#VFU
ADD_CYCLE = 8
SUB_CYCLE = 8
MUL_CYCLE = 8
DIV_CYCLE = 28
EXP_CYCLE = 20

#SFU2
SUM_CYCLE = 16
DIVLEN_CYCLE = 28
ARGMAX_CYCLE = 7
REDUMAX_CYCLE = 7
RECIP_CYCLE = 29
EPSADD_CYCLE = 4
SQRT_CYCLE = 17
LINGOVO_CYCLE = 26 
V2S_CYCLE = 3
S2V_CYCLE = 3


#MFU
MATMUL_DELAY = 36
MATMUL_LATENCY = 37
NUM_LANE = 16 #should get hwconfig from json 
DIM_VECTOR = 64 #should get hwconfig from json
#SFU1
LUT_CYCLE = 7
MAX_CYCLE = 5
ROPE_CYCLE = 5
LERP_CYCLE = 14
EXTDIV_CYCLE = 13
MASKING_CYCLE = 5 #not estimated yet

#Sampler
SAMPLING_CYCLE = 1 #not estimated yet
#Sorter
SORTING_CYCLE = 1 #not estimated yet