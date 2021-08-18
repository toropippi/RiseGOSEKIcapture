#runtime "hsp3_64"
#regcmd "hsp3cmdinit","HSPCL64.dll", 1
#cmd int64 $00
#cmd qpeek $01
#cmd qpoke $02
#cmd varptr64 $03
#cmd dupptr64 $04
#cmd callfunc64i $05
#cmd callfunc64d $06
#cmd callfunc64f $07


#cmd HCLinit $50

#cmd dim_i64 $51

#cmd HCLSetDevice $53
#cmd HCLGetDeviceCount $54
#cmd HCLGetDeviceInfo $55
#cmd HCLGetSettingDevice $56

#cmd HCLCreateProgram $57
#cmd HCLCreateProgramWithSource	$58
#cmd HCLReleaseProgram $59

#cmd HCLCreateKernel $5A
#cmd HCLSetKernel $5B
#cmd HCLSetKrns $5C
#cmd HCLGetKernelName_ $79
#cmd HCLReleaseKernel $5D

#cmd HCLCreateBuffer $5E
#cmd HCLCreateBufferFrom $5F
#cmd HCLWriteBuffer $60
#cmd HCLWriteBuffer_NonBlocking $86
#cmd HCLReadBuffer $61
#cmd HCLReadBuffer_NonBlocking $87
#cmd HCLGet_NonBlocking_Status $88
#cmd HCLCopyBuffer $62
#cmd HCLFillBuffer_i32 $63
#cmd HCLFillBuffer_i64 $64
#cmd HCLFillBuffer_dp $8B
#cmd HCLReleaseBuffer $65

#cmd HCLReadIndex_i32 $66
#cmd HCLReadIndex_i64 $67
#cmd HCLReadIndex_dp $68
#cmd HCLWriteIndex_i32 $69
#cmd HCLWriteIndex_i64 $6A
#cmd HCLWriteIndex_dp $6B

#cmd HCLCall $6C
#cmd HCLDoKrn1 $6D
#cmd HCLDoKernel $6E
#cmd HCLDoKrn1_sub	$6F
#cmd HCLFinish $70
#cmd HCLFlush $71

#cmd _ExHCLSetCommandQueueMax $72
#cmd _ExHCLSetCommandQueueProperties $73
#cmd HCLSetCommandQueue $74
#cmd HCLGetSettingCommandQueue $75

#cmd _ExHCLSetEventMax $76
#cmd HCLSetWaitEvent $77
#cmd HCLSetWaitEvents $78
#cmd HCLGetEventLogs $7A
#cmd HCLGetEventStatus $7B
#cmd HCLWaitForEvent $7C
#cmd HCLWaitForEvents $7D
#cmd HCLCreateUserEvent $89
#cmd HCLSetUserEventStatus $8A

#cmd Min64 $7F
#cmd Max64 $80

#cmd DoubleToFloat $81
#cmd FloatToDouble $82

#cmd _ConvRGBtoBGR 		$83
#cmd _ConvRGBAtoRGB 		$84
#cmd _ConvRGBtoRGBA 		$85
























/* Error Codes */
#define global CL_SUCCESS                                  0
#define global CL_DEVICE_NOT_FOUND                         -1
#define global CL_DEVICE_NOT_AVAILABLE                     -2
#define global CL_COMPILER_NOT_AVAILABLE                   -3
#define global CL_MEM_OBJECT_ALLOCATION_FAILURE            -4
#define global CL_OUT_OF_RESOURCES                         -5
#define global CL_OUT_OF_HOST_MEMORY                       -6
#define global CL_PROFILING_INFO_NOT_AVAILABLE             -7
#define global CL_MEM_COPY_OVERLAP                         -8
#define global CL_IMAGE_FORMAT_MISMATCH                    -9
#define global CL_IMAGE_FORMAT_NOT_SUPPORTED               -10
#define global CL_BUILD_PROGRAM_FAILURE                    -11
#define global CL_MAP_FAILURE                              -12
#define global CL_MISALIGNED_SUB_BUFFER_OFFSET             -13
#define global CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST -14
#define global CL_COMPILE_PROGRAM_FAILURE                  -15
#define global CL_LINKER_NOT_AVAILABLE                     -16
#define global CL_LINK_PROGRAM_FAILURE                     -17
#define global CL_DEVICE_PARTITION_FAILED                  -18
#define global CL_KERNEL_ARG_INFO_NOT_AVAILABLE            -19

#define global CL_INVALID_VALUE                            -30
#define global CL_INVALID_DEVICE_TYPE                      -31
#define global CL_INVALID_PLATFORM                         -32
#define global CL_INVALID_DEVICE                           -33
#define global CL_INVALID_CONTEXT                          -34
#define global CL_INVALID_QUEUE_PROPERTIES                 -35
#define global CL_INVALID_COMMAND_QUEUE                    -36
#define global CL_INVALID_HOST_PTR                         -37
#define global CL_INVALID_MEM_OBJECT                       -38
#define global CL_INVALID_IMAGE_FORMAT_DESCRIPTOR          -39
#define global CL_INVALID_IMAGE_SIZE                       -40
#define global CL_INVALID_SAMPLER                          -41
#define global CL_INVALID_BINARY                           -42
#define global CL_INVALID_BUILD_OPTIONS                    -43
#define global CL_INVALID_PROGRAM                          -44
#define global CL_INVALID_PROGRAM_EXECUTABLE               -45
#define global CL_INVALID_KERNEL_NAME                      -46
#define global CL_INVALID_KERNEL_DEFINITION                -47
#define global CL_INVALID_KERNEL                           -48
#define global CL_INVALID_ARG_INDEX                        -49
#define global CL_INVALID_ARG_VALUE                        -50
#define global CL_INVALID_ARG_SIZE                         -51
#define global CL_INVALID_KERNEL_ARGS                      -52
#define global CL_INVALID_WORK_DIMENSION                   -53
#define global CL_INVALID_WORK_GROUP_SIZE                  -54
#define global CL_INVALID_WORK_ITEM_SIZE                   -55
#define global CL_INVALID_GLOBAL_OFFSET                    -56
#define global CL_INVALID_EVENT_WAIT_LIST                  -57
#define global CL_INVALID_EVENT                            -58
#define global CL_INVALID_OPERATION                        -59
#define global CL_INVALID_GL_OBJECT                        -60
#define global CL_INVALID_BUFFER_SIZE                      -61
#define global CL_INVALID_MIP_LEVEL                        -62
#define global CL_INVALID_GLOBAL_WORK_SIZE                 -63
#define global CL_INVALID_PROPERTY                         -64
#define global CL_INVALID_IMAGE_DESCRIPTOR                 -65
#define global CL_INVALID_COMPILER_OPTIONS                 -66
#define global CL_INVALID_LINKER_OPTIONS                   -67
#define global CL_INVALID_DEVICE_PARTITION_COUNT           -68

/* OpenCL Version */
#define global CL_VERSION_1_0                              1
#define global CL_VERSION_1_1                              1
#define global CL_VERSION_1_2                              1
#define global CL_VERSION_2_0                              1

/* cl_bool */
#define global CL_FALSE                                    0
#define global CL_TRUE                                     1
#define global CL_BLOCKING                                 CL_TRUE
#define global CL_NON_BLOCKING                             CL_FALSE

/* cl_platform_info */
#define global CL_PLATFORM_PROFILE                         0x0900
#define global CL_PLATFORM_VERSION                         0x0901
#define global CL_PLATFORM_NAME                            0x0902
#define global CL_PLATFORM_VENDOR                          0x0903
#define global CL_PLATFORM_EXTENSIONS                      0x0904


/* cl_device_type - bitfield */
#define global CL_DEVICE_TYPE_DEFAULT                      (1 << 0)
#define global CL_DEVICE_TYPE_CPU                          (1 << 1)
#define global CL_DEVICE_TYPE_GPU                          (1 << 2)
#define global CL_DEVICE_TYPE_ACCELERATOR                  (1 << 3)
#define global CL_DEVICE_TYPE_CUSTOM                       (1 << 4)
#define global CL_DEVICE_TYPE_ALL                          0xFFFFFFFF

/* cl_device_info */
#define global CL_DEVICE_TYPE                              0x1000
#define global CL_DEVICE_VENDOR_ID                         0x1001
#define global CL_DEVICE_MAX_COMPUTE_UNITS                 0x1002
#define global CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS          0x1003
#define global CL_DEVICE_MAX_WORK_GROUP_SIZE               0x1004
#define global CL_DEVICE_MAX_WORK_ITEM_SIZES               0x1005
#define global CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR       0x1006
#define global CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT      0x1007
#define global CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT        0x1008
#define global CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG       0x1009
#define global CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT      0x100A
#define global CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE     0x100B
#define global CL_DEVICE_MAX_CLOCK_FREQUENCY               0x100C
#define global CL_DEVICE_ADDRESS_BITS                      0x100D
#define global CL_DEVICE_MAX_READ_IMAGE_ARGS               0x100E
#define global CL_DEVICE_MAX_WRITE_IMAGE_ARGS              0x100F
#define global CL_DEVICE_MAX_MEM_ALLOC_SIZE                0x1010
#define global CL_DEVICE_IMAGE2D_MAX_WIDTH                 0x1011
#define global CL_DEVICE_IMAGE2D_MAX_HEIGHT                0x1012
#define global CL_DEVICE_IMAGE3D_MAX_WIDTH                 0x1013
#define global CL_DEVICE_IMAGE3D_MAX_HEIGHT                0x1014
#define global CL_DEVICE_IMAGE3D_MAX_DEPTH                 0x1015
#define global CL_DEVICE_IMAGE_SUPPORT                     0x1016
#define global CL_DEVICE_MAX_PARAMETER_SIZE                0x1017
#define global CL_DEVICE_MAX_SAMPLERS                      0x1018
#define global CL_DEVICE_MEM_BASE_ADDR_ALIGN               0x1019
#define global CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE          0x101A
#define global CL_DEVICE_SINGLE_FP_CONFIG                  0x101B
#define global CL_DEVICE_GLOBAL_MEM_CACHE_TYPE             0x101C
#define global CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE         0x101D
#define global CL_DEVICE_GLOBAL_MEM_CACHE_SIZE             0x101E
#define global CL_DEVICE_GLOBAL_MEM_SIZE                   0x101F
#define global CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE          0x1020
#define global CL_DEVICE_MAX_CONSTANT_ARGS                 0x1021
#define global CL_DEVICE_LOCAL_MEM_TYPE                    0x1022
#define global CL_DEVICE_LOCAL_MEM_SIZE                    0x1023
#define global CL_DEVICE_ERROR_CORRECTION_SUPPORT          0x1024
#define global CL_DEVICE_PROFILING_TIMER_RESOLUTION        0x1025
#define global CL_DEVICE_ENDIAN_LITTLE                     0x1026
#define global CL_DEVICE_AVAILABLE                         0x1027
#define global CL_DEVICE_COMPILER_AVAILABLE                0x1028
#define global CL_DEVICE_EXECUTION_CAPABILITIES            0x1029
#define global CL_DEVICE_QUEUE_PROPERTIES                  0x102A
#define global CL_DEVICE_NAME                              0x102B
#define global CL_DEVICE_VENDOR                            0x102C
#define global CL_DRIVER_VERSION                           0x102D
#define global CL_DEVICE_PROFILE                           0x102E
#define global CL_DEVICE_VERSION                           0x102F
#define global CL_DEVICE_EXTENSIONS                        0x1030
#define global CL_DEVICE_PLATFORM                          0x1031
#define global CL_DEVICE_DOUBLE_FP_CONFIG                  0x1032
/* 0x1033 reserved for CL_DEVICE_HALF_FP_CONFIG */
#define global CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF       0x1034
#define global CL_DEVICE_HOST_UNIFIED_MEMORY               0x1035
#define global CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR          0x1036
#define global CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT         0x1037
#define global CL_DEVICE_NATIVE_VECTOR_WIDTH_INT           0x1038
#define global CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG          0x1039
#define global CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT         0x103A
#define global CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE        0x103B
#define global CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF          0x103C
#define global CL_DEVICE_OPENCL_C_VERSION                  0x103D
#define global CL_DEVICE_LINKER_AVAILABLE                  0x103E
#define global CL_DEVICE_BUILT_IN_KERNELS                  0x103F
#define global CL_DEVICE_IMAGE_MAX_BUFFER_SIZE             0x1040
#define global CL_DEVICE_IMAGE_MAX_ARRAY_SIZE              0x1041
#define global CL_DEVICE_PARENT_DEVICE                     0x1042
#define global CL_DEVICE_PARTITION_MAX_SUB_DEVICES         0x1043
#define global CL_DEVICE_PARTITION_PROPERTIES              0x1044
#define global CL_DEVICE_PARTITION_AFFINITY_DOMAIN         0x1045
#define global CL_DEVICE_PARTITION_TYPE                    0x1046
#define global CL_DEVICE_REFERENCE_COUNT                   0x1047
#define global CL_DEVICE_PREFERRED_INTEROP_USER_SYNC       0x1048
#define global CL_DEVICE_PRINTF_BUFFER_SIZE                    0x1049
#define global CL_DEVICE_IMAGE_PITCH_ALIGNMENT                 0x104A
#define global CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT          0x104B
#define global CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS             0x104C
#define global CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE              0x104D
#define global CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES            0x104E
#define global CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE        0x104F
#define global CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE              0x1050
#define global CL_DEVICE_MAX_ON_DEVICE_QUEUES                  0x1051
#define global CL_DEVICE_MAX_ON_DEVICE_EVENTS                  0x1052
#define global CL_DEVICE_SVM_CAPABILITIES                      0x1053
#define global CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE  0x1054
#define global CL_DEVICE_MAX_PIPE_ARGS                         0x1055
#define global CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS          0x1056
#define global CL_DEVICE_PIPE_MAX_PACKET_SIZE                  0x1057
#define global CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT   0x1058
#define global CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT     0x1059
#define global CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT      0x105A

/* cl_device_fp_config - bitfield */
#define global CL_FP_DENORM                                (1 << 0)
#define global CL_FP_INF_NAN                               (1 << 1)
#define global CL_FP_ROUND_TO_NEAREST                      (1 << 2)
#define global CL_FP_ROUND_TO_ZERO                         (1 << 3)
#define global CL_FP_ROUND_TO_INF                          (1 << 4)
#define global CL_FP_FMA                                   (1 << 5)
#define global CL_FP_SOFT_FLOAT                            (1 << 6)
#define global CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT         (1 << 7)

/* cl_device_mem_cache_type */
#define global CL_NONE                                     0x0
#define global CL_READ_ONLY_CACHE                          0x1
#define global CL_READ_WRITE_CACHE                         0x2

/* cl_device_local_mem_type */
#define global CL_LOCAL                                    0x1
#define global CL_GLOBAL                                   0x2

/* cl_device_exec_capabilities - bitfield */
#define global CL_EXEC_KERNEL                              (1 << 0)
#define global CL_EXEC_NATIVE_KERNEL                       (1 << 1)

/* cl_command_queue_properties - bitfield */
#define global CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE      (1 << 0)
#define global CL_QUEUE_PROFILING_ENABLE                   (1 << 1)

/* cl_context_info  */
#define global CL_CONTEXT_REFERENCE_COUNT                  0x1080
#define global CL_CONTEXT_DEVICES                          0x1081
#define global CL_CONTEXT_PROPERTIES                       0x1082
#define global CL_CONTEXT_NUM_DEVICES                      0x1083

/* cl_context_properties */
#define global CL_CONTEXT_PLATFORM                         0x1084
#define global CL_CONTEXT_INTEROP_USER_SYNC                0x1085
    
/* cl_device_partition_property */
#define global CL_DEVICE_PARTITION_EQUALLY                 0x1086
#define global CL_DEVICE_PARTITION_BY_COUNTS               0x1087
#define global CL_DEVICE_PARTITION_BY_COUNTS_LIST_END      0x0
#define global CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN      0x1088
    
/* cl_device_affinity_domain */
#define global CL_DEVICE_AFFINITY_DOMAIN_NUMA                     (1 << 0)
#define global CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE                 (1 << 1)
#define global CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE                 (1 << 2)
#define global CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE                 (1 << 3)
#define global CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE                 (1 << 4)
#define global CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE       (1 << 5)

/* cl_command_queue_info */
#define global CL_QUEUE_CONTEXT                            0x1090
#define global CL_QUEUE_DEVICE                             0x1091
#define global CL_QUEUE_REFERENCE_COUNT                    0x1092
#define global CL_QUEUE_PROPERTIES                         0x1093

/* cl_mem_flags - bitfield */
#define global CL_MEM_READ_WRITE                           (1 << 0)
#define global CL_MEM_WRITE_ONLY                           (1 << 1)
#define global CL_MEM_READ_ONLY                            (1 << 2)
#define global CL_MEM_USE_HOST_PTR                         (1 << 3)
#define global CL_MEM_ALLOC_HOST_PTR                       (1 << 4)
#define global CL_MEM_COPY_HOST_PTR                        (1 << 5)
// reserved                                         (1 << 6)    
#define global CL_MEM_HOST_WRITE_ONLY                      (1 << 7)
#define global CL_MEM_HOST_READ_ONLY                       (1 << 8)
#define global CL_MEM_HOST_NO_ACCESS                       (1 << 9)

/* cl_mem_migration_flags - bitfield */
#define global CL_MIGRATE_MEM_OBJECT_HOST                  (1 << 0)
#define global CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED     (1 << 1)

/* cl_channel_order */
#define global CL_R                                        0x10B0
#define global CL_A                                        0x10B1
#define global CL_RG                                       0x10B2
#define global CL_RA                                       0x10B3
#define global CL_RGB                                      0x10B4
#define global CL_RGBA                                     0x10B5
#define global CL_BGRA                                     0x10B6
#define global CL_ARGB                                     0x10B7
#define global CL_INTENSITY                                0x10B8
#define global CL_LUMINANCE                                0x10B9
#define global CL_Rx                                       0x10BA
#define global CL_RGx                                      0x10BB
#define global CL_RGBx                                     0x10BC
#define global CL_DEPTH                                    0x10BD
#define global CL_DEPTH_STENCIL                            0x10BE

/* cl_channel_type */
#define global CL_SNORM_INT8                               0x10D0
#define global CL_SNORM_INT16                              0x10D1
#define global CL_UNORM_INT8                               0x10D2
#define global CL_UNORM_INT16                              0x10D3
#define global CL_UNORM_SHORT_565                          0x10D4
#define global CL_UNORM_SHORT_555                          0x10D5
#define global CL_UNORM_INT_101010                         0x10D6
#define global CL_SIGNED_INT8                              0x10D7
#define global CL_SIGNED_INT16                             0x10D8
#define global CL_SIGNED_INT32                             0x10D9
#define global CL_UNSIGNED_INT8                            0x10DA
#define global CL_UNSIGNED_INT16                           0x10DB
#define global CL_UNSIGNED_INT32                           0x10DC
#define global CL_HALF_FLOAT                               0x10DD
#define global CL_FLOAT                                    0x10DE
#define global CL_UNORM_INT24                              0x10DF

/* cl_mem_object_type */
#define global CL_MEM_OBJECT_BUFFER                        0x10F0
#define global CL_MEM_OBJECT_IMAGE2D                       0x10F1
#define global CL_MEM_OBJECT_IMAGE3D                       0x10F2
#define global CL_MEM_OBJECT_IMAGE2D_ARRAY                 0x10F3
#define global CL_MEM_OBJECT_IMAGE1D                       0x10F4
#define global CL_MEM_OBJECT_IMAGE1D_ARRAY                 0x10F5
#define global CL_MEM_OBJECT_IMAGE1D_BUFFER                0x10F6

/* cl_mem_info */
#define global CL_MEM_TYPE                                 0x1100
#define global CL_MEM_FLAGS                                0x1101
#define global CL_MEM_SIZE                                 0x1102
#define global CL_MEM_HOST_PTR                             0x1103
#define global CL_MEM_MAP_COUNT                            0x1104
#define global CL_MEM_REFERENCE_COUNT                      0x1105
#define global CL_MEM_CONTEXT                              0x1106
#define global CL_MEM_ASSOCIATED_MEMOBJECT                 0x1107
#define global CL_MEM_OFFSET                               0x1108

/* cl_image_info */
#define global CL_IMAGE_FORMAT                             0x1110
#define global CL_IMAGE_ELEMENT_SIZE                       0x1111
#define global CL_IMAGE_ROW_PITCH                          0x1112
#define global CL_IMAGE_SLICE_PITCH                        0x1113
#define global CL_IMAGE_WIDTH                              0x1114
#define global CL_IMAGE_HEIGHT                             0x1115
#define global CL_IMAGE_DEPTH                              0x1116
#define global CL_IMAGE_ARRAY_SIZE                         0x1117
#define global CL_IMAGE_BUFFER                             0x1118
#define global CL_IMAGE_NUM_MIP_LEVELS                     0x1119
#define global CL_IMAGE_NUM_SAMPLES                        0x111A

/* cl_addressing_mode */
#define global CL_ADDRESS_NONE                             0x1130
#define global CL_ADDRESS_CLAMP_TO_EDGE                    0x1131
#define global CL_ADDRESS_CLAMP                            0x1132
#define global CL_ADDRESS_REPEAT                           0x1133
#define global CL_ADDRESS_MIRRORED_REPEAT                  0x1134

/* cl_filter_mode */
#define global CL_FILTER_NEAREST                           0x1140
#define global CL_FILTER_LINEAR                            0x1141

/* cl_sampler_info */
#define global CL_SAMPLER_REFERENCE_COUNT                  0x1150
#define global CL_SAMPLER_CONTEXT                          0x1151
#define global CL_SAMPLER_NORMALIZED_COORDS                0x1152
#define global CL_SAMPLER_ADDRESSING_MODE                  0x1153
#define global CL_SAMPLER_FILTER_MODE                      0x1154

/* cl_map_flags - bitfield */
#define global CL_MAP_READ                                 (1 << 0)
#define global CL_MAP_WRITE                                (1 << 1)
#define global CL_MAP_WRITE_INVALIDATE_REGION              (1 << 2)

/* cl_program_info */
#define global CL_PROGRAM_REFERENCE_COUNT                  0x1160
#define global CL_PROGRAM_CONTEXT                          0x1161
#define global CL_PROGRAM_NUM_DEVICES                      0x1162
#define global CL_PROGRAM_DEVICES                          0x1163
#define global CL_PROGRAM_SOURCE                           0x1164
#define global CL_PROGRAM_BINARY_SIZES                     0x1165
#define global CL_PROGRAM_BINARIES                         0x1166
#define global CL_PROGRAM_NUM_KERNELS                      0x1167
#define global CL_PROGRAM_KERNEL_NAMES                     0x1168

/* cl_program_build_info */
#define global CL_PROGRAM_BUILD_STATUS                     0x1181
#define global CL_PROGRAM_BUILD_OPTIONS                    0x1182
#define global CL_PROGRAM_BUILD_LOG                        0x1183
#define global CL_PROGRAM_BINARY_TYPE                      0x1184
    
/* cl_program_binary_type */
#define global CL_PROGRAM_BINARY_TYPE_NONE                 0x0
#define global CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT      0x1
#define global CL_PROGRAM_BINARY_TYPE_LIBRARY              0x2
#define global CL_PROGRAM_BINARY_TYPE_EXECUTABLE           0x4

/* cl_build_status */
#define global CL_BUILD_SUCCESS                            0
#define global CL_BUILD_NONE                               -1
#define global CL_BUILD_ERROR                              -2
#define global CL_BUILD_IN_PROGRESS                        -3

/* cl_kernel_info */
#define global CL_KERNEL_FUNCTION_NAME                     0x1190
#define global CL_KERNEL_NUM_ARGS                          0x1191
#define global CL_KERNEL_REFERENCE_COUNT                   0x1192
#define global CL_KERNEL_CONTEXT                           0x1193
#define global CL_KERNEL_PROGRAM                           0x1194
#define global CL_KERNEL_ATTRIBUTES                        0x1195

/* cl_kernel_arg_info */
#define global CL_KERNEL_ARG_ADDRESS_QUALIFIER             0x1196
#define global CL_KERNEL_ARG_ACCESS_QUALIFIER              0x1197
#define global CL_KERNEL_ARG_TYPE_NAME                     0x1198
#define global CL_KERNEL_ARG_TYPE_QUALIFIER                0x1199
#define global CL_KERNEL_ARG_NAME                          0x119A

/* cl_kernel_arg_address_qualifier */
#define global CL_KERNEL_ARG_ADDRESS_GLOBAL                0x119B
#define global CL_KERNEL_ARG_ADDRESS_LOCAL                 0x119C
#define global CL_KERNEL_ARG_ADDRESS_CONSTANT              0x119D
#define global CL_KERNEL_ARG_ADDRESS_PRIVATE               0x119E

/* cl_kernel_arg_access_qualifier */
#define global CL_KERNEL_ARG_ACCESS_READ_ONLY              0x11A0
#define global CL_KERNEL_ARG_ACCESS_WRITE_ONLY             0x11A1
#define global CL_KERNEL_ARG_ACCESS_READ_WRITE             0x11A2
#define global CL_KERNEL_ARG_ACCESS_NONE                   0x11A3
    
/* cl_kernel_arg_type_qualifer */
#define global CL_KERNEL_ARG_TYPE_NONE                     0
#define global CL_KERNEL_ARG_TYPE_CONST                    (1 << 0)
#define global CL_KERNEL_ARG_TYPE_RESTRICT                 (1 << 1)
#define global CL_KERNEL_ARG_TYPE_VOLATILE                 (1 << 2)

/* cl_kernel_work_group_info */
#define global CL_KERNEL_WORK_GROUP_SIZE                   0x11B0
#define global CL_KERNEL_COMPILE_WORK_GROUP_SIZE           0x11B1
#define global CL_KERNEL_LOCAL_MEM_SIZE                    0x11B2
#define global CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE 0x11B3
#define global CL_KERNEL_PRIVATE_MEM_SIZE                  0x11B4
#define global CL_KERNEL_GLOBAL_WORK_SIZE                  0x11B5

/* cl_event_info  */
#define global CL_EVENT_COMMAND_QUEUE                      0x11D0
#define global CL_EVENT_COMMAND_TYPE                       0x11D1
#define global CL_EVENT_REFERENCE_COUNT                    0x11D2
#define global CL_EVENT_COMMAND_EXECUTION_STATUS           0x11D3
#define global CL_EVENT_CONTEXT                            0x11D4

/* cl_command_type */
#define global CL_COMMAND_NDRANGE_KERNEL                   0x11F0
#define global CL_COMMAND_TASK                             0x11F1
#define global CL_COMMAND_NATIVE_KERNEL                    0x11F2
#define global CL_COMMAND_READ_BUFFER                      0x11F3
#define global CL_COMMAND_WRITE_BUFFER                     0x11F4
#define global CL_COMMAND_COPY_BUFFER                      0x11F5
#define global CL_COMMAND_READ_IMAGE                       0x11F6
#define global CL_COMMAND_WRITE_IMAGE                      0x11F7
#define global CL_COMMAND_COPY_IMAGE                       0x11F8
#define global CL_COMMAND_COPY_IMAGE_TO_BUFFER             0x11F9
#define global CL_COMMAND_COPY_BUFFER_TO_IMAGE             0x11FA
#define global CL_COMMAND_MAP_BUFFER                       0x11FB
#define global CL_COMMAND_MAP_IMAGE                        0x11FC
#define global CL_COMMAND_UNMAP_MEM_OBJECT                 0x11FD
#define global CL_COMMAND_MARKER                           0x11FE
#define global CL_COMMAND_ACQUIRE_GL_OBJECTS               0x11FF
#define global CL_COMMAND_RELEASE_GL_OBJECTS               0x1200
#define global CL_COMMAND_READ_BUFFER_RECT                 0x1201
#define global CL_COMMAND_WRITE_BUFFER_RECT                0x1202
#define global CL_COMMAND_COPY_BUFFER_RECT                 0x1203
#define global CL_COMMAND_USER                             0x1204
#define global CL_COMMAND_BARRIER                          0x1205
#define global CL_COMMAND_MIGRATE_MEM_OBJECTS              0x1206
#define global CL_COMMAND_FILL_BUFFER                      0x1207
#define global CL_COMMAND_FILL_IMAGE                       0x1208

/* command execution status */
#define global CL_COMPLETE                                 0x0
#define global CL_RUNNING                                  0x1
#define global CL_SUBMITTED                                0x2
#define global CL_QUEUED                                   0x3

/* cl_buffer_create_type  */
#define global CL_BUFFER_CREATE_TYPE_REGION                0x1220

/* cl_profiling_info  */
#define global CL_PROFILING_COMMAND_QUEUED                 0x1280
#define global CL_PROFILING_COMMAND_SUBMIT                 0x1281
#define global CL_PROFILING_COMMAND_START                  0x1282
#define global CL_PROFILING_COMMAND_END                    0x1283

































#module _HCLModule_
#defcfunc HCLGetDeviceInfo_s int a
	ptrk=0:szs=0
	HCLGetDeviceInfo a,ptrk,szs
	if szs!=0{
		dupptr64 out,ptrk,szs,2;2文字列型、ptrkは整数ポインタ、szsはサイズ、「out」はここで作成された新しい変数
		//この時点でoutは、C++側で定義している変数のポインタを参照している→今後内容が変わる可能性
		sdim retstr,szs
		retstr=out
		return retstr
	}else{
		return ""
	}

#defcfunc HCLGetDeviceInfo_i int a,int index
	ptrk=int64(0):szs=0
	HCLGetDeviceInfo a,ptrk,szs
	if szs!=0{
		dupptr64 out,ptrk,szs,4;4整数型、ptrkは整数ポインタ、szsはサイズ、変数名はここで作成された新しい変数
		return out.index
	}else{
		return 0
	}

#defcfunc HCLGetDeviceInfo_i64 int a,int index
	ptrk=int64(0):szs=0
	HCLGetDeviceInfo a,ptrk,szs
	if szs!=0{
		dupptr64 out64,ptrk,szs,8;8int64型、ptrkは整数ポインタ、szsはサイズ、変数名はここで作成された新しい変数
		return str(out64.index)
	}else{
		return str(int64(0))
	}

#defcfunc HCLGetKernelName var kernelid
	ptrk=int64(0):szs=0
	HCLGetKernelName_ kernelid,ptrk,szs
	if szs!=0{
		dupptr64 out,ptrk,szs,2;2文字列型、ptrkは整数ポインタ、szsはサイズ、「out」はここで作成された新しい変数
		//この時点でoutは、C++側で定義している変数のポインタを参照している→今後内容が変わる可能性
		sdim retstr,szs
		retstr=out
		return retstr
	}else{
		return ""
	}





#deffunc convRGBtoBGR array a,array b
	sizea=varsize(a)
	sizeb=varsize(b)
	if sizea>sizeb:sizea=sizeb
	_convRGBtoBGR a,b,sizea
	return
#deffunc convRGBAtoRGB array a,array b
	sizea=varsize(a)
	sizeb=(varsize(b)/3)*4
	if sizea>sizeb:sizea=sizeb
	_convRGBAtoRGB a,b,sizea
	return
#deffunc convRGBtoRGBA array a,array b,int toumeiflg
	if toumeiflg:tmpr=ginfo_r:tmpg=ginfo_g:tmpb=ginfo_b
	sizea=varsize(a)
	sizeb=(varsize(b)/4)*3
	if sizea>sizeb:sizea=sizeb
	if toumeiflg=0:_convRGBtoRGBA a,b,sizea,0,0,0,0:else:_convRGBtoRGBA a,b,sizea,1,tmpr,tmpg,tmpb
	return

#global