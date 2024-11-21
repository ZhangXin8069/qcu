cdef extern from "quda_constants.h":
    cdef enum:
        QUDA_VERSION_MAJOR

        QUDA_VERSION_MINOR

        QUDA_VERSION_SUBMINOR

        QUDA_VERSION

        QUDA_MAX_DIM

        QUDA_MAX_GEOMETRY

        QUDA_MAX_MULTI_SHIFT

        QUDA_MAX_BLOCK_SRC

        QUDA_MAX_ARRAY_SIZE

        QUDA_MAX_DWF_LS

        QUDA_MAX_MG_LEVEL
