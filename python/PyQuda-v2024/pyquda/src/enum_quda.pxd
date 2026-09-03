cdef extern from "enum_quda.h":
    ctypedef enum qudaError_t:
        pass

    ctypedef enum QudaMemoryType:
        pass

    ctypedef enum QudaLinkType:
        pass

    ctypedef enum QudaGaugeFieldOrder:
        pass

    ctypedef enum QudaTboundary:
        pass

    ctypedef enum QudaPrecision:
        pass

    ctypedef enum QudaReconstructType:
        pass

    ctypedef enum QudaGaugeFixed:
        pass

    ctypedef enum QudaDslashType:
        pass

    ctypedef enum QudaInverterType:
        pass

    ctypedef enum QudaEigType:
        pass

    ctypedef enum QudaEigSpectrumType:
        pass

    ctypedef enum QudaSolutionType:
        pass

    ctypedef enum QudaSolveType:
        pass

    ctypedef enum QudaMultigridCycleType:
        pass

    ctypedef enum QudaSchwarzType:
        pass

    ctypedef enum QudaAcceleratorType:
        pass

    ctypedef enum QudaResidualType:
        pass

    ctypedef enum QudaCABasis:
        pass

    ctypedef enum QudaMatPCType:
        pass

    ctypedef enum QudaDagType:
        pass

    ctypedef enum QudaMassNormalization:
        pass

    ctypedef enum QudaSolverNormalization:
        pass

    ctypedef enum QudaPreserveSource:
        pass

    ctypedef enum QudaDiracFieldOrder:
        pass

    ctypedef enum QudaCloverFieldOrder:
        pass

    ctypedef enum QudaVerbosity:
        pass

    ctypedef enum QudaTune:
        pass

    ctypedef enum QudaPreserveDirac:
        pass

    ctypedef enum QudaParity:
        pass

    ctypedef enum QudaDiracType:
        pass

    ctypedef enum QudaFieldLocation:
        pass

    ctypedef enum QudaSiteSubset:
        pass

    ctypedef enum QudaSiteOrder:
        pass

    ctypedef enum QudaFieldOrder:
        pass

    ctypedef enum QudaFieldCreate:
        pass

    ctypedef enum QudaGammaBasis:
        pass

    ctypedef enum QudaSourceType:
        pass

    ctypedef enum QudaNoiseType:
        pass

    ctypedef enum QudaDilutionType:
        pass

    ctypedef enum QudaProjectionType:
        pass

    ctypedef enum QudaPCType:
        pass

    ctypedef enum QudaTwistFlavorType:
        pass

    ctypedef enum QudaTwistDslashType:
        pass

    ctypedef enum QudaTwistCloverDslashType:
        pass

    ctypedef enum QudaTwistGamma5Type:
        pass

    ctypedef enum QudaUseInitGuess:
        pass

    ctypedef enum QudaDeflatedGuess:
        pass

    ctypedef enum QudaComputeNullVector:
        pass

    ctypedef enum QudaSetupType:
        pass

    ctypedef enum QudaTransferType:
        pass

    ctypedef enum QudaBoolean:
        pass

    ctypedef enum QudaBLASType:
        pass

    ctypedef enum QudaBLASOperation:
        pass

    ctypedef enum QudaBLASDataType:
        pass

    ctypedef enum QudaBLASDataOrder:
        pass

    ctypedef enum QudaDirection:
        pass

    ctypedef enum QudaLinkDirection:
        pass

    ctypedef enum QudaFieldGeometry:
        pass

    ctypedef enum QudaGhostExchange:
        pass

    ctypedef enum QudaStaggeredPhase:
        pass

    ctypedef enum QudaSpinTasteGamma:
        pass

    ctypedef enum QudaContractType:
        pass

    ctypedef enum QudaFFTSymmType:
        pass

    ctypedef enum QudaContractGamma:
        pass

    ctypedef enum QudaGaugeSmearType:
        pass

    ctypedef enum QudaWFlowType:
        pass

    ctypedef enum QudaFermionSmearType:
        pass

    ctypedef enum QudaExtLibType:
        pass
