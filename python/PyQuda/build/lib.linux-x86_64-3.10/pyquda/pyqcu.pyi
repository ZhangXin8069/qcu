from pointer import Pointer


class QcuParam:
    def __init__(self) -> None: ...

    lattice_size: int


def testDslashQcu(fermion_out: Pointer, fermion_in: Pointer,
                  gauge: Pointer, param: QcuParam, parity: int) -> None: ...


def dslashQcu(fermion_out: Pointer, fermion_in: Pointer,
              gauge: Pointer, param: QcuParam, parity: int) -> None: ...


def mpiDslashQcu(fermion_out: Pointer, fermion_in: Pointer, gauge: Pointer,
                 param: QcuParam, parity: int, grid: QcuParam) -> None: ...


def mpiCgQcu(fermion_out: Pointer, fermion_in: Pointer, gauge: Pointer,
             param: QcuParam, parity: int, grid: QcuParam) -> None: ...


def mpiBistabCgQcu(gauge: Pointer, param: QcuParam,
                   grid: QcuParam) -> None: ...
