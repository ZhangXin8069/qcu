import cupy
import pyquda
EVEN, ODD = 0, 1
Nd, Ns, Nc = 4, 4, 3
X, Y, Z, T = 0, 1, 2, 3
BACKWARD, FORWARD = 0, 1


def eo_yzt(y: int, z: int, t: int) -> int:
    return (y + z + t) & 0x01


def move(DIM: int, WARD: int):
    if (DIM):
        return
    else:
        return
    pass


def move_parity(param: dict, src: cupy.ndarray, WARD: int):
    """
    LatticeGauge [Nd, 2, Lt, Lz, Ly, Lx//2, Nc, Nc] (Nd == 4, 2 -> even-odd, Nc == 3)
    LatticeFermion [2, Lt, Lz, Ly, Lx//2, Ns] (Ns == 4)
    """

    pass


def _run(param: dict, src: cupy.ndarray, dest: cupy.ndarray, U: cupy.ndarray, parity: int):
    pass


def run(param: dict, src: cupy.ndarray, dest: cupy.ndarray, U: cupy.ndarray):
    """
    param={'Lx':Lx, 'Ly':Ly, 'Lz':Lz, 'Lt':Lt}
    """
    _src = move_parity(param=param, src=src, WARD=ODD)
    _dest = move_parity(param=param, src=dest, WARD=EVEN)
    _run(param=param, src=_src, dest=_dest, U=U, parity=EVEN)
    _src = move_parity(param=param, src=src, WARD=EVEN)
    _dest = move_parity(param=param, src=dest, WARD=ODD)
    _run(param=param, src=_src, dest=_dest, U=U, parity=ODD)
    pass


if __name__ == "__main__":
    print(eo_yzt(1, 2, 2))
