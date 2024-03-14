import cupy
import pyquda

EVEN, ODD = 0, 1
Nd, Ns, Nc = 4, 4, 3
X, Y, Z, T = 0, 1, 2, 3
BACKWARD, FORWARD = 0, 1


def _Lxyzt(src: cupy.ndarray):
    return src.shape[:-2][::-1]


def eo_yzt(y: int, z: int, t: int) -> int:
    return (y + z + t) & 0x01


def move(DIM: int, WARD: int):
    if (DIM):
        return
    else:
        return
    pass


"""
LatticeGauge [Nd, 2, Lt, Lz, Ly, Lx//2, Nc, Nc] (Nd == 4, 2 -> even-odd, Nc == 3)
LatticeFermion [2, Lt, Lz, Ly, Lx//2, Ns, Nc] (Ns == 4)
"""


def _run(src: cupy.ndarray, dest: cupy.ndarray, U: cupy.ndarray, parity: int):
    Lxyzt = _Lxyzt(src=src)
    for x in range(Lxyzt[X]):
        for y in range(Lxyzt[Y]):
            for z in range(Lxyzt[Z]):
                for t in range(Lxyzt[T]):
                    # move()
                    eo = eo_yzt(y=y, z=z, t=t)

    pass


def run(src: cupy.ndarray, dest: cupy.ndarray, U: cupy.ndarray):
    """
    param={'Lx':Lx, 'Ly':Ly, 'Lz':Lz, 'Lt':Lt}
    """
    _run(src=src[ODD], dest=dest[EVEN], U=U, parity=EVEN)
    _run(src=src[EVEN], dest=dest[ODD], U=U, parity=ODD)
    pass


if __name__ == "__main__":
    print(eo_yzt(1, 2, 2))
