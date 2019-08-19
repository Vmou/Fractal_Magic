from PIL import Image
import numba
import numpy as np

MAX_ITER = 1000
MAX_RADIUS = 2
SIZE_Image = 500

Z = complex(0, 0)

START_POINT = (-0.761576,-0.0847596)
ZOOM = 1e3
RANGE = 1 / ZOOM

START_X = START_POINT[0] - RANGE
END_X = START_POINT[0] + RANGE
STEP_X = (END_X - START_X) / SIZE_Image

START_Y = START_POINT[1] + RANGE
END_Y = START_POINT[1] - RANGE
STEP_Y = (END_Y - START_Y) / SIZE_Image

BLACK = (0,0,0)
WHITE = (255,255,255)

# f(z) = z ** 2 + c, z0 = 0
@numba.jit(nopython=True)
def calc(z, c):
    return z ** 2 + c

@numba.jit(nopython=True)
def explode(c):
    z = Z
    for i in range(MAX_ITER):
        z = calc(z, c)
        if abs(z) > MAX_RADIUS:
            return i
    return i


@numba.jit(nopython=True)
def loop(arr):
    j = START_Y
    for y in range(SIZE_Image):
        i = START_X
        for x in range(SIZE_Image):
            c = complex(i, j)
            e = explode(c)
            if e < MAX_ITER:
                color = round(e*255)/MAX_ITER
                arr[y, x] = (color, color, color)
            else:
                arr[y, x] = (0, 0, 0)
            i += STEP_X
        j += STEP_Y


def run():
    arr = np.empty([SIZE_Image,SIZE_Image, 3])
    loop(arr)
    img = Image.fromarray(np.uint8(arr))
    img.save("result.jpg", "JPEG")

if __name__ == "__main__":
    run()