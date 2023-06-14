from hashlib import new
import numpy as np


def sum_non_neg_diag(X: np.ndarray) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    a = np.diag(X)
    sum1 = sum(a[a>=0])
    return sum1 if not np.any(a[a<0]) else -1


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    # ux, indicesx, countx = np.unique(x, return_index=True, return_counts=True)
    # uy, indicesy, county = np.unique(y, return_index=True, return_counts=True)
    # if len(countx) != len(county):
    #     return False
    # print(county, countx)
    # return (county == countx).all()
    return (np.sort(x) == np.sort(y)).all()


def max_prod_mod_3(x: np.ndarray) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    if len(x) <= 1:
        return -1
    xleft = np.roll(x, 1)
    xleft[0] = 0
    xright = np.roll(x, -1)
    xright[-1] = 0
    max1 = max((xleft*x)[(x * xleft) % 3 == 0], default=-1)
    max2 = max((xright*x)[(xright*x) % 3 == 0], default=-1)
    return  max(max1, max2) if max1 !=0 and  max2 != 0 else -1

def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Сложить каналы изображения с указанными весами.
    """
    # res = np.eye(image.shape[0], image.shape[1])
    # count = 0
    # for elem in image:
    #     res[count] = elem.dot(weights)
    #     count += 1
    # return np.apply_along_axis(weights.dot, -1, image)
    return np.dot(image, weights)
def myfunc(a):
    return np.array([a[0]]*a[1])
def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    newx = np.repeat(x[..., 0], x[..., 1])
    newy = np.repeat(y[..., 0], y[..., 1])
    return int(np.dot(newy, newx)) if newx.shape[0] == newy.shape[0] else -1

def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    norm1 = np.linalg.norm(X, axis=1)
    norm2 = np.linalg.norm(Y, axis=1)
    norm = np.outer(norm1, norm2)
    matrix = X.dot(Y.T) 
    return np.where(norm == 0, 1, matrix/norm)