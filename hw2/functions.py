from typing import List


def sum_non_neg_diag(X: List[List[int]]) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """ 
    sum = 0
    flag = False
    for i in range(len(X)):
        for j in range(len(X[0])):
            if i == j and X[i][j] >= 0:
                sum += X[i][j]
                flag = True
                break
    return sum if flag else -1


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    if (len(x) != len(y)):
        return False
    x, y = sorted(x), sorted(y)
    return x == y


def max_prod_mod_3(x: List[int]) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    max = 0
    pos_third = [index for index in range(len(x)) if x[index] % 3 == 0]
    if not pos_third or len(x) == 1:
        return -1
    for elem in pos_third:
        if elem != 0 and elem != len(x)-1:
            if x[elem] * x[elem - 1] > max:
                max = x[elem] * x[elem - 1]
            if x[elem] * x[elem + 1] > max:
                max = x[elem] * x[elem + 1]
        else:
            if elem == 0:
                if x[elem] * x[elem + 1] > max:
                    max = x[elem] * x[elem + 1]
            elif elem == len(x)-1:
                if x[elem] * x[elem - 2] > max:
                    max = x[elem] * x[elem - 2]
            
    return max if max != 0 else -1
def scal_prod(x,y):
    s=0
    for i in range(len(x)):
        s+=x[i]*y[i]
    return s
def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Сложить каналы изображения с указанными весами.
    """
    res = []
    count = 0
    for elem in image:
        res.append([])
        res[count] = [scal_prod(item,weights) for item in elem] 
        count += 1
    return res


def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    new_x = []
    for elem in x:
        for item in range(elem[1]):
            new_x.append(elem[0])
    new_y = []
    for elem in y:
        for item in range(elem[1]):
            new_y.append(elem[0])
    return scal_prod(new_x, new_y) if len(new_x) == len(new_y) else -1

def cosinus(x, y):
    num = scal_prod(x, y)
    den = 0
    for elem in x:
        den += elem*elem
    temp = 0
    for elem in y:
        temp += elem*elem
    denominator = (temp ** (1/2)) * (den ** (1/2))
    return num/denominator
def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y. 
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    mas = [0] * len(X) 
    for i in range(len(X)): 
        mas[i] = [0] * len(Y)
    for index in range(len(X)):
        zero = [0]*len(X[i])
        for j in range(len(Y)):
            mas[index][j] = cosinus(X[index], Y[j]) if (X[index] != zero and Y[j] != zero) else 1
    return mas

