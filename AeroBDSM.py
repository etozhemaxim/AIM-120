import ctypes
from inspect import Attribute
import os
import struct 

_filepath = os.path.dirname(__file__)

try:
    os.add_dll_directory(_filepath)
except AttributeError:
    pass

# Определение разрядности интерпретатора
bitness = struct.calcsize("P") * 8
#  Подключение DLL
if bitness == 64:
    libstructpy = ctypes.CDLL(_filepath + '/AeroBDSM_x64.dll')
else:
    libstructpy = ctypes.CDLL(_filepath + '/AeroBDSM.dll')

# Таблица сообщений об ошибках
ErrorMessageTable = {
    1: 'число должно быть действительным',
    2: 'число не должно быть отрицательным',
    3: 'число строк/столбцов должно быть больше либо равно 1',
    4: 'ошибка выделения памяти',
    5: 'вычисление прервано пользователем',
    6: 'этот аргумент может быть равен только 0 или 1',
    7: 'угол должен быть меньше 90 градусов',
    8: 'число не должно быть больше 1',
    9: 'этот аргумент не должен быть больше следующего аргумента',
    10: 'число не должно быть меньше 1',
    11: 'этот аргумент может быть равен только 0, 1 или 2',
    12: 'количество элементов в массиве не совпадает с количеством элементов в массиве M',
    13: 'количество элементов в массиве Ls должно быть на 1 больше, чем в Ls',
    14: 'несущие поверхности не должны пересекаться',
    15: 'больше половины бортовой хорды несущей поверхности вышло за носовую часть или за донный срез ЛА',
    16: 'Модуль числа должен быть больше либо равен 1',
    17: 'Значение аргумента недопустимо большое',
    18: 'При данных значениях аргументов использовать эту функцию нельзя',
    19: 'Этот аргумент не должен быть равен 0',
    20: 'угол должен быть меньше 180 градусов',
    21: 'угол должен быть меньше 360 градусов',
    22: 'угол должен быть больше -90 градусов'
}

# класс, который соответствует структуре InputComplex (AeroBDSM.h)
class InputComplex(ctypes.Structure):
    _fields_ = (
        ('Min', ctypes.c_float),
        ('Value', ctypes.c_float),
        ('Max', ctypes.c_float),
    )
    
    def __str__(self) -> str:
        return f'Min = {self.Min}; Value = {self.Value}; Max = {self.Max}'

    def __repr__(self) -> str:
        return self.__str__()

INPUT_COMPLEXES_MAXCOUNT = 9

# класс, который соответствует классу EAResult (AeroBDSM.h)
class EAResult(ctypes.Structure):
    _fields_ = (
        ('ErrorCode', ctypes.c_int),
        ('InvalidArgNumber', ctypes.c_int),
        ('Value', ctypes.c_float),
        ('ExtraValue', ctypes.c_float),
        ('InputComplexesCount', ctypes.c_int),
        ('InputComplexes', InputComplex * INPUT_COMPLEXES_MAXCOUNT),
    )

    def __float__(self):
        return self.Value
    
    def __str__(self) -> str:
        return f"Value = {self.Value}" + f"\nExtraValue = {self.ExtraValue}" + f"\nErrorCode = {self.ErrorCode}"  + f"\nInvalidArgNumber = {self.InvalidArgNumber}\n" + '\n'.join([f'x{i}: {self.InputComplexes[i]}' for i in range(self.InputComplexesCount)])

    def __repr__(self) -> str:
        return self.__str__()

    def __float__(self):        
        return self.Value

    def __lt__(self, other):        
        return self.Value < other

    def __le__(self, other):        
        return self.Value <= other

    def __eq__(self, other):        
        return self.Value == other

    def __ne__(self, other):        
        return self.Value != other

    def __gt__(self, other):        
        return self.Value > other

    def __ge__(self, other):        
        return self.Value >= other

    def __add__(self, other):        
        return self.Value + other

    def __sub__(self, other):        
        return self.Value - other

    def __mul__(self, other):        
        return self.Value * other

    def __truediv__(self, other):        
        return self.Value / other

    def __floordiv__(self, other):        
        return self.Value // other

    def __mod__(self, other):        
        return self.Value % other

    def __divmod__(self, other):        
        return divmod(self.Value,other)

    def __pow__(self, other):        
        return self.Value ** other

    def __radd__(self, other):        
        return other + self.Value 

    def __rsub__(self, other):        
        return other - self.Value

    def __rmul__(self, other):        
        return other * self.Value

    def __rtruediv__(self, other):        
        return other / self.Value

    def __rfloordiv__(self, other):        
        return other // self.Value

    def __rmod__(self, other):        
        return other % self.Value

    def __rdivmod__(self, other):        
        return divmod(other,self.Value)

    def __rpow__(self, other):        
        return other ** self.Value

 # класс, который соответствует классу GeomResult (AeroBDSM.h)
class GeomResult(ctypes.Structure):
    _fields_ = (
        ('ErrorCode', ctypes.c_int),
        ('InvalidArgNumber', ctypes.c_int),
        ('Value', ctypes.c_float),
        ('ExtraValue', ctypes.c_float),        
    )
    
    def __str__(self) -> str:
        return f"Value = {self.Value}" + f"\nExtraValue = {self.ExtraValue}" + f"\nErrorCode = {self.ErrorCode}"  + f"\nInvalidArgNumber = {self.InvalidArgNumber}\n"

    def __repr__(self) -> str:
        return self.__str__()

    def __repr__(self) -> str:
        return self.__str__()

    def __float__(self):        
        return self.Value

    def __lt__(self, other):        
        return self.Value < other

    def __le__(self, other):        
        return self.Value <= other

    def __eq__(self, other):        
        return self.Value == other

    def __ne__(self, other):        
        return self.Value != other

    def __gt__(self, other):        
        return self.Value > other

    def __ge__(self, other):        
        return self.Value >= other

    def __add__(self, other):        
        return self.Value + other

    def __sub__(self, other):        
        return self.Value - other

    def __mul__(self, other):        
        return self.Value * other

    def __truediv__(self, other):        
        return self.Value / other

    def __floordiv__(self, other):        
        return self.Value // other

    def __mod__(self, other):        
        return self.Value % other

    def __divmod__(self, other):        
        return divmod(self.Value,other)

    def __pow__(self, other):        
        return self.Value ** other

    def __radd__(self, other):        
        return other + self.Value 

    def __rsub__(self, other):        
        return other - self.Value

    def __rmul__(self, other):        
        return other * self.Value

    def __rtruediv__(self, other):        
        return other / self.Value

    def __rfloordiv__(self, other):        
        return other // self.Value

    def __rmod__(self, other):        
        return other % self.Value

    def __rdivmod__(self, other):        
        return divmod(other,self.Value)

    def __rpow__(self, other):        
        return other ** self.Value

class Vector(object):
    libstructpy.new_vector.restype = ctypes.c_void_p
    libstructpy.new_vector.argtypes = []
    libstructpy.delete_vector.restype = None
    libstructpy.delete_vector.argtypes = [ctypes.c_void_p]
    libstructpy.vector_size.restype = ctypes.c_int
    libstructpy.vector_size.argtypes = [ctypes.c_void_p]
    libstructpy.vector_get.restype = ctypes.c_float
    libstructpy.vector_get.argtypes = [ctypes.c_void_p, ctypes.c_int]
    libstructpy.vector_push_back.restype = None
    libstructpy.vector_push_back.argtypes = [ctypes.c_void_p, ctypes.c_float]

    def __init__(self, arr=None):
        # инициализация вектора
        self.vector = libstructpy.new_vector()

        # добавление элементов из arr в вектор
        if arr is not None:
            # проверяем, является ли arr iterable-объектом
            try:
                iter(arr)
            except:     # иначе, когда arr является обычным числом
                libstructpy.vector_push_back(self.vector, ctypes.c_float(arr))
            else:       # если arr - iterable-объект
                for val in arr:
                    libstructpy.vector_push_back(
                        self.vector, ctypes.c_float(val))

    # деструктор
    def __del__(self):
        libstructpy.delete_vector(self.vector)

    # длина элементов вектора
    def __len__(self):
        return libstructpy.vector_size(self.vector)

    # возвращает элемент вектора по индексу
    def __getitem__(self, ind):
        if 0 <= ind < len(self):
            return libstructpy.vector_get(self.vector, ctypes.c_int(ind))
        raise IndexError('Vector index out of range')

    # возвращает в виде строки список из элементов
    def __repr__(self):
        return '[{}]'.format(', '.join(str(self[i]) for i in range(len(self))))

    # добавляет элемент в конец вектора
    def push(self, val):
        return libstructpy.vector_push_back(self.vector, ctypes.c_float(val))


def get_A_IsP(M, zeta, c_y_alpha_IsP):
    """
    Функция для определения коэффициента дополнительной нормальной силы несущей поверхности при больших углах атаки

    Параметры
    ---------
    M : float
        число Маха, -
    zeta : float
        обратное сужение несущей поверхности, -
    c_y_alpha_IsP : float
        производная по углу атаки коэффициента нормальной силы изолированной несущей поверхности, 1/рад

    Возврат
    -------
    Value : float
        коэффициент дополнительной нормальной силы несущей поверхности при больших углах атаки, -
    x[0] : InputComplex
        входной комплекс c_y_alpha_IsP
    x[1] : InputComplex
        входной комплекс zeta
    x[2] : InputComplex
        входной комплекс M

    Ссылки
    ------
    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 188, Рис.3.35.)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_A_IsP.argtypes = [        
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    libstructpy.get_A_IsP.restype = EAResult

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_A_IsP(        
        ctypes.c_float(M),
        ctypes.c_float(zeta),
        ctypes.c_float(c_y_alpha_IsP),
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result


def get_bar_x_Falpha_IsP(M, lambd, chi_05, zeta):
    """
    Функция для определения относительной продольной координаты фокуса по углу атаки изолированной несущей поверхности

    Параметры
    ---------
    M : float
        число Маха, -
    lambd : float
        удлинение несущей поверхности, -
    chi_05 : float
        угол стреловидности по линии середин хорд, рад
    zeta : float
        обратное сужение несущей поверхности, -

    Возврат
    -------
    Value : float
        относительная продольная координата фокуса по углу атаки изолированной несущей поверхности, -
    x[0] : InputComplex
        входной комплекс lambd * sqrt(abs(sqr(M) - 1)) * sign(M - 1)
    x[1] : InputComplex
        входной комплекс zeta
    x[2] : InputComplex
        входной комплекс lambd * tan(chi_05)

    Ссылки
    ------
    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 265, Рис.5.8.)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_bar_x_Falpha_IsP.argtypes = [        
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    libstructpy.get_bar_x_Falpha_IsP.restype = EAResult
    
    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_bar_x_Falpha_IsP(        
        ctypes.c_float(M),
        ctypes.c_float(lambd),
        ctypes.c_float(chi_05),
        ctypes.c_float(zeta),
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result


def get_bar_z_Falpha_IsP(M, lambd, chi_05, zeta):
    """
    Функция для определения относительной поперечной координаты фокуса по углу атаки изолированной несущей поверхности

    Параметры
    ---------
    M : float
        число Маха, -
    lambd : float
        удлинение несущей поверхности, -
    chi_05 : float
        угол стреловидности по линии середин хорд, рад
    zeta : float
        обратное сужение несущей поверхности, -

    Возврат
    -------
    Value : float
        относительная поперечная координата фокуса по углу атаки изолированной несущей поверхности, -
    x[0] : InputComplex
        входной комплекс lambd * sqrt(abs(sqr(M) - 1)) * sign(M - 1)
    x[1] : InputComplex
        входной комплекс lambd * tan(chi_05)
    x[2] : InputComplex
        входной комплекс zeta

    Ссылки
    ------
    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 300, Рис.6.4.)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_bar_z_Falpha_IsP.argtypes = [        
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    libstructpy.get_bar_z_Falpha_IsP.restype = EAResult

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_bar_z_Falpha_IsP(        
        ctypes.c_float(M),
        ctypes.c_float(lambd),
        ctypes.c_float(chi_05),
        ctypes.c_float(zeta),
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result


def get_bar_z_v(M, lambd, chi_05, zeta):
    """
    Функция для определения относительной поперечной координаты вихря, сбегающего с консоли несущей поверхности

    Параметры
    ---------
    M : float
        число Маха, -
    lambd : float
        удлинение несущей поверхности, -
    chi_05 : float
        угол стреловидности по линии середин хорд, рад
    zeta : float
        обратное сужение несущей поверхности, -

    Возврат
    -------
    Value : float
        относительная поперечная координата вихря, сбегающего с консоли несущей поверхности, -
    x[0] : InputComplex
        входной комплекс lambd * sqrt(abs(sqr(M) - 1)) * sign(M - 1)
    x[1] : InputComplex
        входной комплекс lambd * tan(chi_05)
    x[2] : InputComplex
        входной комплекс zeta

    Ссылки
    ------
    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 168, Рис.3.16.)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_bar_z_v.argtypes = [        
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    libstructpy.get_bar_z_v.restype = EAResult

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_bar_z_v(        
        ctypes.c_float(M),
        ctypes.c_float(lambd),
        ctypes.c_float(chi_05),
        ctypes.c_float(zeta),
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result


def get_c_f0(Re, bar_x_t):
    """
    Функция для определения коэффициента трения плоской пластинки при М=0

    Параметры
    ---------
    Re : float
        число Рейнольдса, -
    bar_x_t : float
        относительная координата точки перехода ламинарного пограничного слоя в турбулентный (отсчитывается от передней точки, в долях длины пластинки), -

    Возврат
    -------
    Value : float
        коэффициент трения плоской пластинки при М=0, -
    x[0] : InputComplex
        входной комплекс Re
    x[1] : InputComplex
        входной комплекс bar_x_t

    Ссылки
    ------
    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 205, Рис.4.2.)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_c_f0.argtypes = [        
        ctypes.c_float,
        ctypes.c_float,
    ]
    libstructpy.get_c_f0.restype = EAResult

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_c_f0(        
        ctypes.c_float(Re),
        ctypes.c_float(bar_x_t),
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result


def get_c_x0_p_Cor_Con(M, lambda_Cor, zeta_Cor):
    """
    Функция для определения коэффициента сопротивления давления конической кормовой части при нулевом угле атаки

    Параметры
    ---------
    M : float
        число Маха, -
    lambda_Cor : float
        удлинение кормовой части, -
    zeta_Cor : float
        сужение кормовой части, -

    Возврат
    -------
    Value : float
        коэффициент сопротивления давления конической кормовой части при нулевом угле атаки, -
    x[0] : InputComplex
        входной комплекс M
    x[1] : InputComplex
        входной комплекс lambda_Cor
    x[2] : InputComplex
        входной комплекс zeta_Cor

    Ссылки
    ------
    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 226, Рис.4.24а.)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_c_x0_p_Cor_Con.argtypes = [
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    libstructpy.get_c_x0_p_Cor_Con.restype = EAResult

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_c_x0_p_Cor_Con(        
        ctypes.c_float(M),
        ctypes.c_float(lambda_Cor),
        ctypes.c_float(zeta_Cor),
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result


def get_c_x0_p_Nos_Con(M, lambda_Nos):
    """
    Функция для определения коэффициента сопротивления давления конической носовой части при нулевом угле атаки

    Параметры
    ---------
    M : float
        число Маха, -
    lambda_Nos : float
        удлинение носовой части, -

    Возврат
    -------
    Value : float
        коэффициент сопротивления давления конической носовой части при нулевом угле атаки, -
    x[0] : InputComplex
        входной комплекс M
    x[1] : InputComplex
        входной комплекс lambda_Nos

    Ссылки
    ------
    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 213, Рис.4.11.)

    Бураго, Корнев. Расчёт аэродинамических характеристик манёвренного ЛА, 2018 (с. 46, Рис.3.6.)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_c_x0_p_Nos_Con.argtypes = [        
        ctypes.c_float,
        ctypes.c_float,
    ]
    libstructpy.get_c_x0_p_Nos_Con.restype = EAResult

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_c_x0_p_Nos_Con(        
        ctypes.c_float(M),
        ctypes.c_float(lambda_Nos),
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result


def get_c_x0_p_Nos_Ell(M, lambda_Nos):
    """
    Функция для определения коэффициента сопротивления давления эллиптической носовой части при нулевом угле атаки

    Параметры
    ---------
    M : float
        число Маха, -
    lambda_Nos : float
        удлинение носовой части, -

    Возврат
    -------
    Value : float
        коэффициент сопротивления давления эллиптической носовой части при нулевом угле атаки, -
    x[0] : InputComplex
        входной комплекс M
    x[1] : InputComplex
        входной комплекс lambda_Nos

    Ссылки
    ------
    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 214, Рис.4.13.)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_c_x0_p_Nos_Ell.argtypes = [        
        ctypes.c_float,
        ctypes.c_float,
    ]
    libstructpy.get_c_x0_p_Nos_Ell.restype = EAResult

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_c_x0_p_Nos_Ell(        
        ctypes.c_float(M),
        ctypes.c_float(lambda_Nos),
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result

 
def get_c_x0_p_Nos_Par(M, lambda_Nos):
    """
    Функция для определения коэффициента сопротивления давления параболической носовой части при нулевом угле атаки

    Параметры
    ---------
    M : float
        число Маха, -
    lambda_Nos : float
        удлинение носовой части, -

    Возврат
    -------
    Value : float
        коэффициент сопротивления давления параболической носовой части при нулевом угле атаки, -
    x[0] : InputComplex
        входной комплекс M
    x[1] : InputComplex
        входной комплекс lambda_Nos

    Ссылки
    ------
    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 213, Рис.4.12.)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_c_x0_p_Nos_Par.argtypes = [        
        ctypes.c_float,
        ctypes.c_float,
    ]
    libstructpy.get_c_x0_p_Nos_Par.restype = EAResult

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_c_x0_p_Nos_Par(        
        ctypes.c_float(M),
        ctypes.c_float(lambda_Nos),
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result
 

def get_c_x0_w_IsP_Rmb(M, bar_c, zeta, chi_c, lambd):
    """
    Функция для определения коэффициента волнового сопротивления изолированной несущей поверхности с ромбовидным профилем при нулевом угле атаки

    Параметры
    ---------
    M : float
        число Маха, -
    bar_c : float
        относительная толщина профиля, -
    zeta : float
        обратное сужение несущей поверхности, -
    chi_c : float
        угол стреловидности по линии максимальных толщин, рад
    lambd : float
        удлинение несущей поверхности, -

    Возврат
    -------
    Value : float
        коэффициент волнового сопротивления изолированной несущей поверхности с ромбовидным профилем при нулевом угле атаки, -
    x[0] : InputComplex
        входной комплекс lambd * sqrt(sqr(M) - 1)
    x[1] : InputComplex
        входной комплекс lambd * pow(bar_c, 1/3)
    x[2] : InputComplex
        входной комплекс lambd * tan(chi_c)
    x[3] : InputComplex
        входной комплекс zeta

    Ссылки
    ------
    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 234, Рис.4.30.)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_c_x0_w_IsP_Rmb.argtypes = [        
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    libstructpy.get_c_x0_w_IsP_Rmb.restype = EAResult

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_c_x0_w_IsP_Rmb(        
        ctypes.c_float(M),
        ctypes.c_float(bar_c),
        ctypes.c_float(zeta),
        ctypes.c_float(chi_c),
        ctypes.c_float(lambd),
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result

    
def get_c_y_alpha_IsP(M, lambd, bar_c, chi_05, zeta):
    """
    Функция для определения производной по углу атаки коэффициента нормальной силы изолированной несущей поверхности

    Параметры
    ---------
    M : float
        число Маха, -
    lambd : float
        удлинение несущей поверхности, -
    bar_c : float
        относительная толщина профиля, -
    chi_05 : float
        угол стреловидности по линии середин хорд, рад
    zeta : float
        обратное сужение несущей поверхности, -

    Возврат
    -------
    Value : float
        производная по углу атаки коэффициента нормальной силы изолированной несущей поверхности, 1/рад
    x[0] : InputComplex
        входной комплекс lambd * sqrt(abs(sqr(M) - 1)) * sign(M - 1)
    x[1] : InputComplex
        входной комплекс lambd * pow(bar_c, 1/3)
    x[2] : InputComplex
        входной комплекс lambd * tan(chi_05)

    Ссылки
    ------
    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 156, Рис.3.5.)

    Егер, Мишин, Лисейцев. Проектирование самолётов, 1983 (с. 368, (14.15)-(14-18))
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_c_y_alpha_IsP.argtypes = [        
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    libstructpy.get_c_y_alpha_IsP.restype = EAResult

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_c_y_alpha_IsP(        
        ctypes.c_float(M),
        ctypes.c_float(lambd),
        ctypes.c_float(bar_c),
        ctypes.c_float(chi_05),
        ctypes.c_float(zeta),
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result


def get_c_y_alpha_NosCil_Con(M, lambda_Nos, lambda_Cil):
    """
    Функция для определения производной по углу атаки коэффициента нормальной силы комбинации конической носовой части и цилиндра

    Параметры
    ---------
    M : float
        число Маха, -
    lambda_Nos : float
        удлинение носовой части, -
    lambda_Cil : float
        удлинение цилиндрической части, -

    Возврат
    -------
    Value : float
        производная по углу атаки коэффициента нормальной силы комбинации конической носовой части и цилиндра, 1/рад
    x[0] : InputComplex
        входной комплекс sqrt(abs(sqr(M) - 1)) / lambda_Nos * sign(M - 1)
    x[1] : InputComplex
        входной комплекс lambda_Cil / lambda_Nos

    Ссылки
    ------
    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 153, Рис.3.2.)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_c_y_alpha_NosCil_Con.argtypes = [        
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,        
    ]
    libstructpy.get_c_y_alpha_NosCil_Con.restype = EAResult
    
    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_c_y_alpha_NosCil_Con(        
        ctypes.c_float(M),
        ctypes.c_float(lambda_Nos),
        ctypes.c_float(lambda_Cil),        
    ) 
   
    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result


def get_c_y_alpha_NosCil_Ell(M, lambda_Nos, lambda_Cil):
    """
    Функция для определения производной по углу атаки коэффициента нормальной силы комбинации эллиптической носовой части и цилиндра

    Параметры
    ---------
    M : float
        число Маха, -
    lambda_Nos : float
        удлинение носовой части (0 - плоская, 0.5 - сферическая), -
    lambda_Cil : float
        удлинение цилиндрической части, -

    Возврат
    -------
    Value : float
        производная по углу атаки коэффициента нормальной силы комбинации эллиптической носовой части и цилиндра, 1/рад
    x[0] : InputComplex
        входной комплекс sqrt(abs(sqr(M) - 1)) / lambda_Cil * sign(M - 1)
    x[1] : InputComplex
        входной комплекс lambda_Nos

    Ссылки
    ------
    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 154, Рис.3.4.)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_c_y_alpha_NosCil_Ell.argtypes = [        
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    libstructpy.get_c_y_alpha_NosCil_Ell.restype = EAResult

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_c_y_alpha_NosCil_Ell(        
        ctypes.c_float(M),
        ctypes.c_float(lambda_Nos),
        ctypes.c_float(lambda_Cil),  
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result


def get_c_y_alpha_NosCil_Par(M, lambda_Nos, lambda_Cil):
    """
    Функция для определения производной по углу атаки коэффициента нормальной силы комбинации параболической носовой части и цилиндра

    Параметры
    ---------
    M : float
        число Маха, -
    lambda_Nos : float
        удлинение носовой части, -
    lambda_Cil : float
        удлинение цилиндрической части, -

    Возврат
    -------
    Value : float
        производная по углу атаки коэффициента нормальной силы комбинации параболической носовой части и цилиндра, 1/рад
    x[0] : InputComplex
        входной комплекс sqrt(abs(sqr(M) - 1)) / lambda_Nos * sign(M - 1)
    x[1] : InputComplex
        входной комплекс lambda_Cil / lambda_Nos

    Ссылки
    ------
    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 154, Рис.3.3.)

    Краснов. Основы АД расчёта, 1981 (с. 212, Рис.2.7.4.)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_c_y_alpha_NosCil_Par.argtypes = [        
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    libstructpy.get_c_y_alpha_NosCil_Par.restype = EAResult

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_c_y_alpha_NosCil_Par(        
        ctypes.c_float(M),
        ctypes.c_float(lambda_Nos),
        ctypes.c_float(lambda_Cil),
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result


def get_c_yPerp_Cil(M_y):
    """
    Функция для определения коэффициента нормальной силы цилиндра бесконечной длины при обтекании перпендикулярно к его оси

    Параметры
    ---------
    M_y : float
        число Маха потока перпендикулярного к оси цилиндра, -

    Возврат
    -------
    Value : float
        коэффициент нормальной силы цилиндра бесконечной длины при обтекании перпендикулярно к его оси, -
    x[0] : InputComplex
        входной комплекс M_y

    Ссылки
    ------
    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 187, Рис.3.32.)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_c_yPerp_Cil.argtypes = [        
        ctypes.c_float,
    ]
    libstructpy.get_c_yPerp_Cil.restype = EAResult

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_c_yPerp_Cil(        
        ctypes.c_float(M_y),
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result


def get_Delta_bar_x_Falpha_NosCil(M, lambda_nos, lambda_cil):
    """
    Функция для определения смещения относительной продольной координаты фокуса по углу атаки комбинации носовой части и цилиндра

    Параметры
    ---------
    M : float
        число Маха, -
    lambda_nos : float
        удлинение носовой части, -
    lambda_cil : float
        удлинение цилиндрической части, -

    Возврат
    -------
    Value : float
        смещение относительной продольной координаты фокуса по углу атаки комбинации носовой части и цилиндра (в долях длины носовой части), -
    x[0] : InputComplex
        входной комплекс sqrt(abs(sqr(M) - 1)) / lambda_nos * sign(M - 1)
    x[1] : InputComplex
        входной комплекс lambda_cil / lambda_nos

    Ссылки
    ------
    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 264, Рис.5.7.)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_Delta_bar_x_Falpha_NosCil.argtypes = [        
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    libstructpy.get_Delta_bar_x_Falpha_NosCil.restype = EAResult

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_Delta_bar_x_Falpha_NosCil(        
        ctypes.c_float(M),
        ctypes.c_float(lambda_nos),
        ctypes.c_float(lambda_cil),
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result


def get_Delta_bar_z_Falpha_iC(bar_d):
    """
    Функция для определения смещения относительной поперечной координаты фокуса по углу атаки силы, индуцированной на консоли из-за влияния фюзеляжа

    Параметры
    ---------
    bar_d : float
        относительный диаметр фюзеляжа в области консоли, -

    Возврат
    -------
    Value : float
        смещение относительной поперечной координаты фокуса по углу атаки силы, индуцированной на консоли из-за влияния фюзеляжа, -
    x[0] : InputComplex
        входной комплекс bar_d

    Ссылки
    ------
    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 268, Рис.5.11.)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_Delta_bar_z_Falpha_iC.argtypes = [        
        ctypes.c_float,
    ]
    libstructpy.get_Delta_bar_z_Falpha_iC.restype = EAResult

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_Delta_bar_z_Falpha_iC(        
        ctypes.c_float(bar_d),
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result
    

def get_i_v(zeta, d, l, y_v, z_v):
    """
    Функция для определения коэффициента интерференции вихрей и задней несущей поверхности

    Параметры
    ---------
    zeta : float
        обратное сужение задней несущей поверхности, -
    d : float
        диаметр фюзеляжа в области задней несущей поверхности, м
    l : float
        размах задней несущей поверхности (включая подфюзеляжную часть), м
    y_v : float
        нормальная координата вихря (отсчитывается от плоскости задней несущей поверхности по нормали), м
    z_v : float
        поперечная координата вихря (отсчитывается от оси фюзеляжа вдоль размаха задней несущей поверхности), м

    Возврат
    -------
    Value : float
        коэффициент интерференции вихрей и задней несущей поверхности, -
    x[0] : InputComplex
        входной комплекс 2 * z_v / l
    x[1] : InputComplex
        входной комплекс 2 * y_v / l
    x[2] : InputComplex
        входной комплекс d_II / l
    x[3] : InputComplex
        входной комплекс zeta

    Ссылки
    ------
    NACA Report 1307 (Chart 7)

    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 169, Рис.3.17.)

    Широкопетлев. Оцифровка графических зависимостей, представленных в виде изолиний // Тезисы докладов Студенческой научной весны, 2022. (с. 99)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_i_v.argtypes = [        
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    libstructpy.get_i_v.restype = EAResult

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_i_v(        
        ctypes.c_float(zeta),
        ctypes.c_float(d),
        ctypes.c_float(l),
        ctypes.c_float(y_v),
        ctypes.c_float(z_v),
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result



def get_kappa_q_IsP(M, L_A, b_A):
    """
    Функция для определения коэффициента торможения потока, вызванного обтеканием несущей поверхности

    Параметры
    ---------
    M : float
        число Маха, -
    L_A : float
        расстояние от задней точки средней аэродинамической хорды (измеренное вдоль хорды), м
    b_A : float
        средняя аэродинамическая хорда, м

    Возврат
    -------
    Value : float
        коэффициент торможения потока, вызванного обтеканием несущей поверхности, -
    x[0] : InputComplex
        входной комплекс M
    x[1] : InputComplex
        входной комплекс L_A / b_A

    Ссылки
    ------
    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 175, Рис.3.22.)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_kappa_q_IsP.argtypes = [        
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    libstructpy.get_kappa_q_IsP.restype = EAResult

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_kappa_q_IsP(        
        ctypes.c_float(M),
        ctypes.c_float(L_A),
        ctypes.c_float(b_A),
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result


def get_kappa_q_Nos_Con(M, lambda_Nos):
    """
    Функция для определения коэффициента торможения потока, вызванного обтеканием конической носовой части

    Параметры
    ---------
    M : float
        число Маха, -
    lambda_Nos : float
        удлинение носовой части, -

    Возврат
    -------
    Value : float
        коэффициент торможения потока, вызванного обтеканием конической носовой части, -
    x[0] : InputComplex
        входной комплекс lambda_Nos
    x[1] : InputComplex
        входной комплекс M

    Ссылки
    ------
    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 174, Рис.3.21.)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_kappa_q_Nos_Con.argtypes = [        
        ctypes.c_float,
        ctypes.c_float,
    ]
    libstructpy.get_kappa_q_Nos_Con.restype = EAResult

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_kappa_q_Nos_Con(        
        ctypes.c_float(M),
        ctypes.c_float(lambda_Nos),
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result



def get_sigma_c_Prf(bar_c, bar_x_t):
    """
    Функция для определения поправочного множителя, учитывающего влияние толщины профиля на коэффициент профильного сопротивления

    Параметры
    ---------
    bar_c : float
        относительная толщина профиля, -
    bar_x_t : float
        относительная координата точки перехода ламинарного пограничного слоя в турбулентный (отсчитывается от передней точки хорды, в долях хорды), -

    Возврат
    -------
    Value : float
        поправочный множитель, учитывающий влияние толщины профиля на коэффициент профильного сопротивления, -
    x[0] : InputComplex
        входной комплекс bar_c
    x[1] : InputComplex
        входной комплекс bar_x_t

    Ссылки
    ------
    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 232, Рис.4.28.)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_sigma_c_Prf.argtypes = [        
        ctypes.c_float,
        ctypes.c_float,
    ]
    libstructpy.get_sigma_c_Prf.restype = EAResult

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_sigma_c_Prf(        
        ctypes.c_float(bar_c),
        ctypes.c_float(bar_x_t),
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result


def get_sigma_cp_Nos_Con(M, lambda_Nos):
    """
    Функция для определения поправочного множителя, учитывающего форму носовой части и число Маха при расчёте коэффициента сопротивления давления конической носовой части при ненулевых углах атаки

    Параметры
    ---------
    M : float
        число Маха, -
    lambda_Nos : float
        удлинение носовой части, -    

    Возврат
    -------
    Value : float
        множитель, учитывающий форму носовой части и число Маха при расчёте коэффициента сопротивления давления конической носовой части при ненулевых углах атаки, -
    x[0] : InputComplex
        входной комплекс sqrt(abs(sqr(M) - 1)) / lambda_Nos * sign(M - 1)

    Ссылки
    ------
    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 245, Рис.4.40.)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_sigma_cp_Nos_Con.argtypes = [        
        ctypes.c_float,
        ctypes.c_float,
    ]
    libstructpy.get_sigma_cp_Nos_Con.restype = EAResult

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_sigma_cp_Nos_Con(        
        ctypes.c_float(M),
        ctypes.c_float(lambda_Nos),
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result

def get_sigma_cp_Nos_Par(M, lambda_Nos):
    """
    Функция для определения поправочного множителя, учитывающего форму носовой части и число Маха при расчёте коэффициента сопротивления давления параболической носовой части при ненулевых углах атаки

    Параметры
    ---------
    M : float
        число Маха, -
    lambda_Nos : float
        удлинение носовой части, -    

    Возврат
    -------
    Value : float
        множитель, учитывающий форму носовой части и число Маха при расчёте коэффициента сопротивления давления параболической носовой части при ненулевых углах атаки, -
    x[0] : InputComplex
        входной комплекс sqrt(abs(sqr(M) - 1)) / lambda_Nos * sign(M - 1)

    Ссылки
    ------
    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 245, Рис.4.40.)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_sigma_cp_Nos_Par.argtypes = [        
        ctypes.c_float,
        ctypes.c_float,
    ]
    libstructpy.get_sigma_cp_Nos_Par.restype = EAResult

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_sigma_cp_Nos_Par(        
        ctypes.c_float(M),
        ctypes.c_float(lambda_Nos),
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result


def get_sigma_f_M(M, bar_x_t):
    """
    Функция для определения поправочного множителя, учитывающего влияние числа Маха на коэффициент трения плоской пластинки

    Параметры
    ---------
    M : float
        число Маха, -
    bar_x_t : float
        относительная координата точки перехода ламинарного пограничного слоя в турбулентный (отсчитывается от передней точки, в долях длины пластинки), -

    Возврат
    -------
    Value : float
        поправочный множитель, учитывающий влияние числа Маха на коэффициент трения плоской пластинки, -
    x[0] : InputComplex
        входной комплекс M
    x[1] : InputComplex
        входной комплекс bar_x_t

    Ссылки
    ------
    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 205, Рис.4.3.)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_sigma_f_M.argtypes = [        
        ctypes.c_float,
        ctypes.c_float,
    ]
    libstructpy.get_sigma_f_M.restype = EAResult

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_sigma_f_M(        
        ctypes.c_float(M),
        ctypes.c_float(bar_x_t),
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result


def get_sigma_zeta_Cor(M, lambda_Cor, zeta_Cor):
    """
    Функция для определения поправочного множителя, учитывающего влияние сужающейся кормовой части на коэффициент донного сопротивления фюзеляжа

    Параметры
    ---------
    M : float
        число Маха, -
    lambda_Cor : float
        удлинение кормовой части, -
    zeta_Cor : float
        обратное сужение кормовой части, -

    Возврат
    -------
    Value : float
        поправочный множитель, учитывающий влияние сужающейся кормовой части на коэффициент донного сопротивления фюзеляжа, -
    x[0] : InputComplex
        входной комплекс (1 - zeta_Cor) / (2 * lambda_Cor * sqr(zeta_Cor))
    x[1] : InputComplex
        входной комплекс M

    Ссылки
    ------
    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 230, Рис.4.27.)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_sigma_zeta_Cor.argtypes = [        
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    libstructpy.get_sigma_zeta_Cor.restype = EAResult

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_sigma_zeta_Cor(        
        ctypes.c_float(M),
        ctypes.c_float(lambda_Cor),
        ctypes.c_float(zeta_Cor),
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result


    
def get_psi_eps(M_I, alpha_p, phi_alpha, psi_I, psi_II, z_v, y_v, L_vI_bII, d_II, l_1c_II, zeta_II, b_b_II, chi_0_II):
    """
    Функция для определения относительной площади несущей поверхности, подверженной скосу потока

    Параметры
    ---------
    M_I : float
        число Маха набегающего потока в области I несущей поверхности, -
    alpha_p : float 
        пространственный угол атаки летательного аппарата, рад
	phi_alpha : float
        аэродинамический угол крена летательного аппарата, рад
	psi_I : float
        угол расположения консоли I несущей поверхности в поперечной плоскости, рад
	psi_II : float
        угол расположения консоли II несущей поверхности в поперечной плоскости, рад
    z_v : float
        координата z точки схода вихря с передней консоли (отсчитывается от бортовой хорды вдоль размаха передней консоли), м
    y_v : float
        координата y точки схода вихря с передней консоли (отсчитывается по нормали от плоскости неотклонённой передней консоли), м
    L_vI_bII : float
        расстояние вдоль оси Ox летательного аппарата от точки схода вихря до передней точки бортовой хорды II несущей поверхности, м
    d_II : float
        диаметр фюзеляжа в области II несущей поверхности, м
    l_1c_II : float
        размах одной консоли II несущей поверхности, м
    zeta_II : float
        обратное сужение II несущей поверхности, -
    b_b_II : float
        длина бортовой хорды II несущей поверхности, м
    chi_0_II : float
        угол стреловидности по передней кромке II несущей поверхности, рад

    Возврат
    -------
    Value : float
        относительная площадь несущей поверхности, подверженная скосу потока, -
    ExtraValue : float
        Тип кривой. Параметр не участвует ни в каких расчетах и предназначен тольлко для контроля ошибок расчетов внутри фунции.
        Классификацию кривых см. в примечании к функции в Руководстве.

    Ссылки
    ------
    Лазарев, Лаптева, Тищенко, Калугин. Методика расчёта относительной площади... // Известия РАРАН, 2023 (№4, с. 65-71)

    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 172)
    """
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_psi_eps.argtypes = [        
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double
    ]
    libstructpy.get_psi_eps.restype = GeomResult

     #  Вызов функций, описанных в DLL
    Result = libstructpy.get_psi_eps(        
        ctypes.c_double(M_I),
        ctypes.c_double(alpha_p),
        ctypes.c_double(phi_alpha),
        ctypes.c_double(psi_I),
        ctypes.c_double(psi_II),
        ctypes.c_double(z_v),
        ctypes.c_double(y_v),
        ctypes.c_double(L_vI_bII),
        ctypes.c_double(d_II),
        ctypes.c_double(l_1c_II),
        ctypes.c_double(zeta_II),
        ctypes.c_double(b_b_II),
        ctypes.c_double(chi_0_II)
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result


def get_S_b_F(S_b_Nos, ds, Ls, Ms, b_bs, L_hvs):
    """
    Функция для определения незатенённой площади боковой проекции фюзеляжа

    Параметры
    ---------
    S_b_Nos : float
        площадь боковой проекции носовой части, м^2
    ds : list[float]
        массив диаметров на границах частей фюзеляжа (начиная с диаметра основания носовой части и заканчивая диаметром донного среза), м
    Ls : list[float]
        массив длин частей фюзеляжа (кроме носовой части), м
    Ms : list[float]
        массив чисел Маха набегающего потока на каждую несущую поверхность, -
    b_bs : list[float]
        массив длин бортовой хорды несущих поверхностей, м
    L_hvs : list[float]
        массив длин хвостовых частей (расстояний от конца бортовой хорды несущей поверхности до донного среза фюзеляжа), м

    Возврат
    -------
    Value : float
        площадь незатенённой области боковой проекции фюзеляжа, м^2

    Ссылки
    ------
    Лазарев, Лаптева, Тищенко. Расчёт площади и координаты центра тяжести боковой проекции... // Известия ТулГУ, 2022 (№12, с. 73 - 79)

    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 185 - 186)
    """
    ds = Vector(ds)
    Ls = Vector(Ls)
    Ms = Vector(Ms)
    b_bs = Vector(b_bs)
    L_hvs = Vector(L_hvs)

    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_S_b_F.argtypes = [        
        ctypes.c_float,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    libstructpy.get_S_b_F.restype = GeomResult  

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_S_b_F(        
        ctypes.c_float(S_b_Nos),
        ds.vector,
        Ls.vector,
        Ms.vector,
        b_bs.vector,
        L_hvs.vector,
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result


def get_x_cS_F(S_b_Nos, L_Nos, x_cS_Nos, ds, Ls, Ms, b_bs, L_hvs):
    """
    Функция для определения координаты x центра тяжести незатенённой площади боковой проекции фюзеляжа

    Параметры
    ---------
    S_b_Nos : float
        площадь боковой проекции носовой части, м^2
    L_Nos : float
        Длина носовой части, м
    x_cS_Nos : float
        координата x центра тжести площади боковой проекции носовой части S_nos, м
    ds : list[float]
        массив диаметров на границах частей фюзеляжа (начиная с диаметра основания носовой части и заканчивая диаметром донного среза), м
    Ls : list[float]
        массив длин частей фюзеляжа (кроме носовой части), м
    Ms : float
        массив чисел Маха набегающего потока на каждую несущую поверхность, -
    b_bs : float
        массив длин бортовой хорды несущих поверхностей, м
    L_hvs : float
        массив длин хвостовых частей (расстояний от конца бортовой хорды несущей поверхности до донного среза фюзеляжа), м

    Возврат
    -------
    Value : float
        координата x центра тяжести незатенённой площади боковой проекции фюзеляжа, м

    Ссылки
    ------
    Лазарев, Лаптева, Тищенко. Расчёт площади и координаты центра тяжести боковой проекции... // Известия ТулГУ, 2022 (№12, с. 73 - 79)

    Лебедев, Чернобровкин. Динамика полёта БПЛА, 1973 (с. 271)
    """
    ds = Vector(ds)
    Ls = Vector(Ls)
    Ms = Vector(Ms)
    b_bs = Vector(b_bs)
    L_hvs = Vector(L_hvs)
    
    #  Указание типов аргументов и возвращаемого значения
    libstructpy.get_x_cS_F.argtypes = [        
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    libstructpy.get_x_cS_F.restype = GeomResult

    #  Вызов функций, описанных в DLL
    Result = libstructpy.get_x_cS_F(        
        ctypes.c_float(S_b_Nos),
        ctypes.c_float(L_Nos),
        ctypes.c_float(x_cS_Nos),
        ds.vector,
        Ls.vector,
        Ms.vector,
        b_bs.vector,
        L_hvs.vector
    )

    if Result.ErrorCode in ErrorMessageTable:
        raise ValueError(
            f'Аргумент {Result.InvalidArgNumber}, {ErrorMessageTable[Result.ErrorCode]}.')
    else:
        return Result

def get_Info():
    """
    Функция для получения информации о библиотеке
    """   
    libstructpy.get_Info.argtypes = None   
    libstructpy.get_Info.restype = ctypes.c_char_p

    #  Вызов функций, описанных в DLL
    return libstructpy.get_Info()
