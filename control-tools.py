from math import prod, ceil
import control
import numpy as np
import matplotlib.pyplot as plt
from mpmath import findroot
import scipy as sp
from sympy import symbols, simplify, Poly, Expr, im, re
from sympy.parsing.sympy_parser import parse_expr
from sympy.solvers.inequalities import reduce_rational_inequalities


def derivar(tf: control.TransferFunction):
    num = np.poly(tf.zero())
    den = np.poly(tf.pole())

    try:
        num_d = np.array([num[i] * (len(num) - i - 1)
                          for i in range(0, len(num) - 1)])
    except:
        num_d = 0
    try:
        den_d = np.array([den[i] * (len(den) - i - 1)
                          for i in range(0, len(den) - 1)])
    except:
        den_d = 0

    derivada = (control.tf(num_d, 1) * control.tf(den, 1) - control.tf(den_d, 1)
                * control.tf(num, 1))  # / (control.tf(den, 1) * control.tf(den, 1))

    return derivada


def stability_test(tf: control.TransferFunction):
    'Realizar todos los calculos que pide Fede Val'

    print(tf)

    print(f'Numero de asíntotas: {str(len(tf.pole())-len(tf.zero()))}')

    if(len(tf.pole())-len(tf.zero()) != 0):
        print(
            f'Ángulo de las asíntotas: {", ".join([str(round((2 * r - 1)/(len(tf.pole())-len(tf.zero())), 2)) + "π" for r in range(1, len(tf.pole())-len(tf.zero())+1)])}')

        print(
            f'Corte de las asíntotas con el eje real: {str(round(((sum(tf.pole())-sum(tf.zero()))/(len(tf.pole())-len(tf.zero()))).real, 2))}')

        print(
            f'Ganancia(s) K para el cruce por el eje imaginario: {", ".join([str(i) for i in control.stability_margins(tf, returnall=True)[0]])}')

        print(
            f'Frecuencia(s) de corte del eje imaginario: {", ".join([f"±{i}j" for i in control.stability_margins(tf, returnall=True)[3]])}')

    puntos_de_quiebre = []
    tf_d = derivar(tf)
    for i in range(-100, 0):
        try:
            temp = str(
                round(float(str(findroot(lambda w: control.evalfr(tf_d, w), i).real)), 3))
            if(temp not in puntos_de_quiebre):
                puntos_de_quiebre += [temp]
        except:
            pass

    print(
        f'Punto(s) de quiebre del eje real: {", ".join(puntos_de_quiebre)}')

    try:
        print(
            f'Márgen de ganacia y wmg: {round(control.stability_margins(tf, returnall=True)[0][0], 2)}, {round(control.stability_margins(tf, returnall=True)[3][0], 2)} rad/s')
    except:
        print('No existe el márgen de ganancia')

    try:
        print(
            f'Márgen de fase y wmf: {round(control.stability_margins(tf, returnall=True)[1][0], 2)}º, {round(control.stability_margins(tf, returnall=True)[4][0], 2)} rad/s')
    except:
        print('No existe el márgen de fase')

    plt.figure(0).canvas.manager.set_window_title('Root Locus')
    try:
        control.root_locus(tf, print_gain=True)
    except:
        print('No se puede calcular el lugar de raices ya que la función no es compatible con sistemas de orden 0')
    plt.figure(1).canvas.manager.set_window_title('Nyquist')
    control.nyquist_plot(tf)
    plt.figure(2).canvas.manager.set_window_title('Bode')
    control.bode_plot(tf, dB=True)
    plt.show()


def partial_fractions(tf: control.TransferFunction):
    'Realizar la expansion en fracciones parciales y obtener sus terminos'

    num = tf.num[0][0]
    den = tf.den[0][0]

    r, p, k = sp.signal.residue(num, den)

    fractions = []
    # pi_old debe iniciarse en un numero distinto a p[0]
    pi_old: float = p[0] + 100000
    p_count: int = 1

    for ri, pi in zip(r, p):

        if (pi == pi_old):
            p_count += 1
        else:
            p_count = 1
            pi_old = pi

        fractions.append(control.tf(
            [ri], np.poly([pi for j in range(p_count)])))

    for ki, i in zip(k, range(len(k))):
        fractions.append(ki *
                         control.tf([j+1 if j == 0 else 0 for j in range(len(k)-i)], [1]))

    num = ''
    bar = ''
    den = ''
    for i in fractions:
        i = str(i).split('\n')[1:-1]
        if(i[2].strip() != '1'):
            bar += i[1] + ' + '
            num += i[0] + ' ' * (len(i[1]) - len(i[0]) + 3)
            den += i[2] + ' ' * (len(i[1]) - len(i[2]) + 3)
        else:
            bar += i[0] + '   '
            num += ' ' * len(bar)
            den += ' ' * len(bar)

    num = num[:-3]
    bar = bar[:-3]
    den = den[:-3]

    print('Expansión en fracciones parciales:')
    print(f'\n{num}\n{bar}\n{den}\n')
    return fractions


def factorize(tf: control.TransferFunction):
    'Factoriza una funcion de transferencia'
    tf = tf.minreal()
    zeros = tf.zero().tolist()
    poles = tf.pole().tolist()

    if(control.isctime(tf)):
        sym = symbols('s')
    else:
        sym = symbols('z')

    num = parse_expr('1')
    den = parse_expr('1')

    zeros = [i for i in zeros if im(i) >= 0]
    poles = [i for i in poles if im(i) >= 0]

    for zero in zeros:
        if(im(zero) == 0):
            num *= sym - zero
        else:
            num *= sym ** 2 - 2 * sym * re(zero) + abs(zero) ** 2
    for pole in poles:
        if(im(pole) == 0):
            den *= sym - pole
        else:
            den *= sym ** 2 - 2 * sym * re(pole) + abs(pole) ** 2

    dcgain = str(tf.num[0][0][0] / tf.den[0][0][0])
    F = str(num / den).replace('**', '^').replace('*', '')[:-1].split('/(')
    F_0 = ' ' * (len(dcgain) + 1) + F[0].center(max(len(F[0]), len(F[1])))
    F_1 = dcgain + ' ' + '-' * max(len(F[0]), len(F[1]))
    F_2 = ' ' * (len(dcgain) + 1) + F[1].center(max(len(F[0]), len(F[1])))
    F = f'\n{F_0}\n{F_1}\n{F_2}\n'

    return F


def z_to_w(tf: control.TransferFunction):
    if(tf.isctime()):
        print('Estas intentando pasar al dominio w una función de transferencia continua')
        return 0
    w = symbols('w')
    z = (1 + (tf.dt / 2) * w) / (1 - (tf.dt / 2) * w)
    num = tf.num[0][0]
    den = tf.den[0][0]
    tf_w_num = sum([(num[i] * z ** (len(num) - i - 1))
                   for i in range(len(num))])
    tf_w_den = sum([(den[i] * z ** (len(den) - i - 1))
                   for i in range(len(den))])
    tf_w = tf_w_num / tf_w_den
    tf_w = simplify(tf_w)
    tf_w_den = Poly(prod([tf_w.args[i] ** -1 for i in range(len(tf_w.args))
                          if tf_w.args[i].args[1] == parse_expr('-1')])).all_coeffs()
    tf_w_num = Poly(prod([tf_w.args[i] for i in range(len(
        tf_w.args)) if tf_w.args[i].args[1] != parse_expr('-1')])).all_coeffs()
    tf_w = control.tf([float(i)/float(tf_w_den[0]) for i in tf_w_num], [float(i)/float(tf_w_den[0])
                      for i in tf_w_den])

    print('Transformación al dominio w: [ z=(1+(T/2)*w)/(1-(T/2)*w) ]')
    print(str(tf_w).replace('s', 'w'))
    return tf_w


def w_to_z(tf: control.TransferFunction, T: float):
    z = symbols('z')
    w = (2 / T) * (z - 1) / (z + 1)
    num = tf.num[0][0]
    den = tf.den[0][0]
    tf_z_num = sum([(num[i] * w ** (len(num) - i - 1))
                   for i in range(len(num))])
    tf_z_den = sum([(den[i] * w ** (len(den) - i - 1))
                   for i in range(len(den))])
    tf_z = tf_z_num / tf_z_den
    tf_z = simplify(tf_z)
    tf_z_den = Poly(prod([tf_z.args[i] ** -1 for i in range(len(tf_z.args))
                          if tf_z.args[i].args[1] == parse_expr('-1')])).all_coeffs()
    tf_z_num = Poly(prod([tf_z.args[i] for i in range(len(
        tf_z.args)) if tf_z.args[i].args[1] != parse_expr('-1')])).all_coeffs()
    tf_z = control.tf([float(i)/float(tf_z_den[0]) for i in tf_z_num], [float(i)/float(tf_z_den[0])
                      for i in tf_z_den], dt=T)

    print('Transformación del dominio w a z: [ w=(2/T)*(z-1)/(z+1) ]')
    print(str(tf_z).replace('dt', ' T')[:-1] + ' seg\n')
    return tf_z


def s_to_z(tf: control.TransferFunction, T: float, verbose=True):
    if(not verbose):
        tf_z = control.sample_system(tf, T)
        print(tf_z)
        return tf_z

    print('Funcion de transferencia en s:')
    print(tf)

    fractions = partial_fractions(tf)

    fractions = map(lambda x: control.sample_system(x, T), fractions)

    num = ''
    bar = ''
    den = ''
    for i in fractions:
        i = str(i).split('\n')[1:-3]
        if(i[2].strip() != '1'):
            bar += i[1] + ' + '
            num += i[0] + ' ' * (len(i[1]) - len(i[0]) + 3)
            den += i[2] + ' ' * (len(i[1]) - len(i[2]) + 3)
        else:
            bar += i[0] + '   '
            num += ' ' * len(bar)
            den += ' ' * len(bar)

    num = num[:-3]
    bar = bar[:-3]
    den = den[:-3]

    print(
        f'Transformación al plano z con T = {T}seg: (https://lpsa.swarthmore.edu/LaplaceZTable/LaplaceZFuncTable.html)')
    print(f'\n{num}\n{bar}\n{den}\n')

    print('Factorizando:')
    tf_z = control.sample_system(tf, T)
    print(factorize(tf_z))
    print(tf_z)

    return tf_z


def routh_hurwitz(gch: control.TransferFunction, K=False):
    if(not K):
        F = control.tf(gch.num[0][0], [1]) + control.tf(gch.den[0][0], [1])
        F = F.num[0][0].tolist()
        if((0 in F) or not (all([i > 0 for i in F]) or all([i < 0 for i in F]))):
            print('El sistema es inestable')
    else:
        k = symbols('k')
        F_0 = [i * k for i in control.tf(gch.num[0][0], [1]).num[0][0]]
        F_1 = control.tf(gch.den[0][0], [1]).num[0][0]
        if (len(F_0) > len(F_1)):
            F_1 = (len(F_0) - len(F_1)) * [0] + F_1
        elif (len(F_0) < len(F_1)):
            F_0 = (len(F_1) - len(F_0)) * [0] + F_0
        F = [i + j for i, j in zip(F_0, F_1)]

    table = np.zeros((len(F), (ceil(len(F) / 2))), dtype='object')
    F_up, F_down = F[::2], F[1::2]
    if(len(F) % 2 != 0):
        F_down += [0]

    table[0] = F_up
    table[1] = F_down
    ps = table[0:2]
    for line, i in zip(table[2:], range(len(table[2:]))):
        templist = ps[0].tolist()
        while(templist[-1] == 0):
            templist = templist[:-1]
        for j in range(len(templist) - 1):
            line[j] = simplify(
                (ps[1, 0] * ps[0, j + 1] - ps[0, 0] * ps[1, j + 1]) / ps[1, 0])
        ps = table[i + 1:i + 3]
    size = 0
    for line in table:
        for column in line:
            if(len(str(column)) > size):
                size = len(str(column))
    for line, i in zip(table, range(len(table))):
        print(f's^{str(len(table) - i - 1)} | ', end='')
        print('  '.join([str(j).center(size) for j in line]))
    if(not K):
        if(all([i > 0 for i in F]) or all([i < 0 for i in F])):
            return print('El sistema es estable')
    else:
        conditions = [reduce_rational_inequalities(
            [[i > 0]], k, relational=False) for i in table[:, 0] if isinstance(i, Expr)]
        interval = conditions[0]
        for i in range(len(conditions)):
            interval = interval.intersect(conditions[i])
        #interval = simplify_logic(interval, force=True)
        print(
            f'El rango de k para el cual el sistema es estable es: {interval.as_relational(k)}')


# Acá se arma la función de transferencia. Las constantes se pueden poner multiplicando por afuera
# Para crear una función nueva se usa control.tf(NUMERADOR, DENOMINADOR)
# Numerador y Denominador pueden ser listas de coeficientes como la primera parte del ejemplo, o se puede
# poner como np.poly(LISTA) para poner las raices en vez de los coeficientes
# Las funciones de transferencia pueden sumarse, multiplicarse, dividirse, etc.
# Ejemplo:
# G = 40 * 0.707 * control.tf([1, 16, 80], [1]) * control.tf(
#    np.poly([-2.78, -.01]), np.poly([-1, -10, -.025, 0, 0]))
# EJERCICIO 1 PARCIAL 21/5/21
#G = 24 * 0.025 * control.tf(np.poly([4, 4, -0.39]), np.poly([0, 0, -9]))
# C = 0.186 * control.tf(np.poly([-.299, -.03]), np.poly([0, -.00017]))

# EJERCICIO 2 PARCIAL 21/5/21
# G = 2.666 * control.tf(np.poly([3]),np.poly([-12, -9]))
# H = 132 * control.tf([1],np.poly([-3]))
# C = -2.95 * control.tf(np.poly([-1.02, -.1]), np.poly([0, -8.84, -.01]))

#G = control.tf([1], np.poly([0, -10]))
#G_z = s_to_z(G, 0.01)
#G_w = z_w_transform(G_z)
# stability_test(G_w)
#routh_hurwitz(G, True)
# stability_test(G)

# G = control.tf([30], np.poly([0, -2, -2.5]))
# H = control.tf([1, 2], [1, 5])

# T = G / (1 + G * H)
# print(factorize(T))

#G = 0.316227766 * control.tf([1/30, 1],[1/300,1])
#stability_test(G)

#G_z = s_to_z(G,0.01)

G = control.tf([1],[1,1,0])
G_z = s_to_z(G, 0.1)
G_w = z_to_w(G_z)
routh_hurwitz(G_w, True)