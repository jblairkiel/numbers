import decimal
from AutoEncoder import train_data
import plotly.express as px

def get_pi():
    """
    Compute Pi to the current precision.

    Examples
    --------
    >>> print(pi())
    3.141592653589793238462643383

    Notes
    -----
    Taken from https://docs.python.org/3/library/decimal.html#recipes
    """
    decimal.getcontext().prec += 2  # extra digits for intermediate steps
    three = decimal.Decimal(3)      # substitute "three=3.0" for regular floats
    lasts, t, s, n, na, d, da = 0, three, 3, 1, 0, 0, 24
    while s != lasts:
        lasts = s
        n, na = n + na, na + 8
        d, da = d + da, da + 32
        t = (t * n) / d
        s += t
    decimal.getcontext().prec -= 2
    return +s               # unary plus applies the new precision


def calculate_pi_digits(n):

    decimal.getcontext().prec = n
    pi = get_pi()

    pi_digits = {}
    pi_str = str(pi).replace('.', '')
    for i in range(1, n + 1):
        pi_digits[str(i)] = int(pi_str[i - 1])

    return pi_digits

if __name__ == "main":
    n_digits = 100000
    pi_digits = calculate_pi_digits(n_digits)
    #[print(pi_digits[a]) for a in pi_digits]
    #print(pi_digits)
    data_dict = [{ "key": a,  "value": pi_digits[a]} for a in pi_digits]
    data_list = [[ int(a),  pi_digits[a]] for a in pi_digits]
    #[print(a) for a in data_list]
    fig = px.histogram(data_dict, x='value')
    fig.write_html('out.html')

    print("Training Dataset")
    train_data(data_list)

