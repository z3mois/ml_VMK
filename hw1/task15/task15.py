from re import I
from typing import List


def hello(name: str = None) -> str:
    if name == None or name == "":
        return "Hello!"
    else:
        return f"Hello, {name}!"


def int_to_roman(num: int) -> str:
    th = ["", "M", "MM", "MMM"]
    hd = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"]
    tn = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
    on = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
    return (th[num // 1000] + hd[num % 1000 // 100] 
            + tn[num % 100 // 10] + on[num % 10])


def longest_common_prefix(strs_input: List[str]) -> str:
    if strs_input == []:
        return ""
    result = ""
    strs_input = [s.strip() for s in strs_input]
    mincount = min([len(s) for s in strs_input])
    for i in range(mincount):
        ch = strs_input[0][i]
        for s in strs_input:
            if s[i] != ch:
                return result
        result += ch
    return result
def simple(a:int)->int:
    k = 0
    for i in range(2, a // 2+1):
        if (a % i == 0):
            k = k + 1
    if (k <= 0):
        return 1
    else:
        return 0
def primes() -> int:
    i = 2
    while True:
        if simple(i) == 1:
            yield i
        i += 1
        
    


class BankCard:
    def __init__(self, total_sum: int, balance_limit: int = None):
        self.total_sum = total_sum
        self.balance_limit = balance_limit
    def __call__(self, sum_spent) -> None:
        if sum_spent > self.total_sum:
            raise ValueError("Balance check limits exceeded.")
        else:
            self.total_sum -= sum_spent
        print(f"You spent {sum_spent} dollars")
    def put(self, sum_put) -> None:
        self.total_sum += sum_put
        print(f"You put {sum_put} dollars.")
    @property
    def balance(self):
        if self.balance_limit == 0:
            raise ValueError("Balance check limits exceeded.")
        if self.balance_limit is not None:
            self.balance_limit -= 1
        return self.total_sum

    def __str__(self):
        return f"To learn the balance call balance."
    def __add__(self, other):
        return  BankCard(self.total_sum+other.total_sum, max(self.balance_limit,other.balance_limit ) 
        if other.balance_limit is not None and self.balance_limit  is not None else None)