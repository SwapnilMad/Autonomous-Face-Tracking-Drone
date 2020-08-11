import math


def repeatedString(s, n):
    count = 0
    for c in s:
        if c == 'a':
            count = count + 1
    f = n / len(s)
    if f.is_integer():
        return int(count * f)
    else:
        c = count * math.floor(f)
        i = 0
        while i < n % len(s):
            if s[i] == 'a':
                c = c + 1
            i = i + 1
        return int(c)


s = 'kmretasscityylpdhuwjirnqimlkcgxubxmsxpypgzxtenweirknjtasxtvxemtwxuarabssvqdnktqadhyktagjxoanknhgilnm'
n = 736778906400
print(repeatedString(s, n))
