import re
find_shortest = lambda l: len(min(re.findall('[a-z]+',l), key = len, default=[]))