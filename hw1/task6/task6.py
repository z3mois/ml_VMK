def check(x: str, file: str):
    list1 = [s.lower() for s in x.split()]
    dictinary = {}
    for s in list1:
        if not s in dictinary:
            dictinary[s] = 0
        dictinary[s] += 1
    answer  = dict(sorted(dictinary.items()))
    answerFile = open(file, "w")
    for key in answer:
        print(key, answer[key], file = answerFile )
    answerFile.close()

