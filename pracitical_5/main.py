def JaccardDistance(s1, s2):
    set1 = set(s1.split())
    set2 = set(s2.split())
    return round((len(set1 | set2) - len(set1 & set2)) / len(set1 | set2), 5)


def DiceCoefficient(s1, s2):
    set1 = set(s1.split())
    set2 = set(s2.split())
    return round(1-(2 * len(set1 & set2)) / (len(set1) + len(set2)), 5)


# def CosineSimilarity():

if __name__ == '__main__':
    # 1a
    s0 = 'today sleep bed school text analytics'
    s1 = 'today read book library history'
    s2 = 'today read book classroom mathematics'
    s3 = 'today study data mining library'
    s4 = 'today study data science school'
    s5 = 'today read comic book classroom'
    print(JaccardDistance(s0,s0))
    print('JaccardDistance of s0 and s1 is:{}'.format(JaccardDistance(s0, s1)))
    print('JaccardDistance of s0 and s2 is:{}'.format(JaccardDistance(s0, s2)))
    print('JaccardDistance of s0 and s3 is:{}'.format(JaccardDistance(s0, s3)))
    print('JaccardDistance of s0 and s4 is:{}'.format(JaccardDistance(s0, s4)))
    print('JaccardDistance of s0 and s5 is:{}'.format(JaccardDistance(s0, s5)))
    print('JaccardDistance of s1 and s2 is:{}'.format(JaccardDistance(s1, s2)))
    print('JaccardDistance of s1 and s3 is:{}'.format(JaccardDistance(s1, s3)))
    print('JaccardDistance of s1 and s4 is:{}'.format(JaccardDistance(s1, s4)))
    print('JaccardDistance of s1 and s5 is:{}'.format(JaccardDistance(s1, s5)))
    print('JaccardDistance of s2 and s3 is:{}'.format(JaccardDistance(s2, s3)))
    print('JaccardDistance of s2 and s4 is:{}'.format(JaccardDistance(s2, s4)))
    print('JaccardDistance of s2 and s5 is:{}'.format(JaccardDistance(s2, s5)))
    print('JaccardDistance of s3 and s4 is:{}'.format(JaccardDistance(s3, s4)))
    print('JaccardDistance of s3 and s5 is:{}'.format(JaccardDistance(s3, s5)))
    print('JaccardDistance of s4 and s5 is:{}'.format(JaccardDistance(s4, s5)))
    print('\n')

    # 1b
    print(DiceCoefficient(s0,s0))
    print('DiceCoefficient of s0 and s1 is:{}'.format(DiceCoefficient(s0, s1)))
    print('DiceCoefficient of s0 and s2 is:{}'.format(DiceCoefficient(s0, s2)))
    print('DiceCoefficient of s0 and s3 is:{}'.format(DiceCoefficient(s0, s3)))
    print('DiceCoefficient of s0 and s4 is:{}'.format(DiceCoefficient(s0, s4)))
    print('DiceCoefficient of s0 and s5 is:{}'.format(DiceCoefficient(s0, s5)))
    print('DiceCoefficient of s1 and s2 is:{}'.format(DiceCoefficient(s1, s2)))
    print('DiceCoefficient of s1 and s3 is:{}'.format(DiceCoefficient(s1, s3)))
    print('DiceCoefficient of s1 and s4 is:{}'.format(DiceCoefficient(s1, s4)))
    print('DiceCoefficient of s1 and s5 is:{}'.format(DiceCoefficient(s1, s5)))
    print('DiceCoefficient of s2 and s3 is:{}'.format(DiceCoefficient(s2, s3)))
    print('DiceCoefficient of s2 and s4 is:{}'.format(DiceCoefficient(s2, s4)))
    print('DiceCoefficient of s2 and s5 is:{}'.format(DiceCoefficient(s2, s5)))
    print('DiceCoefficient of s3 and s4 is:{}'.format(DiceCoefficient(s3, s4)))
    print('DiceCoefficient of s3 and s5 is:{}'.format(DiceCoefficient(s3, s5)))
    print('DiceCoefficient of s4 and s5 is:{}'.format(DiceCoefficient(s4, s5)))
    print(DiceCoefficient('a','a'))
