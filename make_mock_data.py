import random as rnd

with open('./data/mock_data.csv', 'w') as df:
    for i, t in enumerate('ABCDEFGHIJ'):
        for j, s in enumerate('abcde'):
            df.write('{};{};{};{}\n'.format(t, s, i*25+rnd.random()*25, j*2.5+rnd.random()*2.5))
