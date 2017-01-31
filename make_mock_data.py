import random as rnd

with open('./data/mock_data.csv', 'w') as df:
    df.write('properties;2\n')
    df.write('parameters;2\n')
    for i, t in enumerate('ABCDEFGHIJ'):
        for j, s in enumerate('abcde'):
            df.write('{};{};{};{}\n'.format(t, s, i*25+rnd.random()*25, j*2.5+rnd.random()*2.5))

with open('./data/mock_data_rnd.csv', 'w') as df:
    df.write('properties;2\n')
    df.write('parameters;2\n')
    for i, t in enumerate('ABCDEFGHIJ'):
        for j, s in enumerate('abcde'):
            df.write('{};{};{};{}\n'.format(t, s, rnd.random()*25, rnd.random()*2.5))
