with open('./data/mock_data.csv', 'w') as df:
    for i, t in enumerate('ABCDEFGHIJ'):
        for j, s in enumerate('abcde'):
            df.write('{};{};{};{}\n'.format(t, s, i*25, j*25))
