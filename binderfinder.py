from binderfinder import Matrix

m = Matrix('./data/mock_data.csv', reference=[30.0, 30.0], annotate='data', stats=True, sort='row')
print m
m.show_me_where_the_white_rabbit_goes()
m.save_last_run()
