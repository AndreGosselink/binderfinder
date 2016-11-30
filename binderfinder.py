from binderfinder import Matrix

# help(Matrix)
m = Matrix('./data/mock_data_rnd.csv', reference=[100, 100], weights=[0.05, 1],
        annotate='none', stats=True, sort='none', legend='bg', ceil=True,
        normalize='total')

m.show_me_where_the_white_rabbit_goes()
m.save_last_run()
