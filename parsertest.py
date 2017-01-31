from binderfinder.dataparser import Parser
import IPython as ip

p = Parser('./data/mock_csv2.csv')
ip.embed()
