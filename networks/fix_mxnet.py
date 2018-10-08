import numpy as np
import mxnet as mx

#Caricamento file sbagliati
gg = mx.model.load_checkpoint('netTest11',0)

#Caricamento via signoli file
sym = mx.symbol.load('netTest111-symbol.json')
params = mx.nd.load('netTest111-0000.params')

#Mostra i file caricati
sym.list_arguments()
print list(params)

#Inferenza usanto un workarond che include "Intput" dentro il dizionario
inputND = mx.nd.array(np.array([[1,2,3,4,5],[1,2,3,4,6]]));
params["throttle"] = inputND
e = sym.bind(mx.cpu(), params)
out = e.forward()

print out

#Export della rete creata
e.save_checkpoint('export',0)

#Creo il grafo
#invar = mx.symbol.Variable('data')
#net = mx.symbol.FullyConnected(data = inverr, name = 'fc1', num_hidden = 2)

#Creo un modulo per poter inizializzare i pesi
#mod = mx.mod.Module(net)

#Faccio una bind sul modulo per allocarer le giuste dimensioni
#mod.bind(data_shapes=[('data',(1,5))])

#Inizializzo i pesi del mio grafo FullyConnected
#mod.init_params()

#Creo l'input
#Batch = namedtuple('Batch', ['data'])
#invect = [mx.nd.ones((1,5))]

#Eseguo la forward
#mod.forward(Batch(invect))

#Export della rete creata
#mod.save_checkpoint('export',0)

#Ricarico la rete esportata
#sym = mx.symbol.load('export-symbol.json')
#params = mx.nd.load('export-0000.params')

#Mostra i file caricati
#sym.list_arguments()
#list(params)
