# pytorch-helper

---

- Less boilerplate. The goal with this library was to provide an easy and flexible approach to designing neural nets

- Quickly build custom and non standard networks 

- Built on top of pytorch and abstracts pytorch hooks to debug and monitor gradient flows, weights and activations

- Track and plot loss and accuracy curves with training api 

- Create model checkpoints and auto save model based on loss and accuracy thresholds

- Extensible and customizable - easily add your own module or debug and monitoring functions at forward or backward pass

- Fully fleshed out and ready to use modules to build and train FFNs, CNNs, RNNs, LSTMs, GRUs, AEs, VAEs and transformer networks

---

## Usage examples

**LSTM bidirectional network for multilabel classification**

```

batch_size =128
embed_dims =  50
hidden_dims = 256
vocab_size = 20000

conf  = configuration()
embed =  conf.embeddings(vocab_size, embed_dims)
lstm = conf.lstm(embed_dims,hidden_dims,num_layers=2,bidirectional=True)
linear = conf.linear(infeatures = hidden_dims*2,outfeatures = 6)

rnnlayers= {1:embed, 2:lstm, 3:BidirectionextractHiddenfinal(),
            4:hiddenBidirectional()}
fcnlayers = {5:linear}

conf.configureRNN(rnnlayers = rnnlayers, params = {"batch_size": batch_size, "pretrained_embeddings" : False})
conf.configureFCN(layers =fcnlayers )

print('rnnnet: ',conf.rnnnet)
print('fcnnet: ',conf.fcnnet)

rnnnet.configureNetwork(confobj = {'RNN':[conf],'FCN':[conf]}, networktest =False, RNN = True)
print (rnnnet.net)

```
