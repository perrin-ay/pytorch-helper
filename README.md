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

- configuration class implements all standard nn.Module class for FCNs, CNN, RNNs, AE, VAEs and transformers
- design your network using python dictionary with key representing the order of layers in forward pass
- ready to use pytorch custom classes like BidirectionextractHiddenfinal and hiddenBidirectional
- BidirectionextractHiddenfinal extracts and returns the final hidden state from LSTM and hiddenBidirectional processes this and return the concatanated forwards and backwards hidden state
- configureRNN and configureFCN take these dictionaries and converts to pytorch moduledicts/modulelist and registers all elements appropriately as pytorch modules
- configureNetwork creates a pytorch network with forward calling pytorch registered modules in increasing order of keys in dictionary rnnlayers and fcnlayers
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

```
rnnnet:  ({1: Embedding(20000, 500), 2: LSTM(50, 256, num_layers=2, batch_first=True, bidirectional=True), 3: BidirectionextractHiddenfinal(), 4: hiddenBidirectional()}, {'batch_size': 128, 'pretrained_embeddings': False})
fcnnet:  {5: Linear(in_features=512, out_features=6, bias=True)}

RNNpacked(
  (layers): ModuleDict(
    (1): Embedding(20000, 500)
    (2): LSTM(50, 256, num_layers=2, batch_first=True, bidirectional=True)
    (3): BidirectionextractHiddenfinal()
    (4): hiddenBidirectional()
    (5): Linear(in_features=512, out_features=6, bias=True)
  )
)

```

- Training api for multilabel classification ( multiclass where more than one label is true)

```
rnnnet.configureTraining(epochs = 10, lossfun = nn.BCEWithLogitsLoss(), optimizer='adam',lr=0.005,
                          weight_decay=0,momentum=0.9, prntsummary = False, gpu= True)
rnnnet.trainmultilabel(numclasses = 6) # number of classes for multilabel

Is GPU available ? :  True
#####Printing GPU Device Info###### 

ID of current CUDA device:  0
Name of current CUDA device is:  Tesla T4
Amount of GPU memory allocated:  101267968
Amount of GPU memory reserved:  9844031488
Processor device configured is:  cuda:0
Time ( in secs) elapsed since start of training :  0.00571751594543457
At Batchidx 0 in epoch 0:  batchacc is 0.000000 and loss is 0.688006 
At Batchidx 1 in epoch 0:  batchacc is 96.875000 and loss is 0.338370 
At Batchidx 2 in epoch 0:  batchacc is 90.625000 and loss is 0.159723 
At Batchidx 3 in epoch 0:  batchacc is 87.500000 and loss is 0.213743 
At Batchidx 4 in epoch 0:  batchacc is 90.625000 and loss is 0.107479 
At Batchidx 5 in epoch 0:  batchacc is 87.500000 and loss is 0.193964 
At Batchidx 6 in epoch 0:  batchacc is 90.625000 and loss is 0.104387 
At Batchidx 7 in epoch 0:  batchacc is 87.500000 and loss is 0.245084 
At Batchidx 8 in epoch 0:  batchacc is 62.500000 and loss is 0.529418 

```
