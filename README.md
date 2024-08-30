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

# Usage examples

### LSTM bidirectional network

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

**Training for multilabel classification ( multiclass where more than one label is true)**

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

**Training for binary classification**

```

rnnnet.configureTraining(epochs = 15, lossfun = nn.BCEWithLogitsLoss(), optimizer='adam',lr=0.005,
                          weight_decay=0,momentum=0.9, prntsummary = False, gpu= True)
rnnnet.trainbinaryclass()

Is GPU available ? :  True
#####Printing GPU Device Info###### 

ID of current CUDA device:  0
Name of current CUDA device is:  Tesla T4
Amount of GPU memory allocated:  966518272
Amount of GPU memory reserved:  4638900224
Processor device configured is:  cuda:0
Time ( in secs) elapsed since start of training :  0.004150867462158203
At Batchidx 0 in epoch 0:  batchacc is 50.781250 and loss is 0.693170 
At Batchidx 1 in epoch 0:  batchacc is 47.656250 and loss is 0.702270 
At Batchidx 2 in epoch 0:  batchacc is 46.875000 and loss is 0.721781 
At Batchidx 3 in epoch 0:  batchacc is 47.656250 and loss is 0.695286 
At Batchidx 4 in epoch 0:  batchacc is 51.562500 and loss is 0.694537 
At Batchidx 5 in epoch 0:  batchacc is 46.093750 and loss is 0.698308 
At Batchidx 6 in epoch 0:  batchacc is 48.437500 and loss is 0.692062 
At Batchidx 7 in epoch 0:  batchacc is 51.562500 and loss is 0.692961 
At Batchidx 8 in epoch 0:  batchacc is 47.656250 and loss is 0.693732 

```
---

## Implement seq2seq model from Ilya Sutskever paper https://arxiv.org/abs/1409.3215 for machine translation

![Hume image](/images/Hume.jfif){: style="width:200px; float:center;"}


**Encoder LSTM network which generates context vectors for German texts**

```
batch_size = 128
embed_dims =  256
hidden_dims = 512
num_layers =  2
dropout = 0.5

# Encoder net for german texts

confde  = configuration()
embed_de = confde.embeddings(len(vocabularyde),embed_dims )
lstm_de = confde.lstm(embed_dims,hidden_dims,num_layers=num_layers,dropout = dropout)
drop_de =  confde.dropout(dropout)
encoderlayers= {1:embed_de,2:drop_de, 3:lstm_de, 4:UnidirectionextractHiddenCell()}
confde.configureRNN(rnnlayers = encoderlayers, params ={"batch_size":batch_size,"pretrained_embeddings" : False})

encoderGerman = Net()
encoderGerman.configureNetwork(confobj = {'RNN':[confde]}, RNN = True, networktest = False)

```
**Decoder LSTM network that takes encoder hidden states and english token inputs**
```

confen  = configuration()
embed_en = confen.embeddings(len(vocabularyen),embed_dims )
lstm_en = LSTMhc(embed_dims,hidden_dims,num_layers=num_layers,dropout = dropout)
drop_en =  confen.dropout(dropout)
linear  = Linearhc(infeatures = hidden_dims,outfeatures = len(vocabularyen)) 
decoderlayers= {1:unsqueezeOnes(1), 2:embed_en,3:drop_en, 4:lstm_en,  5:linear}
confen.configureRNN(rnnlayers = decoderlayers, params ={"batch_size":batch_size,"pretrained_embeddings" : False} )

decoderEng = Net()
decoderEng.configureNetwork(confobj = {'RNN':[confen]}, RNN = True,rnnhc = True, networktest = False)

```
**Create seq2seq network for end to end forward pass and backprop**
```

seq2seq  = Net()
seq2seq.setupCuda()
params ={'batch_size':128, 'src_vocab_len':len(vocabularyde), 'trg_vocab_len':len(vocabularyen),'device':seq2seq.device}
seq2seq.net = Seq2Seq(encoderGerman.net, decoderEng.net, params = params)

```
**Training is autoregressive with next token prediction. Control teacher_forcing where 0 means fully autoregressive** 
```

seq2seq.configureTraining(epochs=1,lossfun=nn.CrossEntropyLoss(ignore_index = padding_value_en),optimizer='adam',lr=0.001, 
                          weight_decay=0,momentum=0.9, prntsummary = True, gpu = False)
seq2seq.trainseq2seq(teacher_forcing=0.5, clipping =1)

```

---

## Convolution Variational Auto encoder 

- Create two configuration objects , one for encoder with conv layers and second for decoder deconv side
- The encoder net consists of conv->batchnorm->leakyrelu->dropout layers
- The decoder net consists of convtranspose - >batchnorm - >leakyrelu -> dropout layers
- The configureCNN takes the encoder layers and input image size returns final output channels, final output size, and per layer chanels and size

```
confencode =  configuration()
confdecode =  configuration()

conv1 = confencode.conv(3, 32, 3, 2, 1)
batchnorm1 = confencode.batchnorm2d(32)
dropout1 = confencode.dropout2d(0.2)

conv2 = confencode.conv(32,64,3,2,1)
batchnorm2 = confencode.batchnorm2d(64)
dropout2 = confencode.dropout2d(0.2)

conv3 = confencode.conv(64,64,3,2,1)
batchnorm3 = confencode.batchnorm2d(64)
dropout3 = confencode.dropout2d(0.2)

conv4 = confencode.conv(64,64,3,2,1)
batchnorm4 = confencode.batchnorm2d(64)
dropout4 = confencode.dropout2d(0.2)

flat = confencode.flatten()
leakyreluconv = confencode.leaky_relu()

convlayer = {1:conv1,2:batchnorm1,3:leakyreluconv,4:dropout1,
             5:conv2,6:batchnorm2,7:leakyreluconv,8:dropout2,
            9:conv3,10:batchnorm3,11:leakyreluconv,12:dropout3,
            11:conv4,12:batchnorm4,13:leakyreluconv,14:dropout4,
            15:flat}

confencode.configureCNN(convlayers = convlayer, inputsize = [128,128])

fcin = confencode.fcin
dropout = confdecode.dropout2d(0.2)
leakyreludeconv = confdecode.leaky_relu()

linear1 = confdecode.linear(200,fcin)
reshape = confdecode.unflatten(1,(64,8,8))

deconv1 = confdecode.convtranspose(64,64,3,2,0)
batchnorm1 = confdecode.batchnorm2d(64)
deconv2 = confdecode.convtranspose(64,64,3,2,1)
batchnorm2 = confdecode.batchnorm2d(64)
deconv3 = confdecode.convtranspose(64,32,3,2,1)
batchnorm3 = confdecode.batchnorm2d(32)
deconv4 = confdecode.convtranspose(32,3,3,2,1)
trim = Trim(128,128) 
sigmoid = confdecode.sigmoid() 

deconvlayer = {16:linear1,17:reshape,
               18:deconv1, 19:batchnorm1,20:leakyreludeconv,21:dropout,
               22:deconv2,23:batchnorm2,24:leakyreludeconv,25:dropout,
               26:deconv3,27:batchnorm3,28:leakyreludeconv,29:dropout,
               30:deconv4, 31:trim,32:sigmoid}

confdecode.configureCNN(deconvlayers = deconvlayer, deconv_inputsize = confencode.convoutimgsize)

```

**Create encoder-decoder VAE network**

```

vae = {'encoder':convlayer,'decoder':deconvlayer,'fcin':fcin,'zdims':200 } 
celebnet =  Net(print_console =True,logtodisk=False, savestart = False)
celebnet.configureNetwork(vae=vae , networktest =False)
print (celebnet.net)

VAE(
  (encoder): ModuleDict(
    (1): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): LeakyReLU(negative_slope=0.01)
    (4): Dropout2d(p=0.2, inplace=False)
    (5): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (6): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): LeakyReLU(negative_slope=0.01)
    (8): Dropout2d(p=0.2, inplace=False)
    (9): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (10): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (12): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (13): LeakyReLU(negative_slope=0.01)
    (14): Dropout2d(p=0.2, inplace=False)
    (15): Flatten(start_dim=1, end_dim=-1)
  )
  (decoder): ModuleDict(
    (16): Linear(in_features=200, out_features=4096, bias=True)
    (17): Unflatten(dim=1, unflattened_size=(64, 8, 8))
    (18): ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(2, 2))
    (19): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (20): LeakyReLU(negative_slope=0.01)
    (21): Dropout2d(p=0.2, inplace=False)
    (22): ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (23): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (24): LeakyReLU(negative_slope=0.01)
    (25): Dropout2d(p=0.2, inplace=False)
    (26): ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (27): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (28): LeakyReLU(negative_slope=0.01)
    (29): Dropout2d(p=0.2, inplace=False)
    (30): ConvTranspose2d(32, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (31): Trim()
    (32): Sigmoid()
  )
  (z_mean): Linear(in_features=4096, out_features=200, bias=True)
  (z_log_var): Linear(in_features=4096, out_features=200, bias=True)
)

```

**Train VAE network**

```

celebnet.configureTraining(epochs=40,lossfun=nn.MSELoss(reduction='none'),optimizer='adam',lr=0.0005,
                          weight_decay=0,momentum=0.9, prntsummary = False, gpu = True)
results = celebnet.trainVAE(lossreduction = False)

Time ( in secs) elapsed since start of training :  5.698938608169556
At Batchidx 0 in epoch 0:  batchacc abs error (mean if batches) is 0.276839 and loss is 5463.798340  

At Batchidx 1 in epoch 0:  batchacc abs error (mean if batches) is 0.274612 and loss is 5355.296875  

At Batchidx 2 in epoch 0:  batchacc abs error (mean if batches) is 0.266851 and loss is 5101.672852  

At Batchidx 3 in epoch 0:  batchacc abs error (mean if batches) is 0.265190 and loss is 5020.875977  

At Batchidx 4 in epoch 0:  batchacc abs error (mean if batches) is 0.256942 and loss is 4763.616211  

At Batchidx 5 in epoch 0:  batchacc abs error (mean if batches) is 0.248016 and loss is 4741.430664  

At Batchidx 6 in epoch 0:  batchacc abs error (mean if batches) is 0.244713 and loss is 4343.865723  

At Batchidx 7 in epoch 0:  batchacc abs error (mean if batches) is 0.245002 and loss is 4297.454590  

At Batchidx 8 in epoch 0:  batchacc abs error (mean if batches) is 0.242950 and loss is 4198.756836  

At Batchidx 9 in epoch 0:  batchacc abs error (mean if batches) is 0.242138 and loss is 4154.397949

```

---
