# Seq2Seq Machine Translation
## Overview

Seq2seq turns one sequence into another sequence. It does so by use of a recurrent neural network (RNN) or more often LSTM or GRU to avoid the problem of vanishing gradient. The context for each item is the output from the previous step. The primary components are one encoder and one decoder network. The encoder turns each item into a corresponding hidden vector containing the item and its context. The decoder reverses the process, turning the vector into an output item, using the previous output as the input context.

[LANGUAGE TRANSLATION WITH TORCHTEXT](https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html)

[Sequence to Sequence Learning with Neural Network / Kaggle](https://www.kaggle.com/columbine/seq2seq-pytorch)

![Logo](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSKoRK4deYAwNU_B_CKWG0FwGFsCzal27mzvo04s3jvSQk6-Soh&s)

![Logo](https://upload.wikimedia.org/wikipedia/commons/7/7c/Kaggle_logo.png)