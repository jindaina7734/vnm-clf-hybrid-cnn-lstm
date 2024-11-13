# Hybrid CNN-LSTM

This model combines two main components: **CNN (Convolutional Neural Network)** and **LSTM (Long Short-Term Memory)**, with each part performing different roles in processing sequence data:

- CNN: Handles local features in the input sequence.
- LSTM: Manages long-term dependencies and context within the data sequence.

![image](https://github.com/user-attachments/assets/238bc084-16cc-46bc-8fcd-8d74eda66b6a)


<p align="center">
  <img src="https://github.com/user-attachments/assets/238bc084-16cc-46bc-8fcd-8d74eda66b6a" width="200"/>
</p>
<p align="center">Fig. Hybrid CNN-LSTM structure</p>

Main components in model initialization:

- **Embedding Layer (nn.Embedding)**: Converts words in the text sequence into embedding vectors as a preprocessing step before feeding data into CNN.
- **Convolution Layer (nn.Conv1d)**: Detects local patterns in the input sequence, similar to the CNN part. This layer extracts local features from the embedding vectors.
- **ReLU Activation Function**: Adds non-linearity to the model after convolution, allowing it to learn more complex features.
- **LSTM Layer (nn.LSTM)**: Processes long-term dependencies and context in the sequence data after CNN feature extraction.
- **Pooling (torch.max)**: Selects the most important values, reducing input dimensionality for the next layer.
- **Dropout Layer (nn.Dropout)**: Minimizes overfitting by randomly dropping neurons during training.
- **Final Layer (nn.Linear)**: Fully connects neurons to provide the model's final prediction.

Forward Function: From input x (a sequence of word indices), it is converted into embedding vectors of a specified length. Each word is represented as a vector to feed into the CNN. The order of tensor dimensions is adjusted to fit Conv1d’s input requirements using .permute(). Next, the convolution extracts local features, followed by the ReLU activation for non-linearity, aiding in identifying complex patterns.

After the convolutional layer, data is transformed back to [batch_size, seq_length, cnn_out_channels] and fed into the LSTM layer. In the LSTM layer, the output from CNN captures long-term dependencies and context in the sequence. Max pooling selects the highest value for each sequence, reducing data size. Finally, dropout is applied to avoid overfitting, and the data is passed through the fully connected layer for final prediction with num_classes outputs.
