# 🧠 Hybrid CNN-LSTM for Vietnamese News Classification

\<p align="center"\>
\<img src="[https://github.com/user-attachments/assets/238bc084-16cc-46bc-8fcd-8d74eda66b6a](https://github.com/user-attachments/assets/238bc084-16cc-46bc-8fcd-8d74eda66b6a)" width="700"/\>
\</p\>
\<p align="center"\>\<i\>Fig. Hybrid CNN-LSTM Architecture Pipeline\</i\>\</p\>

## 📌 Architecture Overview

This project implements a robust **Hybrid CNN-LSTM network** designed for text classification. By synergizing Convolutional Neural Networks (CNN) with Long Short-Term Memory (LSTM) networks, the model effectively captures both local semantic structures and long-term contextual dependencies within text sequences.

  * **CNN (Feature Extractor):** Acts as an n-gram detector to identify prominent local phrases and syntactic patterns.
  * **LSTM (Context Analyzer):** Processes the spatial features extracted by the CNN to understand the temporal sequence and global context of the document.

## ⚙️ Model Pipeline & Data Flow

1.  **Word Embedding:** Input text sequences (token indices) are mapped into dense vector representations using an Embedding layer.
2.  **Local Feature Extraction (1D Convolution):** The embedded vectors are permuted and passed through a `Conv1d` layer. This acts as a sliding window over the text, capturing critical local bi-grams or tri-grams regardless of their position in the sentence. A ReLU activation is applied to introduce non-linearity.
3.  **Temporal Dependency Modeling (LSTM):** The feature maps from the CNN are realigned `[batch_size, seq_length, cnn_out_channels]` and fed into the LSTM layer. The LSTM analyzes this sequence of high-level features to retain long-term memory of the document's overall narrative.
4.  **Global Max Pooling:** Instead of using the hidden state of the final time step, a 1D Global Max Pooling operation (`torch.max`) is applied across the sequence dimension. This extracts the most salient features from the entire document, aggressively reducing dimensionality while preserving the strongest signals.
5.  **Classification Head:** The pooled feature vector passes through a Dropout layer (for regularization and preventing overfitting) before entering the final Fully Connected (`Linear`) layer, which outputs the logits for the `num_classes` news categories.
