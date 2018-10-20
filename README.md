An implementation of “Reading and Thinking: Re-read LSTM Unit for Textual Entailment Recognition” by Lei Sha, Baobao Chang, Zhifang Sui, Sujian Li.

The authors approach derive from the notion that encoding the premise sentence and the hypothesis sentence individually is missing useful information that one sentence can contribute to the other. i.e., without the impact between the two sentences, it is difficult for the encoder to extract sentence-relationship-specific features.
In order to implement the above notion, the authors made some changes to the LSTM unit and they called it re-read LSTM (rLSTM):
  1. The rLSTM unit has another input and output which represent the memory of the attention (m) over a vector P that represent the premise.
  2. The hidden state of the rLSTM is efected by the memory m. m is added to the hidden state calculation by applying tanh on the memory m, doing element wise multiplication between the result and the LSTM output gate, and adding it to the original hidden state.
  3. The attention of the next time step is calculated as function of the P the premise vector, the previous cell state (Ct-1) and the current memory(m). The attention calculation is done using the additional parameters collections: Wm, Wp, Wc and Walpha.
