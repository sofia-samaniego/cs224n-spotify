# cs224n-spotify

Final project for CS224n Winter 2018 at Stanford University.


In this project we propose a model that generates the name for a playlist given information about its tracklist. We first use a standard sequence-to-sequence model with unidirectional single-layer LSTM cells for both the encoder and the decoder and GloVe embedding initialization. We then add drop-out for regularization and introduce an attention mechanism to establish alternative connections between rel- evant words in song titles and playlist names. Finally, we provide an interactive console that generates playlist name suggestions given an input tracklist, which could serve as a useful automated suggestion system for digital music services users.
