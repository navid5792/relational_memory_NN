# Investigating Relational Recurrent Neural Networks with Variable Length Memory Pointer

Memory based neural networks can remember information longer while modelling temporal data. To improve LSTM’s memory, we encode a novel Relational Memory Core (RMC) as the cell state inside an LSTM cell using the standard multi-head self attention mechanism with variable length memory pointer and call it LSTM_RMC. Two improvements are claimed: The area on which the RMC operates is expanded to create the new memory as more data is seen with each time step, and the expanded area is treated as a fixed size kernel with shared weights in the form of query, key, and value projection matrices. We design a novel sentence encoder using LSTM_RMC and test our hypotheses on four NLP tasks showing improvements over the standard LSTM and the Transformer encoder as well as state-of-the-art general sentence encoders.

https://www.springerprofessional.de/en/investigating-relational-recurrent-neural-networks-with-variable/17956224

Canadian AI 2020

### Presentation Slide

https://www.caiac.ca/sites/default/files/shared/canai-2020-presentations/slides-83.pdf 