# Machine-Learning Emulation of Climatic Impact-Drivers A Cold-Spell Case Study (EAEE4000 final project)
# Machine-Learning Emulation of Climatic Impact-Drivers: Cold Spells

This repository contains code used in a final project studying whether machine-learning models can emulate **cold spells**, a key *Climatic Impact-Driver (CID)*, using output from the NASA GISS ModelE2.1 Earth system model.

The project compares two neural-network architectures:
- a **Convolutional Neural Network (CNN)** emphasizing spatial learning, and  
- a **Long Short-Term Memory (LSTM)** network emphasizing temporal learning.

The goal is to evaluate model performance and explore what these approaches reveal about the predictability of cold-spell intensity.

---

## Repository Structure

├── cnn_cid.ipynb # CNN-based cold-spell emulator
├── lstm_cid.ipynb # LSTM-based cold-spell emulator
├── utils.py # Shared utility functions (data processing, metrics, plotting)
