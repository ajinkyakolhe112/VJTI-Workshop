# Introduction to Deep Learning with Neural Networks

- Neural networks are error minimizing machines. 
  - It's basic building block is a neuron
    - They are made up of neurons
    - Neurons are connected to each other
    - Neurons are organized into layers
    - Layers are organized into networks / models
    - Models are organized into features
    - features are organized into products
  - Same network can work on any kind of problem or dataset
    - [List of Image Based problems](https://www.kaggle.com/competitions?prestigeFilter=money&searchQuery=image)
      - Farm monitoring via Drone
      - Scanning for tumors, in Brain MRI
      - Sports feed augmentation
      - Special Effects in movies
      - Video Games, Photos
      - Predicting Chemical Molecule structure
---

# Problem Complexity & Number of Pixels
- Problem complexity increases exponentially with number of pixels
- `TODO:` Same Network training on MNIST & CIFAR10. Both similar image sizes, but very different performance

---
- Problem complexity. More complex the problem, more intelligence required to solve it. 
- Increasing model or dataset size leads to performance improvements

---

- Use **Dataset's** actual values to train models to predict that same value
- Model training, we monitor
  - In training we learn to copy the answer.
  - (x_actual, y_actual) -> y_predicted = untrained( x_actual )
  - Accuracy of unseen Data to know how good model is at solving new problems. **Called generalization**

---
## Single Neuron
```python

```

---
## Single Layer
```python
```

---
## Simple Neural Network
```python

```

---

Rule: Simplest dataset to model to training pipeline with lowest possible accuracy. 
- Just to get a pipeline working
- Work on accuracy later

---
# Entire field of Deep Learning is made of "Model Architecture" trained on "Dataset"

Write simplest pipeline for all 4. Only then improve on Model Architecture
Four Standard Stages of Deep Learning for
1. Model Architecture - `get_model()`
2. Dataset - `get_data_loding_function()`
3. Model Training - `trainer.train(model, dataset)`
4. Monitoring Training performance - `graph of training & validation accuracy per epochs`

---
