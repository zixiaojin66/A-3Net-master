# A3Net
A3Net is a deep learning model we proposed to predict the frequency of drug side effects.

# Requirements
* matplotlib==3.6.2
* networkx==2.8.8
* numpy==1.22.4
* pandas==1.5.1
* Pillow==9.4.0
* rdkit==2022.9.4
* rdkit-pypi==2022.9.4
* scikit_learn==1.2.1
* scipy==1.10.1
* torch==1.12.1+cu113
* torch-geometric==1.7.2
* torch-cluster==1.6.0
* torch-scatter==2.0.9
* torch-sparse==0.6.15
  
# Files:
1.data

This folder contains original side effects and drugs data.

* **frequency_data.txt:**
  The standardised drug side effect frequency classes used in our study.


2. warm-scence_data
   
This folder contains side effects and drugs data in warm start scence.


3. cold-scence_data
   
This folder contains side effects and drugs data in cold start scence.

# Code 
Net.py: It defines the model used by the code.
vector.py: It defines a method to calculate the smiles of drugs as vertices and edges of a graph.
