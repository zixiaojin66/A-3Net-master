# A3Net
A3Net is a deep learning model we proposed to predict the frequency of drug side effects.

# Requirements
* networkx==2.8.8
* numpy==1.22.4
* pandas==1.5.1
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
Net.py: It defines the model.

vector.py: It defines a method to calculate the smiles of drugs as vertices and edges of a graph.

utils.py: It defines some other functions.

warm-scence.py: Warm start scence of 750 drugs and 994 side effects.

cold-scence.py: Cold start scence of 750 drugs and 994 side effects.

# Run
```bash
python warm-scence.py --tenfold --save_model --epoch 3000 --lr 0.0001
```
# Contact
If you have any questions or suggestions with the code, please let us know. Contact ZiXiao Jin at wqq123@cug.edu.cn.
