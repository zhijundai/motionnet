# **Analysis of Seismic Recording Scaling Method Based on Deep Learning**

## Project description
> Ground motion input is the foundation of structure anti-seismic analysis, but it is impossible to have large earthquake recordings for every place where structure anti-seismic analysis is required. Therefore, this article compares the widely used seismic recording scaling method that directly adjusts the recording peak acceleration to the required value with the frequency domain scaling method, and builds a model based on deep learning to distinguish whether the acceleration recording is small earthquake or large earthquake after training. Model as a criterion shows that most of the small earthquakes adjusted by the seismic recording scaling method in the frequency domain can be identified as large earthquakes by the model. This result shows that, compared with the method of directly amplifying the peak acceleration, the acceleration recordings obtained by adjusting the frequency domain seismic recording scaling method may be closer to real large earthquake recordings, and it may be better as ground motion input. And it proves again, the main difference between large earthquakes and small earthquakes is that the frequency of large earthquakes is lower than that of small earthquakes.

## Configuration instructions  
>|Name|Version|
|:---:|:---:|
|python|3.7.3|
|numpy|1.16.4|
|pandas|0.25.1|
|tensorflow|1.13.1|
|scipy|1.2.1|  

## Usage
> There are four python files in this project. preprocessing.py is the data preprocessing part and provides input for model training. training.py is the model training part. generation.py is the input adjustment part. the small earthquake recordings after adjusting input the model to see if it is judged as a large earthquake. test.py is the test part and integrates the functions of generation.py.
> Since the model has been trained and placed in the mdoel folder, you only need to run test.py. After running, it will output three sets of data, each with two rows. The first line of each group is a list, and the two numbers in the list respectively represent the results obtained after model judgment: how many large earthquakes and small earthquakes are. The second line indicates how many large earthquakes and small earthquakes are in the input data, that is the correct answer, which can be compared with the first line.The three sets of data input are small earthquake recordings, large earthquake recordings, and small earthquake recordings after adjustment.

## Maintainers
> @ zhijundai  
> @ LiuT-92
