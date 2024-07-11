## Coronary heart disease prognosis and diagnosis using Machine Learning techniques: Feature Extraction, Supervised and Unsupervised Learning
### _By Catalina Mihai and Thomas Halkier Nikolajsen_
---
A retrospective sample of males in a heart-disease high-risk region
of the Western Cape, South Africa. There are roughly two controls per
case of CHD. Many of the CHD positive men have undergone blood
pressure reduction treatment and other programs to reduce their risk
factors after their CHD event. In some cases the measurements were
made after these treatments. These data are taken from a larger
dataset, described in  Rousseauw et al, 1983, South African Medical
Journal. 

sbp		systolic blood pressure
tobacco		cumulative tobacco (kg)
ldl		low densiity lipoprotein cholesterol
adiposity
famhist		family history of heart disease (Present, Absent)
typea		type-A behavior
obesity
alcohol		current alcohol consumption
age		age at onset
chd		response, coronary heart disease

To read into R:
read.table("http://www-stat.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data",
	sep=",",head=T,row.names=1)

Lastly, we have investigated patient readings _grouping_ and _anomaly detection_ using Unsupervised Learning methods of __density estimation__ and __clustering__, together with finding frequently-occurring disease-confident patterns from patients' data using __association mining__:

* **_[Project 3: Heart Disease - Unsupervised Machine Learning](https://github.com/seby-sbirna/DTU-Introduction-to-Machine-Learning-and-Data-Mining-Capstone-Project/tree/master/Project%203%20-%20UCL%20Heart%20Disease%20-%20Unsupervised%20Machine%20Learning)_**
---




Determining presence of any kind of diseases is a skill which has always been needed by society and, up until recently, could only be performed meticulously by doctors with extensive training and experience. 

**_Our problem of interest is to be able to take advantage of the high computational power available nowadays by using various Machine Learning techniques upon patients’ data, in order to detect accurately and rapidly whether such patients are suffering from diseases._**

For this project, we have decided to focus on detecting the presence of _coronary heart disease_ using a dataset provided by UCL and Kaggle. The dataset’s creators were Andras Janosi, M.D., William Steinbrunn, M.D., Matthias Pfisterer, M.D. and Robert Detrano, M.D. 

Firstly, we have analysed our dataset using various _**data visualization and feature extraction**_ methods, among which the most beneficial for our project was __PCA__: 

* **_[Project 1: Heart Disease - Data feature extraction and visualisations](https://github.com/seby-sbirna/DTU-Introduction-to-Machine-Learning-and-Data-Mining-Capstone-Project/tree/master/Project%201%20-%20UCL%20Heart%20Disease%20-%20Analysis%20and%20Visualization%20(PCA))_**

Afterwards, we have performed and evaluated the performance and characteristics of various types of Supervised Learning models upon the Heart Disease data, using __Neural Networks__, __Decision Trees__, __Logistic Regressions__ and baselines for model comparison:

* **_[Project 2: Heart Disease - Supervised Machine Learning](https://github.com/seby-sbirna/DTU-Introduction-to-Machine-Learning-and-Data-Mining-Capstone-Project/tree/master/Project%202%20-%20UCL%20Heart%20Disease%20-%20Supervised%20Machine%20Learning)_**


