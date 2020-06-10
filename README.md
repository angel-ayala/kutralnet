# KutralNet: A Portable Deep Learning Model for Fire Recognition

Most of the automatic fire alarm systems detect the fire presence through sensors like thermal, smoke, or flame.
One of the new approaches to the problem is the use of images to perform the detection.
The image approach is promising since it does not need specific sensors and can be easily embedded in different devices.
However, besides the high performance, the computational cost of the used deep learning methods is a challenge to their deployment in portable devices.
In this work, we propose a new deep learning architecture that requires fewer floating points operations (flops) for fire recognition.
Additionally, we propose the use of modern techniques such as inverted residual block, convolution like depth-wise, and octave, to reduce the model’s computational cost and build a portable approach for fire recognition.
The experiments show that our model keeps high accuracy while substantially reducing the number of parameters and flops.
One of our models presents 71% fewer parameters than FireNet, while still presenting competitive accuracy and AUROC performance.
The proposed methods are evaluated on FireNet and FiSmo datasets.
The obtained results are promising for the implementation of the model in a mobile device, considering the reduced number of flops and parameters acquired.

This study was financed in part by the Coordenação de Aperfeiçoamento de Pessoal de Nı́vel Superior - Brasil (CAPES) - Finance Code 001, Fundação de Amparo a Ciência e Tecnologia do Estado de Pernambuco (FACEPE), and Conselho Nacional de Desenvolvimento Cientı́fico e Tecnológico (CNPq) - Brazilian research agencies.

---
## KutralNet architectures

This work presents a lightweight baseline deep model which is compared with three previously used models for the fire recognition task with the following model-id:

* **firenet**: for the FireNet model proposed by Jadon et al. [1].
* **octfiresnet**: for the OctFiResNet model proposed by Ayala et al. [2].
* **resnet**: for the modified version of ResNet50 presented by Sharma et al. [3].
* **kutralnet**: for the proposed approach KutralNet.

The model's architecture of KutralNet is represented by the image below.
<p align="center">
  <img src="https://github.com/angel-ayala/kutralnet/blob/master/assets/KutralNet_model.png?raw=true" height=100% width=50%>
</p>

From this baseline, three portables deep model are presented to evaluate the performance of inverted residual block, depth-wise convolution and, octave convolution, with the following model-id:

* **kutralnetoct**: for the KutralNet Octave model.
* **kutralnet_mobile**: for the KutralNet Mobile model.
* **kutralnet_mobileoct**: for the KutralNet Mobile Octave model.

The combination of the three methods results in the convolutional block represented by the image below.
<p align="center">
  <img src="https://github.com/angel-ayala/kutralnet/blob/master/assets/MobileOctave_block.png?raw=true" height=100% width=80%>
</p>

## Experimental Setup

For this work, two datasets were used:

* **FireNet**: used in a previous work [link here](https://github.com/arpit-jadon/FireNet-LightWeight-Network-for-Fire-Detection), with a [Google Drive folder](https://drive.google.com/drive/folders/1HznoBFEd6yjaLFlSmkUGARwCUzzG4whq) public available [1].
* **FiSmo**: A Compilation of Datasets fromEmergency Situations for Fire and Smoke Analysis, with a public dataset published in their [github repo](https://github.com/mtcazzolato/dsw2017) [4].

Also, some variations from the dataset were used, for use with the source code, the next dataset-id must be used in order to work with a given dataset:

* **firenet**, **firenet_test**: are the training and test subset from FireNet, with 2425 and 871 images, respectively.
* **fismo**, **fismo_balanced**: are the FiSmo dataset variants the first one is unbalanced with 6063 images, and the second one is balanced with 1968 images.
* **fismo_black**, **fismo_balanced_black**: are the same previous subsets but augmented with black images in the no-fire label.


## Training and testing

The available code can be executed on Google Colab using the following examples for training the models.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1T6FuLalbQc6uRX37eIdgzjDFxLZ-_YFq)

For training the FireNet model, another Google Colab is implemented due to the model is developed in Keras.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DJ62C0Rdj6b1DBzltw2Ak6hpUqm-J8ft)

And another one, for testing the models and graphics the results.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mhF-xfBtvAfvIo7CfcS3jCl8m9OMIVoV)

For the execution of a determined model, and dataset must specify the model-id and dataset-id, respectively.


### References

* [1] A. Jadon, M. Omama, A. Varshney, M. S. Ansari, and R. Sharma, "Firenet: A specialized lightweight fire & smoke detection model for real-time iot applications," CoRR, vol. abs/1905.11922, 2019.
* [2] A. Ayala, E. Lima, B. Fernandes, B. Bezerra, and F. Cruz, "Lightweight and efficient octave convolutional neural network for fire recognition," in 2019 IEEE Latin American Conference on Computational Intelligence (LA-CCI) (inpress), 2019, pp. 87–92.
* [3] J. Sharma, O.-C. Granmo, M. Goodwin, and J. T. Fidje, "Deep convolutional neural networks for fire detection in images," in Engineering Applications of Neural Networks, 2017, pp. 183–193.
* [4] M. T. Cazzolato, L. P. Avalhais, D. Y. Chino, J. S. Ramos, J. A. de Souza, J. F. Rodrigues-Jr, and A. Traina, "Fismo: A compilation of datasets from emergency situations for fire and smoke analysis," in Brazilian Symposium on Databases - SBBD, 2017, pp. 213–223.
