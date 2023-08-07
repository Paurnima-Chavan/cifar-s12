# CIFAR10 classification using Custom Resnet trained on PyTorch Lightning framework
##Model
In the presented project, two key techniques were employed to enhance the training of a machine learning model: Image Augmentation and the OneCycle Learning Rate (LR) Scheduler.

Image Augmentation was skillfully employed to artificially expand the diversity of the training dataset. By applying a range of transformations such as flips, rotations, and changes in brightness, the model was exposed to a broader variety of training instances. This process aids in improving the model's ability to generalize and perform well on previously unseen data.

Furthermore, the OneCycle LR Scheduler was strategically implemented to dynamically adjust the learning rate during training. This technique involves gradually increasing the learning rate to a maximum value and then gradually decreasing it again. This approach facilitates faster convergence in the initial stages while preventing overfitting in later epochs. The combination of these two techniques resulted in a model that not only learned effectively from the training data but also demonstrated improved generalization and performance.

![image](https://github.com/Paurnima-Chavan/cifar-s12/assets/25608455/4a9c4cc1-fe82-4284-a0a1-c46084d88865)

## Logs and Graphs
By leveraging the capabilities of PyTorch Lightning, essential information about the training process, such as training and validation loss, accuracy, and other metrics, were efficiently captured and recorded.

Additionally, the framework facilitated the creation of insightful graphs that visually depicted the model's training journey. These graphs provided a clear visualization of how the model's performance evolved over epochs, enabling a comprehensive understanding of its convergence and effectiveness.

![image](https://github.com/Paurnima-Chavan/cifar-s12/assets/25608455/d524f205-03c4-4730-956f-abad5943f510)

![image](https://github.com/Paurnima-Chavan/cifar-s12/assets/25608455/575a7740-f0f4-40ea-86cb-9406d835f8af)

## Misclassified Images
Through the examination of misclassified images, the project not only illuminated areas for potential enhancement but also contributed to a more comprehensive evaluation of the model's strengths and weaknesses. This approach is pivotal for iterative model refinement and achieving higher levels of accuracy and generalization.

![image](https://github.com/Paurnima-Chavan/cifar-s12/assets/25608455/1d41dfac-5961-4975-891c-dfbbc140e304)



