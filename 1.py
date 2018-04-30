import mnist
from PIL import Image
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
class mnistMLP:
    def __init__(self, batch_size=128, epochs=20 , validation_split=0.3
                 ,layer =[512,100,10], act_func=['relu','relu','softmax']
                 ,show_metrics = True , error_figs=True , weight_first_layer =True
                 ,dropout_flag = True , dropout_rate = 0.2, train_test_1stimage = False
                 ,train_test_fig_name ='1.png' , weight_first_layer_fig_name='2.png'):
        self.num_classes = 10
        self.input_shape = 784
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.figure_counter = 1
        self.tv_name = train_test_fig_name
        self.w_name = weight_first_layer_fig_name
        self.error_fig = error_figs

        self.loadMnistData()
        if train_test_1stimage:
            self.showImage(self.train_images[0],"test.png")
            self.showImage(self.test_images[0],"train.png")
        self.reshapeData()
        self.normilizeData()
        self.categorizeLabels()
        self.initializeModel()

        for i in range(len(layer)):
            self.addLayer(layer[i], activationFunction=act_func[i])
            if dropout_flag and i < len(layer)-1:
                self.addDropout(dropout_rate)

        self.compileModel()
        self.history = self.fitModel()
        self.calculatePrecisionRecallFscore(showFlag=show_metrics)
        if weight_first_layer:
            self.drawLayerWeightsGrayScale(layer_depth=1) #input layer
        if error_figs:
            self.drawTrainValidationLoss()

    def loadMnistData(self):
        self.train_images = mnist.train_images()
        self.train_labels = mnist.train_labels()
        self.test_images = mnist.test_images()
        self.test_labels = mnist.test_labels()

    def reshapeData(self):
        self.train_images = self.train_images.reshape(60000, self.input_shape).astype('float32')
        self.test_images = self.test_images.reshape(10000, self.input_shape).astype('float32')

    def normilizeData(self):
        self.train_images /= 255
        self.test_images /= 255

    def showImage(self,sample2show , name):
        img = Image.fromarray(sample2show)
        img.save(name)
        img.show()

    def categorizeLabels(self):
        self.train_labels = keras.utils.to_categorical(self.train_labels, self.num_classes)
        self.test_labels = keras.utils.to_categorical(self.test_labels, self.num_classes)

    def initializeModel(self):
        self.model = Sequential()

    def addLayer(self, layerSize , activationFunction):
        self.model.add(Dense(layerSize, activation=activationFunction, input_shape=(self.input_shape,)))

    def addDropout(self, rate):
        self.model.add(Dropout(rate=rate))

    def compileModel(self):
        self.model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

    def fitModel(self):
        if self.error_fig :
            history =self.model.fit(self.train_images, self.train_labels,
                      batch_size=self.batch_size,
                      epochs=self.epochs,
                      verbose=1,
                      validation_split=self.validation_split,
                      validation_data=(self.test_images,self.test_labels) )
        else :
            history = self.model.fit(self.train_images, self.train_labels,
                                     batch_size=self.batch_size,
                                     epochs=self.epochs,
                                     verbose=1,
                                     validation_split=self.validation_split)
        return history

    def calculatePrecisionRecallFscore(self, showFlag = False):
        predict_labels = list(self.model.predict_classes(self.test_images))
        true_labels = self.testLabels2List()
        precision, recall, fscore, support = precision_recall_fscore_support(true_labels , predict_labels)
        allprec, allrec, allf, allsupp = precision_recall_fscore_support(true_labels , predict_labels, average="micro")
        if showFlag :
            print("\n")
            for i in range(len(precision)):
                print (i," class ---> ","precision=",precision[i]," recall=",recall[i]," fscore=",fscore[i])
            print ("\naverage precision =",allprec,"\naverage recall =",allrec,"\naverage fscore=",allf)

    def testLabels2List(self):
        trueLabelsList =[]
        for i in range(len(self.test_labels)):
            counter = 0
            for j in self.test_labels[i]:
                if j == 1 :
                    trueLabelsList.append(counter)
                counter+=1
        return trueLabelsList

    def drawLayerWeightsGrayScale(self,layer_depth):
        layer = self.model.get_layer(index=layer_depth)

        output_weights = layer.get_weights()[1]
        input_weights = layer.get_weights()[0]

        x = input_weights[:, 40]
        x = x.reshape(28,28)
        plt.imshow(x,cmap='gray')
        plt.savefig(self.w_name)
        plt.close()

    def drawTrainValidationLoss(self):
        loss = self.history.history['loss']
        if self.validation_split or self.error_fig:
            val_loss = self.history.history['val_loss']
        plt.plot(loss)
        if self.validation_split or self.error_fig:
            plt.plot(val_loss)
        plt.legend(['train_loss', 'test_loss'])
        plt.savefig(self.tv_name)
        self.figure_counter+=1
        plt.close()


x=mnistMLP(batch_size=60000, epochs=50 , validation_split=0
                 ,layer =[128,10], act_func=['relu','softmax']
                 ,show_metrics = True , error_figs=True , weight_first_layer =True
                 ,dropout_flag = True , dropout_rate = 0.2, train_test_1stimage = False
                 ,train_test_fig_name='loss.png', weight_first_layer_fig_name='weight.png')
