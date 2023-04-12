import numpy as np
import scipy.signal
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import sklearn.metrics
import glob
import os
import sys
import cv2
from numpy.lib.stride_tricks import as_strided
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
RESIZE_IMAGE=28 # The images will be resized to 28x28 pixels
# from OFFV3 import *





# RESIZE_DIM=28 # The images will be resized to 28x28 pixels


data_dir=os.path.join('') # path to the data folder

def load_model(model_file):
    with open(model_file, "rb") as f:
        weights_biases = pickle.load(f)
        print("weights_biases: ", weights_biases)
    return weights_biases
def get_key(path):
    # seperates the key of an image from the filepath
    key=path.split(sep=os.sep)[-1]
    return key
def get_data(paths_img,path_label=None,resize_dim=None):
    '''reads images from the filepaths, resizes them (if given), and returns them in a numpy array
    Args:
        paths_img: image filepaths
        path_label: pass image label filepaths while processing training data, defaults to None while processing testing data
        resize_dim: if given, the image is resized to resize_dim x resize_dim (optional)
    Returns:
        X: group of images
        y: categorical true labels
    '''
    X=[] # initialize empty list for resized images
    for i,path in enumerate(paths_img):
        img=cv2.imread(path,cv2.IMREAD_COLOR) # images loaded in color (BGR)
        #img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # cnahging colorspace to GRAY
        if resize_dim is not None:
            img=cv2.resize(img,(resize_dim,resize_dim),interpolation=cv2.INTER_AREA) # resize image to 28x28
        #X.append(np.expand_dims(img,axis=2)) # expand image to 28x28x1 and append to the list.
        gaussian_3 = cv2.GaussianBlur(img, (9,9), 10.0) #unblur
        img = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0, img)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) #filter
        img = cv2.filter2D(img, -1, kernel)
        thresh = 200
        maxValue = 255
        #th, img = cv2.threshold(img, thresh, maxValue, cv2.THRESH_BINARY);
        ret,img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # print("img.shape: ", img.shape)
        X.append(img) # expand image to 28x28x1 and append to the list
        # display progress
        if i==len(paths_img)-1:
            end='\n'
        else: end='\r'
        print('processed {}/{}'.format(i+1,len(paths_img)),end=end)
        
    X=np.array(X) # tranform list to numpy array
    if  path_label is None:
        return X
    else:
        df = pd.read_csv(path_label) # read labels
        df=df.set_index('filename') 
        y_label=[df.loc[get_key(path)]['digit'] for path in  paths_img] # get the labels corresponding to the images
        # y = np.eye(10)[y_label.reshape(-1)] # transfrom integer value to categorical variable
        y = np.eye(10)[np.array(y_label).reshape(-1)]

        # y=to_categorical(y_label,10) # transfrom integer value to categorical variable
        return X, y


def get_expanded_dim(input, output_size, kernel_size, padding=0, stride=1, dilate=0):
    working_input = input
    working_pad = padding
    # dilate the input if necessary
    if dilate != 0:
        working_input = np.insert(working_input, range(1, input.shape[1]), 0, axis=1)
        working_input = np.insert(working_input, range(1, input.shape[2]), 0, axis=2)

    # pad the input if necessary
    if working_pad != 0:
        working_input = np.pad(working_input, pad_width=((0,), (0,), (working_pad,), (working_pad,)), mode='constant', constant_values=(0.,))

    in_b, out_h, out_w,in_c = output_size
    out_b, _, _,out_c = input.shape
    batch_str, kern_h_str, kern_w_str, channel_str = working_input.strides

    return np.lib.stride_tricks.as_strided(
        working_input,
        (out_b, out_h, out_w, out_c, kernel_size, kernel_size),
        (batch_str, stride * kern_h_str, stride * kern_w_str,channel_str, kern_h_str, kern_w_str)
    )
class Convolution_Layer:
    def __init__(self,number_of_filters,filter_dimension,stride,padding,number_of_channels,input_dim):
        self.number_of_filters=number_of_filters
        self.filter_dimension=filter_dimension
        self.padding=padding
        self.stride=stride
        self.number_of_channels=number_of_channels
        self.input_dim=input_dim
        self.save_windows = None
        
        self.weight=np.random.randn(self.filter_dimension,self.filter_dimension,self.number_of_channels,self.number_of_filters)*(np.sqrt(2/self.input_dim))
        self.previous_output_image=None
        self.bias=np.zeros((1,1,1,self.number_of_filters))
    def forward_propagation(self, input_image):
        self.previous_output_image = input_image
        number_of_samples, input_height, input_width, input_channels = input_image.shape
        output_height = int((input_height - self.filter_dimension + 2 * self.padding) / self.stride) + 1
        output_width = int((input_width - self.filter_dimension + 2 * self.padding) / self.stride) + 1
        input_image = np.pad(input_image, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant')
        
        # --------------------------------- vectorized implementation ---------------------------------
        patch_images = np.zeros((number_of_samples, self.filter_dimension, self.filter_dimension, input_channels, output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                t1=i*self.stride
                t2=j*self.stride
                patch_images[:, :, :, :, i, j] = input_image[:, t1:t1+self.filter_dimension, t2:t2+self.filter_dimension, :]

        out = np.tensordot(patch_images, self.weight, ((1, 2, 3), (0, 1, 2))) + self.bias[None, None, None, :]
        out = np.rollaxis(out, 3, 1)
        out = np.reshape(out, (number_of_samples, output_height, output_width, self.number_of_filters))
        print("out shape as this working - ",out.shape)
        # --------------------------------- vectorized implementation ---------------------------------
        #print(out.shape)

        # -----------------------vectorized implementation of convolution2 -----------------------
        # windows = get_expanded_dim(input_image, (number_of_samples, output_height, output_width, input_channels), self.filter_dimension, self.padding, self.stride)
        # # print(windows.shape)
        # out = np.einsum('bhwikl,klio->bhwo', windows, self.filter)
        # out += self.bias
        # # print("out shape as this working - ",out.shape)
        # self.save_windows = windows

        # ---------------------------------non vectorized implementation ---------------------------------
        # out=np.zeros((number_of_samples,output_height,output_width,self.number_of_filters))
       
        # for i in range(output_height):
        #     for j in range(output_width):
        #         t1=i*self.stride
        #         t2=j*self.stride
        #         patch_image=input_image[:,t1:t1+self.filter_dimension,t2:t2+self.filter_dimension,:]
        #         #print(patch_image.shape,self.filter.shape)
                
        #         for f in range(self.number_of_filters):        
        #             temp=np.multiply(patch_image,self.filter[:,:,:,f])
        #             #print(patch_image,self.filter[:,:,0,f])
        #             out[:,i,j,f]=np.sum(temp,axis=(1,2,3))+self.bias[:,:,:,f]
        #--------------------------non vectorized implementation--------------------------        
        #print(out.shape)
        return out
        # return out


    
    
    def backward_propagation(self, gradient, learning_rate):

        # --------------------------------- vectorized implementation ---------------------------------
        # input = self.previous_output_image
        # windows = self.save_windows
        # padding = self.kernel_size - 1 if self.padding == 0 else self.padding
        # d_windows = get_expanded_dim(error_image, input.shape,self.filter_dimension, padding, 1,dilate=self.stride -1)
        
        # filter_rotate = np.rot90(self.filter, 2, (0, 1))
        # dbias = np.sum(error_image, axis=(0, 1, 2))
        # dfilter = np.einsum('bhwikl,bhwo->klio', windows, error_image)
        # dx = np.einsum('bhwokl,klio->bhwi', d_windows, filter_rotate)

        # # update weights and bias
        # self.filter -= learning_rate * dfilter
        # self.bias -= learning_rate * dbias
        # # print("backprop in conv done")
        # return dx
        # --------------------------------- vectorized implementation ---------------------------------
        number_of_samples=self.previous_output_image.shape[0]
        
        output_height=gradient.shape[1]
        output_width=gradient.shape[2]
        
        
        number_of_channels=self.previous_output_image.shape[3] #if not batch
        
        dX=np.zeros(self.previous_output_image.shape)
        dW=np.zeros(self.weight.shape)
        dB=np.zeros(self.bias.shape)
        #A_prev_pad = zero_pad(A_prev, pad) #previous_output_image
        self.previous_output_image=np.pad(self.previous_output_image,((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        #dA_prev_pad = zero_pad(dA_prev, pad) #dX
        dX2=np.pad(dX,((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        
        
        for i in range(output_height):
            for j in range(output_width):
                t1=i*self.stride
                t2=j*self.stride
                for f in range(self.number_of_filters):
                        
                    patch_image=self.previous_output_image[:,t1:t1+self.filter_dimension,t2:t2+self.filter_dimension,:]
                    temp_filter=self.weight[:,:,:,f].reshape(1,self.filter_dimension*self.filter_dimension*self.number_of_channels)
                    

                    temp_gradient= gradient[:,i,j,f].reshape(-1,1)
                    #print(temp_gradient.shape,temp_filter.shape)
                    
                    dX2[:,t1:t1+self.filter_dimension,t2:t2+self.filter_dimension,:]+= np.dot(temp_gradient,temp_filter).reshape(-1,self.filter_dimension,self.filter_dimension,self.number_of_channels)
                    temp_gradient2=temp_gradient.T
                    
                    temp_image=patch_image.reshape(-1,self.filter_dimension*self.filter_dimension*self.number_of_channels)
                    #print(temp_gradient2.shape,temp_image.shape)
                    dW[:,:,:,f]+=np.dot(temp_gradient2,temp_image)[0].reshape(self.filter_dimension,self.filter_dimension,self.number_of_channels)
                    dB[:,:,:,f]+=np.sum(gradient[:,i,j,f])
        
            #unpadded
        if self.padding != 0:
            dX[:, :, :, :] = dX2[:,self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            dX[:,:,:,:]=dX2
        self.weight-=(dW*learning_rate)
        self.bias-=(dB*learning_rate)
        
        return dX
        
class Relu_Layer:
    def __init__(self):
        self.previous_output_image=None
        
        self.weight=None
        self.bias=None
        
    def forward_propagation(self,input_image):
        self.previous_output_image=input_image
        Relu_output=np.where(input_image <= 0,0,input_image)
        return Relu_output
    def backward_propagation(self,gradient,learning_rate):
        # print("gradient shape in relu",gradient.shape)
        # print("previous output image shape in relu",self.previous_output_image.shape)
        Relu_derivative=np.where(self.previous_output_image <= 0,0,1)
        dX=np.zeros(self.previous_output_image.shape)
        dX=gradient * Relu_derivative
        return dX
class MaxPool:
    def __init__(self,filter_dimension,stride):
        self.filter_dimension=filter_dimension
        self.stride=stride
        self.previous_output_image=None
        
        self.weight=None
        self.bias=None

    def forward_propagation(self, input_image):
        self.previous_output_image=input_image
        number_of_samples, input_height, input_width, input_channels = input_image.shape
        output_height = int((input_height - self.filter_dimension) / self.stride) + 1
        output_width = int((input_width - self.filter_dimension) / self.stride) + 1

        out = np.zeros((number_of_samples, output_height, output_width, input_channels))
        batch_stride,height_stride,width_stride,channel_stride=input_image.strides
        out = as_strided(input_image, shape=(number_of_samples, output_height, output_width, input_channels, self.filter_dimension, self.filter_dimension), strides=(batch_stride, height_stride * self.stride, width_stride * self.stride, channel_stride, height_stride, width_stride))

        out = np.max(out, axis=(4, 5))
        

        
        return out
        
    
    def backward_propagation(self, gradient, learning_rate):

        # -------------------------vectorized version-------------------------
        # number_of_samples = self.previous_output_image.shape[0]
        # output_channels = self.previous_output_image.shape[3]
        # output_height, output_width = gradient.shape[1], gradient.shape[2]
        # input_height = (output_height - 1) * self.stride + self.filter_dimension
        # input_width = (output_width - 1) * self.stride + self.filter_dimension

        # dLdInput = np.zeros_like(self.previous_output_image)
        # batch_stride, height_stride, width_stride, channel_stride = self.previous_output_image.strides
        

        # out = as_strided(self.previous_output_image, shape=(number_of_samples, output_height, output_width, output_channels, self.filter_dimension, self.filter_dimension), strides=(batch_stride, height_stride * self.stride, width_stride * self.stride, channel_stride, height_stride, width_stride))
        # out = np.max(out, axis=(4, 5))

        # filter_mask = (self.previous_output_image == out[:, :, :, :, np.newaxis, np.newaxis])
        # dLdInput[filter_mask] = gradient[:, :, :, :, np.newaxis, np.newaxis][filter_mask]
        # dLdInput /= np.sum(filter_mask, axis=(4, 5), keepdims=True)

        # return dLdInput
        # -------------------------vectorized version-------------------------
        number_of_samples = self.previous_output_image.shape[0]
        output_channels = self.previous_output_image.shape[3]
        output_height, output_width = gradient.shape[1], gradient.shape[2]
        
        # Initialize the gradient with respect to input image
        gradient_wrt_input = np.zeros(self.previous_output_image.shape)

        
        
        for i in range(output_height):
            for j in range(output_width):
                t1 = i * self.stride
                t2 = j * self.stride
                
                for f in range(output_channels):
                    patch_image=self.previous_output_image[:,t1:t1+self.filter_dimension,t2:t2+self.filter_dimension,f]
                    
                    sample,row,col=patch_image.shape
                    
                    input2=patch_image.reshape(sample,row*col)
                    max_index=np.argmax(input2,axis=1)+[j*row*col for j in range(sample)]
                    filter_mask=np.zeros(patch_image.shape)
                    filter_mask[np.unravel_index(max_index,patch_image.shape)]=1
                    gradient_wrt_input[:,t1:t1+self.filter_dimension,t2:t2+self.filter_dimension,f]+=filter_mask*gradient[:,i,j,f,np.newaxis,np.newaxis]
        
        return gradient_wrt_input


class Softmax_Layer:
    def __init__(self):
        
        self.weight=None
        self.bias=None
    def forward_propagation(self,previous_image):
        # print("shape in softmax",previous_image.shape)
        return np.exp(previous_image.T)/np.sum(np.exp(previous_image.T),axis=0)
    def backward_propagation(self,gradient,learning_rate):
        return gradient
    
class Flattening_layer:
    def __init__(self):
        self.weight=None
        self.bias=None
    def forward_propagation(input_image):
        number_of_samples=input_image.shape[0]
        output_image = input_image.flatten('C').reshape(number_of_samples,-1)
        return output_image


class Fully_Connected_Layer:
    def __init__(self,output_dimension,input_dimension):
        self.input_dimension=input_dimension
        self.output_dimension=output_dimension
        
        
        self.bias=np.zeros((self.output_dimension,1))
        self.weight=np.random.randn(self.output_dimension,self.input_dimension)*(np.sqrt(2/self.input_dimension))
        self.previous_output_image=None 
        self.final_out_image=None
        self.previous_image_shape=None  

    def forward_propagation(self,previous_image):
        self.previous_image_shape=previous_image.shape
        previous_image=Flattening_layer.forward_propagation(previous_image)
        input_dimension=previous_image.shape[1]
        sample_count=previous_image.shape[0]
        self.previous_output_image=previous_image
        

        out_fully_connected=np.zeros((self.output_dimension,sample_count)) 
        out_fully_connected=np.dot(self.weight,previous_image.T) + self.bias
        out_fully_connected=out_fully_connected.T
        self.final_out_image=out_fully_connected
        return out_fully_connected
    def backward_propagation(self,gradient,learning_rate):
        # initialize the gradient with respect to weight and bias
        dW=np.zeros(self.weight.shape)
        dB=np.zeros(self.bias.shape)
        dX=np.zeros(self.previous_output_image.shape)
        
        # calculate the gradient with respect to weight and bias
        dW=np.dot(gradient.T,self.previous_output_image)
        dB=np.sum(gradient.T,axis=1,keepdims=True)
        dX=np.dot(gradient,self.weight)
        
        
        self.weight-=(dW*learning_rate)
        self.bias-=(dB*learning_rate)
        return dX.reshape(self.previous_image_shape)
    
def cross_entropy_loss(y_pred,y_true):
    # print(np.sum(-y_true*np.log(y_pred)))
    return np.sum(-y_true*np.log(y_pred))
# def confusion_matrix(y_true, y_pred):
#     # Convert the input arrays to NumPy arrays
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)

#     # Compute the True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)
#     TP = np.sum((y_true == 1) & (y_pred == 1))
#     FP = np.sum((y_true == 0) & (y_pred == 1))
#     TN = np.sum((y_true == 0) & (y_pred == 0))
#     FN = np.sum((y_true == 1) & (y_pred == 0))

#     # Return the computed confusion matrix as a NumPy array
#     return np.array([[TN, FP], [FN, TP]])

def gradient_clipping(gradients, max_norm):
    # Calculate the L2 norm of the gradients
    norm = np.linalg.norm(gradients)
    
    # If the norm is larger than the maximum norm, scale the gradients
    if norm > max_norm:
        gradients = gradients * max_norm / norm
        
    return gradients
def get_confusion_matrix(pred_labels,true_labels):

            # Calculate the confusion matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    # Create a copy of the confusion matrix
    conf_matrix_plot = conf_matrix.copy()

    # Add a border around the diagonal elements
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            if i == j:
                conf_matrix_plot[i, j] = conf_matrix[i, j] + 100
                
    plt.imshow(conf_matrix_plot, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.yticks(tick_marks, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, conf_matrix[i, j],
                    horizontalalignment="center")

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()




class CNN:
    def __init__(self,learning_rate=0.002):
        self.layers=[]
        self.learning_rate=learning_rate
    def add(self,layer):
        self.layers.append(layer)
    def forward_pass(self,input_image,weights,biases):
        
        count = 0

        for layer in self.layers:
            if isinstance(layer, Convolution_Layer) or isinstance(layer,Fully_Connected_Layer):

                layer.weight=weights[count]
                layer.bias=biases[count]
                count+=1
                
            # print("input shape",input_image.shape)
            input_image=layer.forward_propagation(input_image)
            print("in layer",layer)
            
            
        return input_image
    def backward_pass(self,gradient):
        for layer in reversed(self.layers):
            # print("gradient shape",gradient.shape)
            gradient=layer.backward_propagation(gradient,self.learning_rate)
        return gradient
    # def fit(self,X,y,x_val,y_val,epochs=1,batch_size=32):
    #     for epoch in range(epochs):
    #         total_loss=0
    #         total_accuracy=0
    #         print('Epoch:',epoch+1)
    #         for i in range(0,X.shape[0],batch_size):
    #             print("batch no ",i/batch_size,"/",X.shape[0]/batch_size,"")
    #             X_batch=X[i:i+batch_size]
    #             y_batch=y[i:i+batch_size]
    #             y_pred=self.forward_pass(X_batch).transpose()
    #             print('y_pred:',y_pred)
    #             label_pred=np.argmax(y_pred,axis=1) # return index of max value
    #             label_y = np.argmax(y_batch,axis=1) # return index of max value
    #             # print("label_pred: ",label_pred)

    #             convert_to_binary_labels = np.zeros((y_batch.shape[0],y_batch.shape[1]))
    #             # print("label_y.shape==y_batch.shape: ",label_y.shape==y_batch.shape)
    #             convert_to_binary_labels[np.arange(label_y.shape[0]),label_y] = 1
    #             # print("y_batch: ",y_batch)
    #             # print("convert_to_binary_labels: ",convert_to_binary_labels)
    #             # print("y_batch==convert_to_binary_labels: ",y_batch==convert_to_binary_labels)
    #             # print("y_pred: ",y_pred)

    #             loss=cross_entropy_loss(y_pred,convert_to_binary_labels)/X_batch.shape[0]
    #             print('Loss:',loss)
    #             total_loss+=loss

    #             # accuracy
    #             result_compare = np.equal(label_pred,label_y)
    #             accuracy = np.count_nonzero(result_compare)
    #             total_accuracy+=accuracy
    #             print('Accuracy:',accuracy)

    #             # f1_score

    #             gradient=(y_pred-y_batch) / X_batch.shape[0]
    #             self.backward_pass(gradient)
    #         # print('Total Loss:',total_loss)
    #         # print('Total Accuracy:',total_accuracy)

    #         total_loss/=((epoch+1)*(i+1))
    #         total_accuracy=(total_accuracy/X.shape[0])*100

    #         print("Training Set: \n","train loss: ",total_loss,"training accuracy: ",total_accuracy)

    #         # validation
    #         self.validate(x_val,y_val)
    #     # save model
    #     weights = []
    #     biases = []
    #     for layer in self.layers:
    #         if layer.__class__.__name__=='Convolutional_layer' or layer.__class__.__name__=='Fully_connected_layer':
    #             weights.append(layer.weight)
    #             biases.append(layer.bias)
    #     save_model(weights, biases, "model.pkl")

    def predict(self,X,weights,biases):
        return self.forward_pass(X,weights,biases).transpose()
    
    def evaluate(self,X,y):
        y_pred=self.predict(X)
        y_pred=np.argmax(y_pred,axis=1)
        y_true=np.argmax(y,axis=1)
        return np.sum(y_pred==y_true)/len(y_true)
    def validate(self,X,y,weights,biases):

        print("Validation Set: \n")
        y_pred=self.predict(X,weights,biases)
        y_pred_label=np.argmax(y_pred,axis=1)
        y_true=np.argmax(y,axis=1)
        f1_scr = f1_score(y_true,y_pred_label,average='macro')
        # print("f1_score: ",f1_score)

        convert_to_binary_labels = np.zeros((y.shape[0],y.shape[1]))
        # print("label_y.shape==y_batch.shape: ",label_y.shape==y_batch.shape)
        convert_to_binary_labels[np.arange(y_true.shape[0]),y_true] = 1
        
        # print("Y_PRED_SHAPE: ",y_pred.shape)
        # print("Y_TRUE_SHAPE: ",y_true.shape)
        loss = cross_entropy_loss(y_pred,convert_to_binary_labels)/X.shape[0]
        # print("loss: ",loss)

        # accuracy
        result_compare = np.equal(y_pred_label,y_true)
        accuracy = np.count_nonzero(result_compare)
        # print("accuracy: ",accuracy)
        accuracy=(accuracy/X.shape[0])*100
        get_confusion_matrix(y_pred_label,y_true)
        
        # print acc,loss,f1_score
        print("acc: ",accuracy,"loss: ",loss,"f1_score: ",f1_scr)
        return y_pred

    



    
# Create a list to store the file names and corresponding predicted labels

def main():
# Load the pickle file
    saved_model_file = 'model.pkl'
    weights_biases = load_model(saved_model_file)
    weights = weights_biases[0]
    biases = weights_biases[1]

    # Get the path to the folder containing the query images as a command line parameter
    folder_path = sys.argv[1]
    predictions = []

    # Loop through all the files in the folder

    # Load the image and convert it to a numpy array
    file_path = str(folder_path)+".csv"
    path_label_train_a=os.path.join(data_dir,file_path)
    paths_train_a=glob.glob(os.path.join(data_dir,folder_path,'*.png'))
    X_train_a,y_train_a=get_data(paths_train_a,path_label_train_a,resize_dim=RESIZE_IMAGE)


    indices=list(range(len(X_train_a)))
    ind=int(len(indices)*0.4)

    X_train_a=X_train_a[indices[:ind]] 
    y_train_a=y_train_a[indices[:ind]]


    X_train_all = X_train_a
    y_train_all = y_train_a

    X_train_all = X_train_all.reshape(X_train_all.shape[0],28, 28,1).astype('float32')
    print('X_train_all shape:', X_train_all.shape)
    print('y_train_all shape:', y_train_all.shape)

    X_train_all.shape

    X_train_all = X_train_all / 255


    model = CNN()
    # ,number_of_filters,filter_dimension,stride,padding,number_of_channels,input_dim
    dim = X_train_all.shape[1]
    num_channels = X_train_all.shape[-1]

    conv1 = Convolution_Layer(16,3,1,1,num_channels,dim*dim*num_channels)
    num_channels = 16
    dim = int((dim-3+2*1)/1+1)
    model.add(conv1)
    model.add(Relu_Layer()) 
    dim = int((dim-3+2*0)/1+1) #32 26 26
    model.add(MaxPool(3,1))



    conv2 = Convolution_Layer(32,3,1,1,num_channels,dim*dim*num_channels)
    num_channels = 32
    dim = int((dim-3+2*1)/1+1) #64 24 24
    model.add(conv2)
    model.add(Relu_Layer())
    dim = int((dim-3+2*0)/1+1) #64 22 22
    model.add(MaxPool(3,1)) 


    conv3 = Convolution_Layer(11,3,1,1,num_channels,dim*dim*num_channels)
    num_channels = 11
    dim = int((dim-3+2*1)/1+1) #64 24 24
    model.add(conv3)
    model.add(Relu_Layer())
    dim = int((dim-3+2*0)/1+1) #64 22 22
    model.add(MaxPool(3,1)) 


    model.add(Fully_Connected_Layer(256,dim*dim*num_channels))
    model.add(Relu_Layer())  
    model.add(Fully_Connected_Layer(10,256))
    model.add(Softmax_Layer())
    # Use the trained model to make a prediction
    # y_pred = model.predict(X_train_all,weights,biases)
    # y_pred_label=np.argmax(y_pred,axis=1)

    # validation
    y_pred  = model.validate(X_train_all,y_train_all,weights,biases)
    y_pred_label=np.argmax(y_pred,axis=1)

    count = 0

    for filename in os.listdir(folder_path):

        if count == ind :
            break
        # Add the file name and predicted label to the list of predictions
        predictions.append([filename, y_pred_label[count]])
        count = count + 1

    # Write the predictions to a CSV file
    prediction_df = pd.DataFrame(predictions, columns=["filename", "predicted_label"])
    prediction_df.to_csv("1705006_prediction.csv", index=False)

if __name__ == "__main__":
    main()