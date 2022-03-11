# -*- coding: utf-8 -*-
"""
Created on Sun May  3 23:36:26 2020

@author: Yuhao
"""

import random
import types
from lime.utils.generic_utils import has_arg
from skimage.segmentation import felzenszwalb, slic, quickshift
import copy
from functools import partial

import pickle
import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
from skimage.color import gray2rgb
from progressbar import ProgressBar

import scipy.ndimage as ndi
from skimage.segmentation._quickshift_cy import _quickshift_cython

from lime import lime_base
from lime.wrappers.scikit_image import SegmentationAlgorithm

from keras.models import load_model
import scipy
import scipy.io
import keras
from keras import backend as K
import skimage
import matplotlib.pyplot as plt
from matplotlib import colors
from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.morphology import dilation,square
from collections import Counter


class BaseWrapper(object):
    """Base class for LIME Scikit-Image wrapper
    Args:
        target_fn: callable function or class instance
        target_params: dict, parameters to pass to the target_fn
    'target_params' takes parameters required to instanciate the
        desired Scikit-Image class/model
    """

    def __init__(self, target_fn=None, **target_params):
        self.target_fn = target_fn
        self.target_params = target_params

        self.target_fn = target_fn
        self.target_params = target_params

    def _check_params(self, parameters):
        """Checks for mistakes in 'parameters'
        Args :
            parameters: dict, parameters to be checked
        Raises :
            ValueError: if any parameter is not a valid argument for the target function
                or the target function is not defined
            TypeError: if argument parameters is not iterable
         """
        a_valid_fn = []
        if self.target_fn is None:
            if callable(self):
                a_valid_fn.append(self.__call__)
            else:
                raise TypeError('invalid argument: tested object is not callable,\
                 please provide a valid target_fn')
        elif isinstance(self.target_fn, types.FunctionType) \
                or isinstance(self.target_fn, types.MethodType):
            a_valid_fn.append(self.target_fn)
        else:
            a_valid_fn.append(self.target_fn.__call__)

        if not isinstance(parameters, str):
            for p in parameters:
                for fn in a_valid_fn:
                    if has_arg(fn, p):
                        pass
                    else:
                        raise ValueError('{} is not a valid parameter'.format(p))
        else:
            raise TypeError('invalid argument: list or dictionnary expected')

    def set_params(self, **params):
        """Sets the parameters of this estimator.
        Args:
            **params: Dictionary of parameter names mapped to their values.
        Raises :
            ValueError: if any parameter is not a valid argument
                for the target function
        """
        self._check_params(params)
        self.target_params = params

    def filter_params(self, fn, override=None):
        """Filters `target_params` and return those in `fn`'s arguments.
        Args:
            fn : arbitrary function
            override: dict, values to override target_params
        Returns:
            result : dict, dictionary containing variables
            in both target_params and fn's arguments.
        """
        override = override or {}
        result = {}
        for name, value in self.target_params.items():
            if has_arg(fn, name):
                result.update({name: value})
        result.update(override)
        return result


class SegmentationAlgorithm(BaseWrapper):
    """ Define the image segmentation function based on Scikit-Image
            implementation and a set of provided parameters
        Args:
            algo_type: string, segmentation algorithm among the following:
                'quickshift', 'slic', 'felzenszwalb'
            target_params: dict, algorithm parameters (valid model paramters
                as define in Scikit-Image documentation)
    """

    def __init__(self, algo_type, **target_params):
        self.algo_type = algo_type
        if (self.algo_type == 'quickshift'):
            BaseWrapper.__init__(self, quickshift, **target_params)
            kwargs = self.filter_params(quickshift)
            self.set_params(**kwargs)
        elif (self.algo_type == 'felzenszwalb'):
            BaseWrapper.__init__(self, felzenszwalb, **target_params)
            kwargs = self.filter_params(felzenszwalb)
            self.set_params(**kwargs)
        elif (self.algo_type == 'slic'):
            BaseWrapper.__init__(self, slic, **target_params)
            kwargs = self.filter_params(slic)
            self.set_params(**kwargs)

    def __call__(self, *args):
        return self.target_fn(args[0], **self.target_params)


#def quickshift(image, ratio=1.0, kernel_size=5, max_dist=10,
#               return_tree=False, sigma=0, random_seed=42):   
#    image = np.atleast_3d(image)
#    if kernel_size < 1:
#        raise ValueError("`kernel_size` should be >= 1.")
#
#    image = ndi.gaussian_filter(image, [sigma, sigma, 0])
#    image = np.ascontiguousarray(image * ratio)
#
#    segment_mask = _quickshift_cython(
#        image, kernel_size=kernel_size, max_dist=max_dist,
#        return_tree=return_tree, random_seed=random_seed)
#    return segment_mask



class ImageExplanation(object):
    def __init__(self, image, segments):
        """Init function.
        Args:
            image: 3d numpy array
            segments: 2d numpy array, with the output from skimage.segmentation
        """
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = None

    def get_image_and_mask(self, label, positive_only=True, negative_only=False, hide_rest=False,
                           num_features=5, min_weight=0.):
        """Init function.
        Args:
            label: label to explain
            positive_only: if True, only take superpixels that positively contribute to
                the prediction of the label.
            negative_only: if True, only take superpixels that negatively contribute to
                the prediction of the label. If false, and so is positive_only, then both
                negativey and positively contributions will be taken.
                Both can't be True at the same time
            hide_rest: if True, make the non-explanation part of the return
                image gray
            num_features: number of superpixels to include in explanation
            min_weight: minimum weight of the superpixels to include in explanation
        Returns:
            (image, mask), where image is a 3d numpy array and mask is a 2d
            numpy array that can be used with
            skimage.segmentation.mark_boundaries
        """
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        if positive_only & negative_only:
            raise ValueError("Positive_only and negative_only cannot be true at the same time.")
        segments = self.segments
        image = self.image
        exp = self.local_exp[label]
        mask = np.zeros(segments.shape, segments.dtype)
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.copy()
        if positive_only:
            fs = [x[0] for x in exp
                  if x[1] > 0 and x[1] > min_weight][:num_features]
        if negative_only:
            fs = [x[0] for x in exp
                  if x[1] < 0 and abs(x[1]) > min_weight][:num_features]
        if positive_only or negative_only:
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask
        else:
            for f, w in exp[:num_features]:
                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                mask[segments == f] = 1 #-1 if w < 0 else 1
                temp[segments == f] = image[segments == f].copy()
                #temp[segments == f, c] = np.max(image)
            return temp, mask


class LimeImageExplainer(object):
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, kernel_width=.25, kernel=None, verbose=False,
                 feature_selection='auto', random_state=None):
        """Init function.
        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)

    def explain_instance(self, image, classifier_fn, labels=(1,),
                         hide_color=None,
                         top_labels=5, num_features=10000, num_samples=1000,
                         batch_size=10,
                         segmentation_fn=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None):
        global  yuhao_exp 
        global  yuhao_score 
        global  yuhao_origpred 
        global  yuhao_label
        global  yuhao_distance
        global  yuhao_segment
        """Generates explanations for a prediction.
        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).
        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: TODO
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: TODO
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
            segmentation function
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.
        Returns:
            An ImageExplanation object (see lime_image.py) with the corresponding
            explanations.
        """
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        if segmentation_fn is None:
             #segmentation_fn = SegmentationAlgorithm('felzenszwalb', scale=50, sigma=0.8, min_size = 2, multichannel=True)
#             segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=1,
#                                                    max_dist=5, ratio=0.2,
#                                                    random_seed=random_seed)
            segmentation_fn = SegmentationAlgorithm('slic',n_segments=5000, compactness=1000, max_iter=5, sigma=0.8)
        try:
           segments = segmentation_fn(image)
           yuhao_segment = segments 
        except ValueError as e:
            raise e

        fudged_image = image.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2]))
        else:
            fudged_image[:] = hide_color

        top = labels

        data, labels = self.data_labels(image, fudged_image, segments,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size)

        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()
        yuhao_distance = distances

        ret_exp = ImageExplanation(image, segments)
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection='auto')
#            print(top)
            yuhao_label = labels
            yuhao_exp = ret_exp.local_exp
            yuhao_score = ret_exp.score
            yuhao_origpred = ret_exp.local_pred
#            print(ret_exp.local_exp)
#            print(ret_exp.score)
#            print(ret_exp.local_pred)
            #self.feature_selection
        return ret_exp


    def data_labels(self,
                    image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=10):
        """Generates images and predictions in the neighborhood of this image.
        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.
        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        n_features = np.unique(segments).shape[0]
        data = self.random_state.randint(0, 2, num_samples * n_features)\
            .reshape((num_samples, n_features))
        labels = []
        data[0, :] = 1
        imgs = []
        pbar = ProgressBar(num_samples)
        pbar.start()
        for row in data:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                preds = classifier_fn(np.array(imgs))
                labels.extend(preds)
                imgs = []
            pbar.currval += 1
            pbar.update()
        pbar.finish()
        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)
        return data, np.array(labels)


global  yuhao_exp 
global  yuhao_score 
global yuhao_origpred 
global yuhao_label
global  yuhao_distance
global  yuhao_segment
num_classes = 2

best_model = load_model('bestmodel_sq600_clean_50x_500.h5')
X_train = np.load('x_sq600_clean_50x_py.npy')
Y_train = np.load('y_300_py.npy')
    
count = []
score = []
high_r_s = []
##blue
#high_r = [252, 47, 74, 291, 38, 285, 96, 25, 281, 14, 131, 15, 237, 70, 51, 196, 130, 183, 58, 55, 124, 274, 23, 297, 160]
##red
#high_r_s = [291, 158, 20, 43, 190, 16, 71, 89, 229, 236, 3, 29, 290, 230, 122, 9, 220, 128, 114, 203, 151, 80, 282, 238, 271]
#number = len(high_r_s)
#high_r_s = [x + 300 for x in high_r]

for i in range(300,600):
   high_r_s.append(i)
sd = 1
num_sam = 300

mi = np.min(high_r_s)
ma = np.max(high_r_s)
if  ma < 300:
    reg = 0
    region = 'T'
if mi >=300:
    reg = 1
    region = 'B'
    
for i in high_r_s:
    np.random.seed(i+sd)
    random.seed(i+sd)
    x_test = X_train[i,:,:,0].copy()
    explainer = LimeImageExplainer()
    explanation = explainer.explain_instance(x_test, best_model.predict, top_labels=num_classes, hide_color = 0,num_samples=num_sam)
    score.append(yuhao_score)
    
    if yuhao_score>= 0.8 and yuhao_score<1:
        
      features = [a[0] for a in yuhao_exp[reg][:375]]
#      importances = [a[1] for a in yuhao_exp[reg][:375]]
#      index = [a for a, b in enumerate(importances)]        
#      posi_features = [features[c] for c in index]
      count.append(features)
      
new_count = sum(count, [])
count = Counter(new_count)
idx = sorted(count.keys())
y = [count[i] for i in idx]
plt.bar(idx,y,width = 0.6)  
#
ymax = max(y)
idx_2 = [i for i, j in enumerate(y) if j == ymax]
for i in idx_2:
     plt.gca().text(idx[i], y[i], idx[i], color='k',fontsize = 5)
     print(idx[i])
     
y2 = y.copy()
for k in idx_2:
   y2.remove(y[k])
ymax2 = max(y2)
idx_3 = [i for i, j in enumerate(y) if j == ymax2]
for i in idx_3:
     plt.gca().text(idx[i], y[i], idx[i],color = 'r',fontsize = 5)
     print(idx[i])
#     
y3 = y2.copy()
for k in idx_3:
   y3.remove(y[k])
ymax3 = max(y3)
idx_4 = [i for i, j in enumerate(y) if j == ymax3]
for i in idx_4:
     plt.gca().text(idx[i], y[i], idx[i],color = 'darkgreen',fontsize = 5)
     print(idx[i])
#
y4 = y3.copy()
for k in idx_4:
   y4.remove(y[k])
ymax4 = max(y4)
idx_5 = [i for i, j in enumerate(y) if j == ymax4]
for i in idx_5:
     plt.gca().text(idx[i], y[i], idx[i],color = 'saddlebrown',fontsize = 5) 
     print(idx[i])
#     
y5 = y4.copy()
for k in idx_5:
   y5.remove(y[k])
ymax5 = max(y5)
idx_6 = [i for i, j in enumerate(y) if j == ymax5]
for i in idx_6:
     plt.gca().text(idx[i], y[i], idx[i],color = 'b',fontsize = 5)
     print(idx[i])

plt.xlabel('Frequency ranges')
plt.ylabel('Count')
plt.savefig('sq600_50x_'+region+'_abs_sd'+str(sd)+'_'+str(num_sam)+'.png', dpi=500)

with open('sq600_50x_'+region+'_abs_count_sd'+str(sd)+'_'+str(num_sam)+'.data', 'wb') as filehandle:
    pickle.dump(new_count, filehandle)
with open('sq600_50x_'+region+'_abs_score_sd'+str(sd)+'_'+str(num_sam)+'.data', 'wb') as filehandle:
    pickle.dump(score, filehandle)



