# -*- coding: utf-8 -*-
"""
Created on Sun May  3 23:36:26 2020

@author: Yuhao
"""
#%%
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
from sklearn.linear_model import Ridge, lars_path

import scipy.ndimage as ndi
from skimage.segmentation._quickshift_cy import _quickshift_cython

# from lime import lime_base
from lime.wrappers.scikit_image import SegmentationAlgorithm

from tensorflow.keras.models import load_model
import scipy as sp
import scipy.io
import tensorflow.keras
from tensorflow.keras import backend as K
import skimage
import matplotlib.pyplot as plt
from matplotlib import colors
from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.morphology import dilation,square
from collections import Counter
from scipy.io import savemat

#%%
class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):
        """Init function
        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.
        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel
        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            clf = Ridge(alpha=0.01, fit_intercept=True,
                        random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)

            coef = clf.coef_
            if sp.sparse.issparse(data):
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                # Note: most efficient to slice the data before reversing
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()
                # Edge case where data is more sparse than requested number of feature importances
                # In that case, we just pad with zero-valued features
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                    indices_set = set(indices)
                    pad_counter = 0
                    for i in range(data.shape[1]):
                        if i not in indices_set:
                            indices[pad_counter + sdata] = i
                            pad_counter += 1
                            if pad_counter >= num_to_pad:
                                break
                else:
                    nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                    indices = weighted_data.indices[nnz_indexes]
                return indices
            else:
                weighted_data = coef * data[0]
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: np.abs(x[1]),
                    reverse=True)
                return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method)

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='auto',
                                   model_regressor=None):
        """Takes perturbed data, labels and distances, returns explanation.
        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()
        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
            local_pred is the prediction of the explanation model on the original instance
        """

        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)
        ###################################################################################
        if model_regressor is None:
            model_regressor = Ridge(alpha=50, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model = model_regressor
        easy_model.fit(neighborhood_data[:, used_features],
                       labels_column, sample_weight=weights)
        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            labels_column, sample_weight=weights)

        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))

        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0, label])
        return (easy_model.intercept_,
                sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: np.abs(x[1]), reverse=True),
                prediction_score, local_pred)

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
    #########################################################################################################################
    def __init__(self, kernel_width=1000, kernel=None, verbose=False,
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
        self.base = LimeBase(kernel_fn, verbose, random_state=self.random_state)

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
            segmentation_fn = SegmentationAlgorithm('slic',n_segments=200, compactness=100, max_iter=10, sigma=0.8)
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
        global yuhao_image
        
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
                yuhao_image = imgs
                imgs = []
            pbar.currval += 1
            pbar.update()
        pbar.finish()
        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)
        return data, np.array(labels)

#%%
global  yuhao_exp 
global  yuhao_score 
global yuhao_origpred 
global yuhao_label
global  yuhao_distance
global  yuhao_segment
global  yuhao_image
num_classes = 2

# with open('Our_wrong_test_id'+'.data', 'rb') as fp:
#     Our_wrong_id =  pickle.load(fp)

best_model = load_model('bestmodel_sq600_clean_10x_our.h5')
X_train = np.load('x_sqOur_600_py.npy')
Y_train = np.load('y_300_py.npy')
    
count = []
score = []
high_r_s = []

# for i in range(0,len(Our_wrong_id)):
#    high_r_s.append(i)

# mi = np.min(high_r_s)
# ma = np.max(high_r_s)
for i in range(300,600):
   high_r_s.append(i)

mi = np.min(high_r_s)
ma = np.max(high_r_s)
if  ma < 300:
    reg = 0
    region = 'T'
if mi >=300:
    reg = 1
    region = 'B'

sd = 700
for i in high_r_s:
    np.random.seed(i+sd)
    random.seed(i+sd)
    x_test = X_train[i,:,:,0].copy()
    explainer = LimeImageExplainer()
    explanation = explainer.explain_instance(x_test, best_model.predict, top_labels=num_classes, hide_color = 0,num_samples=250)
    score.append(yuhao_score)
    
    if yuhao_score>= 0.7 and yuhao_score<1:
        
      features = [a[0] for a in yuhao_exp[reg][:15]]
      importances = [a[1] for a in yuhao_exp[reg][:15]]
      index = [a for a, b in enumerate(importances) if b > 0]        
      posi_features = [features[c] for c in index]
      count.append(posi_features)
      
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
#y4 = y3.copy()
#for k in idx_4:
#   y4.remove(y[k])
#ymax4 = max(y4)
#idx_5 = [i for i, j in enumerate(y) if j == ymax4]
#for i in idx_5:
#     plt.gca().text(idx[i], y[i], idx[i],color = 'saddlebrown',fontsize = 5) 
#     print(idx[i])
#     
#y5 = y4.copy()
#for k in idx_5:
#   y5.remove(y[k])
#ymax5 = max(y5)
#idx_6 = [i for i, j in enumerate(y) if j == ymax5]
#for i in idx_6:
#     plt.gca().text(idx[i], y[i], idx[i],color = 'b',fontsize = 5)
#     print(idx[i])

plt.xlabel('Frequency ranges')
plt.ylabel('Count')

plt.savefig('hyper tuning\lambda_0_N_250_delta_25_'+region+'_posi_ds'+str(sd)+'.png', dpi=400)

with open('hyper tuning\lambda_0_N_250_delta_25_'+region+'_posi_count_sd'+str(sd)+'.data', 'wb') as filehandle:
    pickle.dump(new_count, filehandle)
# with open('hyper tuning\lambda_1_N_250_delta_0_75_'+region+'_posi_score_sd'+str(sd)+'.data', 'wb') as filehandle:
#     pickle.dump(score, filehandle)

FrameStack = np.empty((len(yuhao_image),), dtype=np.object)
for i in range(len(yuhao_image)):
    FrameStack[i] = yuhao_image[i]

savemat('hyper tuning\lambda_0_N_250_delta_25_'+region+'_posi_ds'+str(sd)+'.mat', {"FrameStack":FrameStack})

#plt.savefig('sq600_10x_our_clean_'+region+'_posi_ds'+str(sd)+'.png', dpi=400)
#
#with open('sq600_10x_our_'+region+'_posi_count_sd'+str(sd)+'.data', 'wb') as filehandle:
#    pickle.dump(new_count, filehandle)
#with open('sq600_10x_our_'+region+'_posi_score_sd'+str(sd)+'.data', 'wb') as filehandle:
#    pickle.dump(score, filehandle)

# loca = np.zeros((11,21))
# for i in [0,14,15,33,34,37,17,29,20,26,44,22,45,13,23,21]:
#       loca[yuhao_segment==i] = None
# plt.imshow(loca)

# lala = np.load('x_osq300_py.npy')
# color_max = max(np.max(X_train),np.max(lala))
# color_min = min(np.min(X_train),np.min(lala))

# plt.imshow(X_train[2,:,:,0],vmax = color_max, vmin = color_min)
# plt.imshow(lala[2,:,:,0],vmax = color_max, vmin = color_min)
# plt.colorbar()
