#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'examples'))
	print(os.getcwd())
except:
	pass


#%%
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["KERAS_BACKEND"]="tensorflow"


#%%
import sys
sys.path.append('..')


#%%
import warnings
warnings.simplefilter("ignore")


#%%
import numpy as np

import keras
import keras.backend as K


#%%
K.set_image_data_format('channels_last')


#%%
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# helper to simplify displaying multiple images
def imshow(image, width=64, height=64, size=None):
    
    im = np.array(image, copy=True)
    if image.min() < 0:
        im -= im.min()
    im /= im.max()
    im = (im*255.).astype(np.uint8)
    if np.ndim(im) == 3:
        im = np.expand_dims(im, 0)
 
    n = len(im)
    s = int(np.ceil(np.sqrt(n)))
    
    if size is None:
        size = 2*s
    
    fig, axs = plt.subplots(s, s)
    fig.set_size_inches(size,size)
    if s == 1:
        axs.imshow(im[0])
    else:
        cnt = 0
        for i in range(s):
            for j in range(s):
                if cnt < n:
                    axs[i,j].imshow(im[cnt, :,:,:])
                axs[i,j].axis('off')
                axs[i,j].set_aspect('equal')
                cnt += 1
        fig.subplots_adjust(wspace=0, hspace=0.)

    plt.show()


#%%
import tempfile

def replace_softmax_with_logits(model):
    model.layers[-1].activation = keras.activations.linear
    tmp_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
    try:
        model.save(tmp_path)
        return keras.models.load_model(tmp_path)
    finally:
        os.remove(tmp_path)

#%% [markdown]
# # Target model: small Labeled Faces in the Wild (LFW)

#%%
from sklearn.datasets import fetch_olivetti_faces, fetch_lfw_people

# Use only classes that have at least 100 images
# There are five such classes in LFW
lfw_slice = (slice(68, 196, None), slice(61, 190, None))
faces_data = fetch_lfw_people(min_faces_per_person=100, color=True, slice_=lfw_slice)
images = faces_data.images
n_classes = faces_data.target.max()+1
x, y = faces_data.data, keras.utils.to_categorical(faces_data.target, n_classes)
images /= 255.0

# Use 3/4 for training, the rest for testing
N_tr = int(len(x)*0.75)
N_te = len(x) - N_tr
x_tr, y_tr = x[:N_tr], y[:N_tr]
x_te, y_te = x[N_tr:], y[N_tr:]
im_tr, im_te = images[:N_tr], images[N_tr:]


#%%
imshow(images[:36])

#%% [markdown]
# We'll use a small convnet to demonstrate, but with enough depth for interesting results

#%%
inp = keras.layers.Input(shape=im_tr[0].shape)
out = keras.layers.Conv2D(128, (3,3), activation='relu')(inp)
out = keras.layers.MaxPooling2D(pool_size=(2,2))(out)
out = keras.layers.Conv2D(64, (3,3), activation='relu')(out)
out = keras.layers.MaxPooling2D(pool_size=(2,2))(out)
out = keras.layers.Conv2D(32, (3,3), activation='relu')(out)
out = keras.layers.MaxPooling2D(pool_size=(2,2))(out)
out = keras.layers.Conv2D(16, (3,3), activation='relu')(out)
out = keras.layers.MaxPooling2D(pool_size=(2,2))(out)
out = keras.layers.Flatten()(out)
out = keras.layers.Dense(16, activation='relu')(out)
out = keras.layers.Dense(y[0].shape[0], activation='softmax')(out)
model = keras.Model(inp, out)
model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['acc'])
model.summary()

#%% [markdown]
# The model was originally trained with:
# 
# `model.fit(im_tr, y_tr, batch_size=32, epochs=40, validation_data=(im_te, y_te))`

#%%
model.load_weights('weights/lfw-small-tf.h5')


#%%
print('accuracy:')
print('train={:.2}'.format(model.evaluate(im_tr, y_tr, verbose=False)[1]))
print('test={:.2}'.format(model.evaluate(im_te, y_te, verbose=False)[1]))


#%%
model = replace_softmax_with_logits(model)

#%% [markdown]
# # Finding invariants
#%% [markdown]
# Now let's find some invariants of the model. The main class for doing so is ``ActivationInvariants``.

#%%
get_ipython().run_cell_magic('time', '', 'from attribution.ActivationInvariants import ActivationInvariants\n\nmaxinv = ActivationInvariants(model, agg_fn=K.max).compile()')


#%%
get_ipython().run_cell_magic('time', '', 'maxinvs = maxinv.get_invariants(im_tr)')

#%% [markdown]
# There are quite a few invariants that it found, but most of them have small support.
# 
# We define support as the percentage of samples with the relevant $Q$ value covered by the predicate. For example, using the default $Q$ which corresponds to the class label, the support is the percentage of the class that matches the invariant. This makes it possible to specify support without having to normalize for dataset size.
# 
# Let's look at a subset of them, and see the average support.

#%%
print('# invariants:', len(maxinvs), '\n')
for inv in maxinvs[:5]:
    print(inv, '\n')
supports = [inv.support for inv in maxinvs]
print('avg support: {:.2}'.format(np.array(supports).mean()))

#%% [markdown]
# We can tell ``get_invariants`` to only return invariants with at least a minimum threshold of support. Likewise for precision.

#%%
get_ipython().run_cell_magic('time', '', 'maxinvs_highsup = maxinv.get_invariants(im_tr, min_support=0.5)\nprint(len(maxinvs_highsup))')

#%% [markdown]
# ### Flat aggregation of convolutional layers
#%% [markdown]
# It appears difficult to find high-support invariants for this model. This might be due to the fact that we use ``K.max`` to aggregate the convolutional layers, resulting in many fewer features for the decision tree learner to use.

#%%
get_ipython().run_cell_magic('time', '', 'flatinv = ActivationInvariants(model, agg_fn=None).compile()')

#%% [markdown]
# Note that this will take significantly longer to find invariants, because there are now so many features

#%%
get_ipython().run_cell_magic('time', '', 'flatinvs = flatinv.get_invariants(im_tr, min_support=0.5)\nprint(len(flatinvs))')


#%%
for inv in flatinvs[:5]:
    print(inv, '\n')
supports = [inv.support for inv in flatinvs]
print('avg support: {:.2}'.format(np.array(supports).mean()))

#%% [markdown]
# In general, we would not expect flattened features in convolutional layers to generalize to new data.
# Quite simply, they are location-specific, and so very sensitive to non-essential aspects of the image.
# 
# We can see this plainly on the second invariant by evaluating it on training and test data.
# The ``Invariant`` class has a ``get_executable`` method that compiles the predicate for the backend given in ``K.backend()``.

#%%
get_ipython().run_cell_magic('time', '', 'flatinv1_f = flatinvs[1].get_executable()\nflatinv1_tr_inds = np.where(flatinv1_f(im_tr))\nflatinv1_te_inds = np.where(flatinv1_f(im_te))')

#%% [markdown]
# The invariant has precision=1.0 on the training data, so we would expect the model to predict label 4 for all of the instances satisfying this predicate (those indices are in ``flatinv1_tr_inds``).

#%%
model.predict(im_tr[flatinv1_tr_inds]).argmax(axis=1)

#%% [markdown]
# The question is how many test instances satisfying the predicate are given label 4 by the model?
# As we see below, this is not a great predicate.

#%%
model.predict(im_te[flatinv1_te_inds]).argmax(axis=1)

#%% [markdown]
# Perhaps we can find a better set of high-support invariants by looking higher in the network.
# ``AttributionInvariant`` lets us pass in a list of layers to gather features from.
# We'll try from ``max_pooling2d_2`` through ``dense_1``.

#%%
get_ipython().run_cell_magic('time', '', 'flatinv = ActivationInvariants(model, layers=list(range(4,11)), agg_fn=None).compile()')


#%%
get_ipython().run_cell_magic('time', '', 'flatinvs = flatinv.get_invariants(im_tr, min_support=0.5)\nprint(len(flatinvs))')


#%%
for inv in flatinvs[:5]:
    print(inv, '\n')
supports = [inv.support for inv in flatinvs]
print('avg support: {:.2}'.format(np.array(supports).mean()))


#%%
get_ipython().run_cell_magic('time', '', "flatinv0_f = flatinvs[0].get_executable()\nflatinv0_te_inds = np.where(flatinv0_f(im_te))\nflatinv0_te_acc = np.mean(model.predict(im_te[flatinv0_te_inds]).argmax(axis=1) == flatinvs[0].Q)\nprint('inv[0] test precision: {:.2}'.format(flatinv0_te_acc))\nflatinv1_f = flatinvs[1].get_executable()\nflatinv1_te_inds = np.where(flatinv1_f(im_te))\nflatinv1_te_acc = np.mean(model.predict(im_te[flatinv1_te_inds]).argmax(axis=1) == flatinvs[1].Q)\nprint('inv[1] test precision: {:.2}'.format(flatinv1_te_acc))")

#%% [markdown]
# Still not great.
# Let's go back to aggregating with ``K.max``, but still use the same range of layers.
# Finding the right set of layers can help with invariant quality, much in the same way that feature selection can help with model quality when using decision tree learning.

#%%
get_ipython().run_cell_magic('time', '', "maxinv = ActivationInvariants(model, layers=list(range(4,11)), agg_fn=K.max).compile()\nmaxinvs = maxinv.get_invariants(im_tr, min_support=0.2)\nfor inv in maxinvs[:5]:\n    print(inv, '\\n')\nsupports = [inv.support for inv in maxinvs]\nprint('avg support: {:.2}\\n'.format(np.array(supports).mean()))")

#%% [markdown]
# Interestingly, the invariants focus only on the flat layers.
# As we see below, the invariants appear to be of significantly higher quality.

#%%
get_ipython().run_cell_magic('time', '', "inv0_f = maxinvs[0].get_executable()\ninv0_te_inds = np.where(inv0_f(im_te))\ninv0_te_acc = np.mean(model.predict(im_te[inv0_te_inds]).argmax(axis=1) == maxinvs[0].Q)\nprint('inv[0] test precision: {:.2}'.format(inv0_te_acc))\ninv1_f = maxinvs[1].get_executable()\ninv1_te_inds = np.where(inv1_f(im_te))\ninv1_te_acc = np.mean(model.predict(im_te[inv1_te_inds]).argmax(axis=1) == maxinvs[1].Q)\nprint('inv[1] test precision: {:.2}'.format(inv1_te_acc))")

#%% [markdown]
# # Visualizing invariants

#%%
get_ipython().run_cell_magic('time', '', 'from attribution import visualizations\n\nvis_f = visualizations.UnitsWithBlur(maxinv._attributers[-2], [57])\nvis_d = visualizations.UnitsWithBlur(maxinv._attributers[-1], [0,2,9,10])')


#%%
get_ipython().run_cell_magic('time', '', 'vis_f_ims = vis_f.visualize(im_te[inv1_te_inds])\nvis_d_ims = vis_d.visualize(im_te[inv1_te_inds])')

#%% [markdown]
# First we visualize the dense units, which appear to correspond to Powell's nose.

#%%
imshow(np.concatenate((im_te[inv1_te_inds],vis_d_ims), axis=0), size=12)

#%% [markdown]
# Does this set of neurons light up on other people's nose?
# The result directly below shows that it generally does, although not as consistently as with Powell.
# The fact that this collection of neurons is highly predictive of Powell suggests that this neuron is influential for him, and not for others.

#%%
vis_d_ims = vis_d.visualize(im_te[:18])
imshow(np.concatenate((im_te[:18],vis_d_ims), axis=0), size=14)

#%% [markdown]
# Before visualizing the part of the invariant from ``flatten_1``, let's follow up on our hypothesis about influence.
# 
# We specify a quantity of interest corresponding to the logit for the quantity (class) of invariant 1.
# Then we construct an internal attributer for ``dense_1`` using this quantity, and get the attributions for each of the instances that satisfy the invariant.

#%%
get_ipython().run_cell_magic('time', '', "from attribution import methods\n\ndense_1 = model.get_layer(name='dense_1')\nQ_d1_l0 = model.layers[-1].output[:,maxinvs[1].Q]\ninfl_d1 = methods.AumannShapley(model, dense_1, Q=Q_d1_l0).compile()\nd1_attrs = infl_d1.get_attributions(im_te[inv1_te_inds], resolution=100)")

#%% [markdown]
# Now that we have the attributions for these instances towards prediction as Powell, we want to see if the top-attributed neurons intersect with those mentioned in the predicate at this layer.
# 
# As the results below show, this is not the case.

#%%
flat_order_infl = d1_attrs.argsort(axis=1)[::-1][:,:5]
units = np.array([0,2,9,10])
def intersect(a):
    return len(np.intersect1d(units, a))
np.apply_along_axis(intersect, 1, flat_order_infl)

#%% [markdown]
# We can do the same thing with conductance. 
# There is very slightly more intersection, but not much.

#%%
get_ipython().run_cell_magic('time', '', 'cond_d1 = methods.Conductance(model, dense_1, Q=Q_d1_l0).compile()\nd1_conds = cond_d1.get_attributions(im_te[inv1_te_inds])')


#%%
flat_order_cond = d1_conds.argsort(axis=1)[::-1][:,:5]
units = np.array([0,2,9,10])
def intersect(a):
    return len(np.intersect1d(units, a))
np.apply_along_axis(intersect, 1, flat_order_cond)

#%% [markdown]
# Overall, there is more agreement between influence and conductance than either with invariants.

#%%
[len(np.intersect1d(flat_order_infl[a], flat_order_cond[a])) for a in range(10)]

#%% [markdown]
# And now we visualize the unit identified at ``flatten_1``

#%%
imshow(np.concatenate((im_te[inv1_te_inds], vis_f_ims), axis=0), size=12)

#%% [markdown]
# Strange, it looks like there aren't any pixels in these images that are influential on the flat unit identified in the invariant.
# This could have to do with the fact that the unit is constrained to 0 in the invariant, so it is always turned off in the relevant images.
# 
# Let's see if it shows up in some other random images.

#%%
vis_f_ims = vis_f.visualize(im_te[:18])


#%%
imshow(np.concatenate((im_te[:18], vis_f_ims), axis=0), size=14)

#%% [markdown]
# Hard to tell what this neuron represents, but there is some consistency with activating on nose+mouth or on a collared shirt.

