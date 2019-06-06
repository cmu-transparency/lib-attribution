#%%
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
# get_ipython().run_line_magic('matplotlib', 'inline')

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
model.load_weights('examples/weights/lfw-small-tf.h5')

#%%
print('accuracy:')
print('train={:.2}'.format(model.evaluate(im_tr, y_tr, verbose=False)[1]))
print('test={:.2}'.format(model.evaluate(im_te, y_te, verbose=False)[1]))

#%%
from attribution.model_utils import replace_softmax_with_logits
model = replace_softmax_with_logits(model)

#%% [markdown]
# # Finding invariants
#%% [markdown]
# Now let's find some invariants of the model. The main class for doing so is ``ActivationInvariants``.


#%%
from attribution.ActivationInvariants import ActivationInvariants
from attribution.InfluenceInvariants import InfluenceInvariants

#%%
for layer in range(1,len(model.layers)-1):
    actinv = ActivationInvariants(model, layers=[layer], agg_fn=None).compile()
    infinv = InfluenceInvariants(model, layer=layer, agg_fn=None).compile()

    invs_act = actinv.get_invariants(im_tr, batch_size=1)
    invs_inf = infinv.get_invariants(im_tr, batch_size=1)

    supports_act = [inv.support for inv in invs_act]
    supports_inf = [inv.support for inv in invs_inf]
    print('layer {}, act: #={}, avg supp.: {:.2}'.format(layer, len(invs_act), np.array(supports_act).mean()))
    print('layer {}, inf: #={}, avg supp.: {:.2}'.format(layer, len(invs_inf), np.array(supports_inf).mean()))
    print('-'*20)


#%%


#%%


#%%
