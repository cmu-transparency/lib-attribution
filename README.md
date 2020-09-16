# lib-attribution
Library providing attribution methods for Keras models. Supports Tensorflow and Theano backends.

This code is provided as-is, and is intened simply to accompany the paper, "Influence-directed Explanations for Convolutional Neural Networks" [(arXiv)](https://arxiv.org/pdf/1802.03788.pdf).

If using the code from this repository, please use the following citation:

```
@INPROCEEDINGS{
  leino18influence,
  author={Klas Leino and Shayak Sen and Anupam Datta and Matt Fredrikson and Linyi Li},
  booktitle={2018 IEEE International Test Conference (ITC)},
  title={Influence-Directed Explanations for Deep Convolutional Networks},
  year={2018},
}
```



## Requirements

  - Keras (cannot assume tensorflow or theano)

  - Python2 and Python3 compatible.

    TODO: Check if typing can be imported in python2 from the future.

## Top-level interface

- ```AttributionMethod``` (abstract interface)

  Quantify attribution per input.

  - Required methods:

	- ```compile```

    - ```get_attributions```

	  Immediately compute the attributions.

  - Optional methods:

    - ```get_sym_attributions```

	  Symbolic tensor representation of what get_attributions computes. Consistent with Keras
      backend (not general tensorflow).

  - Example implementations in methods.py

  - ```InternalInfluence```

    - ```IntegratedGradients```

    - ```SaliencyMaps```

	- TODO: ```SmoothGradients```

  - ```AumannShapley``` (deprecated, kept for regression testing)

  - ```Conductance``` (same people who did Integrated Gradients)

  - ```Activation```

  - TODO: Guided back-propagation

- ```VisualizationMethod```

  - Required methods:

    - ```visualize```

	- ```mask```

  - TODO: LocalizationMethod (may be top-level interface)

    May just be a map from method-specific-shaped attributions to input-shaped attribution.
    Example: upscale for GradCAM.

  - TODO: Visualization Combinators

    - Compose localizers, visualizers

  - TODO: AggregatedVisualization

    - TODO: GradCAM

    - TODO: Channelwise IDE

  - ```UnitsWithBlur``` (should be in visualizations.py)

    - Visualize a given set of units as the parts of input that have the most influence on them.
      Also add blur and threshold.

  - ```TopKWithBlur```

## Testing

  - ```tests/all_tests.py```

  - ```tests/influence_tests.py```

    Several hard-coded models. Example: ```basic_twoclass_model``` .
