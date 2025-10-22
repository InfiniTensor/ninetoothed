Visualization
=============

This module provides tools to generate useful visualizations.

Installation
------------

Since this module is optional, we need to install it before using it:

.. code-block::

    pip install ninetoothed[visualization]

Visualizing a Tensor
--------------------

``visualize`` can be used to visualize a single tensor:

.. autofunction:: ninetoothed.visualization.visualize

Basic Usage
^^^^^^^^^^^

If we just want to temporarily visualize a tensor, the simplest way is to pass the tensor to ``visualize``.

.. code-block:: python

    x = Tensor(shape=(4, 8))
    visualize(x)

To save the visualization to a file, provide the ``save_path``. Then, ``visualize`` will save the generated image at the specified location.

.. code-block:: python

    x = Tensor(shape=(4, 8))
    visualize(x, save_path="x.png")

This method can also be used to temporarily visualize multiple tensors. You just need to pass the corresponding tensors and save paths to ``visualize``.

.. code-block:: python

    x = Tensor(shape=(4, 8))
    visualize(x, save_path="x.png")

    y = Tensor(shape=(8, 4))
    visualize(y, save_path="y.png")

    z = Tensor(shape=(4, 4))
    visualize(z, save_path="z.png")

Specifying Colors
^^^^^^^^^^^^^^^^^

When using the above method, tensors are assigned default colors. To customize colors, we can use the ``color`` parameter, following `Matplotlib's color formats <https://matplotlib.org/stable/users/explain/colors/colors.html>`_.

.. code-block:: python

    x = Tensor(shape=(4, 8))
    visualize(x, color="orange", save_path="x.png")

Visualizing an Arrangement
--------------------------

``visualize_arrangement`` can be used to visualize an arrangement:

.. autofunction:: ninetoothed.visualization.visualize_arrangement

Note: Currently, this API must be run in a CUDA-available environment.

This API requires ``arrangement`` and ``tensors`` as parameters, similar to :func:`ninetoothed.make`. However, there is a key difference: neither ``arrangement`` nor ``tensors`` can contain symbols; they can only contain concrete values.

As demonstrated below, we can use ``functools.partial`` to pass concrete arguments to the ``_arrangement`` function, constructing a symbol-free ``arrangement``. In addition, we can create a tensor with a concrete shape using the ``shape`` parameter of ``Tensor``.

.. code-block:: python

    def _arrangement(tensor, tile_shape):
        return (tensor.tile(tile_shape),)


    arrangement = functools.partial(_arrangement, tile_shape=(2, 2))
    tensors = (Tensor(shape=(5, 5)),)

    visualize_arrangement(arrangement, tensors)
