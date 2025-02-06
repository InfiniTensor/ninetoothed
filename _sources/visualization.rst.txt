Visualization
=============

The ``visualization`` module provides a way to visualize tensors.

Installation
------------

Since this module is optional, we need to install it before using it:

.. code-block::

    pip install ninetoothed[visualization]

Visualizing a Tensor
--------------------

``visualize`` is the primary function of this module:

.. autofunction:: ninetoothed.visualization.visualize

Basic Usage
^^^^^^^^^^^

If we just want to temporarily visualize a tensor, the simplest way is to pass the tensor to ``visualize`` and provide a save path. Then, ``visualize`` will save the generated image at the specified location.

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
