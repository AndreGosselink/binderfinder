Examples
--------
The binderfinder employs two main features: PCA analysis and data reduction of multi dimensional data into an RGB Matrix. In both cases the data needs to be pre-processed into a certain csv format, see :ref:`fileformat`. For the mapping of multidimensional data into RGB see :ref:`evalfunc`

Matrix
""""""

.. code-block:: python

   defaults = dict(filename='./data/iris_dataset/iris.data',
                annotate='none',
                   stats=True,
                    sort='none',
                  legend='bg',
                    ceil=True,
               normalize='channels',
               ch_labels=['red', 'Rnd', 'Rnd'],
               )

   start_binderfinder(defaults)

PCA
"""

.. code-block:: python

    params = dict(annotate=False,
                 normalize=True,
                   covplot=True,
                  portions=True,
                 )
    
    p = PCA('./data/mock_data_pca_rnd.csv', annotate=True, normalize=False, covplot=True)
    p.show()


.. _fileformat:
Fileformat
----------
The general fileformat is defined by a header and a data section. The header has the general fromat::

    properties;n
    parameters;m

where n is the number of properties and m is the number of parameters. With properties, the a-priori known properties of a sample are known. A property of a sample would translate to a gen modification, treatment e.g. The properties are the label of the data or in other terms: properties are those words, numbers and sequences you'd write onto your falcon tube or eppi, describing your sample.

Leaving us with the parameters m. This is the number of parameters you are measuring. This referres to your number of readouts e.g. channels used in flowcytometry. Optionally the header can have the format::

    properties;n
    parameters;ma;name_1;name_2;...;name_m

where name are the description od the measured parameter (e.g. Intensity, weight, velocity,...)

A file for Matrix Analysis would look like this::

    # header
    properties;2
    parameters;2;par1;par2
    # data
    A;a;2.1;0.25
    A;b;2.2;0.89
    A;c;2.3;0.98
    A;d;2.4;0.57

Lines starting with an '#' will be ignored. There are two properties defined in the header, thus the first two field in each row are the labels. E.g. in the first data row we would have sample Aa, in the second Ab, Ac,... The following fields are the values, measuerd for the parameter par1, par2,... 

A file for PCA would look like this, showing the iris dataset::

    properties;0
    parameters;4;sepal_length;sepal_width;petal_length;petal_width;class
    5.1;3.5;1.4;0.2;0
    4.9;3.0;1.4;0.2;0
    4.7;3.2;1.3;0.2;0
    4.6;3.1;1.5;0.2;0

Properties needs to be '0' for PCA, thus in the data segment, there are only parameters shown.

.. _evalfunc:
Evaluation/RGB mapping
----------------------
During Matrix creation, upt to thre functions are called. Those functions are externalized for simple access and can be found in *.\\binderfinder\\evaluate.py*

.. currentmodule:: binderfinder.evaluate
.. autofunction:: evaluate
.. autofunction:: stats_calculation
.. autofunction:: sort_reduction
