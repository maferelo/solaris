Deployment
==========

.. _heroku_deployment:


Heroku
------

First steps
***********

.. code-block:: bash

 $ sudo apt-get install build-essential python3-dev python3-pip python3-cffi libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info
 $ sudo apt-autoremove

Then activate the env

 $ source Environments/solaris/bin/activate
 $ cd Apps/solaris
 $ pip install -r requirements.txt

Install geospatial libraries

GEOS

 $ wget http://download.osgeo.org/geos/geos-3.4.2.tar.bz2
 $ tar xjf geos-3.4.2.tar.bz2

 $ cd geos-3.4.2
 $ ./configure
 $ make
 $ sudo make install
 $ cd ..

PROJ.4

 $ wget http://download.osgeo.org/proj/proj-4.9.1.tar.gz
 $ wget http://download.osgeo.org/proj/proj-datumgrid-1.5.tar.gz

 $ tar xzf proj-4.9.1.tar.gz
 $ cd proj-4.9.1/nad
 $ tar xzf ../../proj-datumgrid-1.5.tar.gz
 $ cd ..

 $ ./configure
 $ make
 $ sudo make install
 $ cd ..

GDAL

 $ wget http://download.osgeo.org/gdal/1.11.2/gdal-1.11.2.tar.gz
 $ tar xzf gdal-1.11.2.tar.gz
 $ cd gdal-1.11.2

 $ ./configure
 $ make # Go get some coffee, this takes a while.
 $ sudo make install
 $ cd ..


.. note::
 https://saleor.readthedocs.io/en/latest/gettingstarted/installation-linux.html
 https://docs.djangoproject.com/en/2.0/ref/contrib/gis/install/geolibs/



