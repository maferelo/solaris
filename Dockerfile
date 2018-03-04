FROM python:3.5
ENV PYTHONUNBUFFERED 1

RUN \
  apt-get -y update && \
  apt-get install -y gettext && \
  apt-get clean

RUN apt-get --assume-yes install binutils libproj-dev gdal-bin

RUN wget http://download.osgeo.org/gdal/1.11.2/gdal-1.11.2.tar.gz
RUN tar -xzf gdal-1.11.2.tar.gz
RUN cd gdal-1.11.2; ./configure --with-python; make; make install

ADD requirements.txt /app/
RUN pip install -r /app/requirements.txt

ADD . /app
WORKDIR /app

EXPOSE 8000
ENV PORT 8000

CMD ["uwsgi", "/app/saleor/wsgi/uwsgi.ini"]
