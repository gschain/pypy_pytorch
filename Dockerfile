FROM pypy:3.6-slim
COPY . /app
ENV PYTHONPATH "/app:${PYTHONPATH}"
WORKDIR /app
RUN pip3 install Cython
RUN pip3 install tools
RUN pip3 install numpy
RUN pypy3 -m pip install -r requirements.txt
#RUN pip3 install -r requirements.txt
EXPOSE 5000

# Define environment variable
ENV MODEL_NAME MyModel
ENV API_TYPE REST
ENV SERVICE_TYPE MODEL
ENV PERSISTENCE 0

RUN sed -i '3ifrom DeepFM import DeepFM' /usr/local/bin/seldon-core-microservice

CMD ["/bin/bash"]
#CMD pypy3 /usr/local/bin/seldon-core-microservice $MODEL_NAME $API_TYPE --service-type $SERVICE_TYPE --persistence $PERSISTENCE
