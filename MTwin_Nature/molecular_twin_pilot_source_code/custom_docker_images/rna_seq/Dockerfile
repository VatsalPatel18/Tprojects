FROM 965350412536.dkr.ecr.us-west-2.amazonaws.com/betteromics-python-slim:latest
RUN wget https://github.com/pachterlab/kallisto/releases/download/v0.46.1/kallisto_linux-v0.46.1.tar.gz 
RUN tar xvzf kallisto_linux-v0.46.1.tar.gz; 
RUN cp kallisto/kallisto /usr/local/bin
RUN apt-get update
RUN apt-get install -y bamtools time
RUN wget https://github.com/shenwei356/csvtk/releases/download/v0.20.0/csvtk_linux_amd64.tar.gz
RUn tar -xvf csvtk_linux_amd64.tar.gz
RUN mv csvtk /usr/local/bin/
