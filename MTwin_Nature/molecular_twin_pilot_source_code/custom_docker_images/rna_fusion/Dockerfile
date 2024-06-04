FROM 965350412536.dkr.ecr.us-west-2.amazonaws.com/betteromics-python-slim:latest
RUN wget https://dl.google.com/go/go1.11.5.linux-amd64.tar.gz;   tar -xvzf go1.11.5.linux-amd64.tar.gz
ENV GOROOT=/usr/src/app/go
ENV PATH="/usr/src/app/go/bin:${PATH}"
RUN mkdir /root/workdir
RUN cd /root/workdir/; go mod init playground/; GOBIN=/usr/local/bin/ go install github.com/grailbio/bio/cmd/bio-fusion
RUN apt-get update
RUN apt-get install -y bamtools
