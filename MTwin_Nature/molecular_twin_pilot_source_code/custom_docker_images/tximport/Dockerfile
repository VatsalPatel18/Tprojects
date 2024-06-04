FROM r-base:3.6.3

RUN R -e "install.packages('readr', dependencies=TRUE, repos='https://ftp.osuosl.org/pub/cran/')"
RUN R -e "install.packages('dplyr', dependencies=TRUE, repos='https://ftp.osuosl.org/pub/cran/')"
RUN R -e "install.packages('reshape2',  dependencies=TRUE, repos='https://ftp.osuosl.org/pub/cran/')"
# will install Bioconductor 3.10, BiocManager 1.30.10
RUN R -e "install.packages('BiocManager',  dependencies=TRUE, repos='https://ftp.osuosl.org/pub/cran/')"
# will install tximport 1.14.2
RUN R -e "BiocManager::install('tximport')"

# install python + argparse, json for R::argparse package
RUN apt-get update -y && apt-get install -y python3.6 pip
RUN pip install argparse

RUN R -e "install.packages('argparse',  dependencies=TRUE, repos='https://ftp.osuosl.org/pub/cran/')"

CMD ["/bin/bash"]