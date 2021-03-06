#!/bin/bash

RUN_SUDO=sudo

if [ ! -d "/usr/include/" ]; then
    echo "Not develop environment, no action for setup"
    exit 1
fi

if [ "$(expr substr $(uname -s) 1 5)" != "Linux" ]; then
    unset RUN_SUDO
else
    user=$(env | grep USER | cut -d "=" -f 2)
    if [ "$user" == "root"  ]; then
        unset RUN_SUDO
    fi
fi

if [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    if [ ! -f ${PWD}/faiss/libfaiss.a ]; then
        echo libfaiss.a not exists, build faiss first !
        exit 1
    fi
    eval $RUN_SUDO cp -f ${PWD}/faiss/libfaiss.a /usr/local/lib/libfaiss.a
    eval $RUN_SUDO ldconfig
fi

if [ -L "/usr/include/faiss" ]; then
    eval $RUN_SUDO rm -r /usr/include/faiss
fi
eval $RUN_SUDO ln -s ${PWD}/faiss /usr/include/faiss
