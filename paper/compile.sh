#!/bin/bash
# compile the pdf with the openjournals docker image. Use enroot if installed.
IMAGE=openjournals/inara
IMAGE_BASENAME=$(dirname $IMAGE)
if [[ -x "/usr/bin/enroot" ]] ; then
    echo "using enroot container runtime..."
    if ! enroot list | grep $IMAGE_BASENAME > /dev/null ; then
        echo "retrieving container image..."
        enroot import -o $IMAGE_BASENAME.sqsh docker://$IMAGE
        enroot create $IMAGE_BASENAME.sqsh
        rm $IMAGE_BASENAME.sqsh
    fi
    export ENROOT_MOUNT_HOME=y
    enroot start --mount $PWD:/data --env JOURNAL=joss $(dirname $IMAGE)
else
    echo "using docker container runtime..."
    docker run --rm \
        --volume $PWD:/data \
        --user $(id -u):$(id -g) \
        --env JOURNAL=joss \
        $IMAGE
fi
