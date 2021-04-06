#!/bin/bash
set -e
source .dockerenv

it=''
ssh_params=''

print_usage() {
  echo "The purpose of this script is to run docker container with the proper parameters. The script has the following arguments:"
  echo "The script has the following arguments: "
  echo "     -c   the command to run inside the container"
  echo "          Example usage: docker_run.sh -c /bin/bash -i"
  echo "                         docker_run.sh -c python script.py -py_param"
  echo "     -p   if given, the container will start an ssh server and map port number 22 from inside to the specified port number outside"
  echo "          Example usage: docker_run.sh -p 2233 -c /bin/bash -i"
  echo "                         After this command if you ssh into the host computer with port number 2233, you find yourself inside the docker"
  echo "                         It is useful for ssh interpreters"
  echo "     -g   Which GPU(s) to use, it will set the CUDA_VISIBLE_DEVICES env variable inside the container"
  echo "          Example usage: docker_run.sh -c /bin/bash -g 2 -i"
  echo "                         docker_run.sh -c python script.py -py_param -g 2"
  echo "     -i   If give, the docker will be run with -it parameter"
  echo "          If you want to attach to the container, specify this option"
  echo "          Example usage: docker_run.sh -p 2233 -c /bin/bash -i"
}

while getopts ihp:g:c: flag
do
    # shellcheck disable=SC2220
    case "${flag}" in
        c) command=${OPTARG};;
        g) gpu=${OPTARG};;
        i) it="-it";;
        p) ssh_params="-v `pwd`/entry_ssh.sh:/entry.sh -p ${OPTARG}:22";;
        h) print_usage
           exit 0 ;;
    esac
done

IMAGE="${IMAGE:-$IMAGE_NAME}"

CONTAINER_ID=$(docker inspect --format="{{.Id}}" ${IMAGE} 2> /dev/null)
if [[ "${CONTAINER_ID}" ]]; then
    docker run --shm-size=2g --rm \
        $it $ssh_params \
        --user $(id -u):$(id -g) \
        -e CUDA_VISIBLE_DEVICES=$gpu \
        -v `pwd`/../:/workspace \
        -v /home/${USER}/cache:/cache \
        -v /home/${USER}/.pycharm_helpers:/home/${USER}/.pycharm_helpers \
        -v /home/${USER}/.cache:/home/${USER}/.cache \
        -v /etc/sudoers:/etc/sudoers:ro \
        -v /etc/passwd:/etc/passwd:ro \
        -v /etc/group:/etc/group:ro \
        -v /etc/shadow:/etc/shadow:ro \
        --workdir=/workspace \
        $IMAGE $command
else
    echo "Unknown container image: ${IMAGE}"
    exit 1
fi
