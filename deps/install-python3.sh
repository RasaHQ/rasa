#!/bin/bash

#/ Usage: install-python3.sh [<num_jobs = 4>] [<version = 3.6.8>]
#/ Setup Python3 installation on a Linux system.
#/
#/ Example:
#/ ./install-python3.sh 8 3.6.8 true
usage() {
    grep '^#/' "$0" | cut -c4-
    exit 0
}
expr "$*" : ".*--help" > /dev/null && usage

# Parse arguments
jobs=4
version="3.6.8"
non_invasive_install=true
[ $# -ge 1 ] && jobs="$1"
[ $# -ge 2 ] && version="$2"

# Test if this version of Python exists
run_version="$(echo "$version" | cut -d. -f1).$(echo "$version" | cut -d. -f2)"
if which "python${run_version}"; then
	echo "Python-${run_version} already installed. Nothing to do."
	exit 0
fi

# Login as a root
(( EUID != 0 )) && echo "sudo required! - run with sudoer privileges: 'sudo $0'" && exit 1

# Download and install
wget "https://www.python.org/ftp/python/${version}/Python-${version}.tgz"
tar -xvf "Python-${version}.tgz"
cd "Python-${version}"
./configure --enable-optimizations
make -j "${jobs}"
make install

rm -rf "Python-${version}"

