#!/bin/bash

# This bash script has been tested on macOS 11.6 and Windows Server 2019 Datacenter
# It will rename all the files in the current directory
# to follow I0.dcm -> In.dcm and move them into a
# target directory e.g.
#
# macOS: ctcopy.sh /path/to/ai-ct-scans/extra_data/data/1/1/Abdo1
# Windows: sh ctcopy.sh /path/to/ai-ct-scans/extra_data/data/1/1/Abdo1

# check correct usage
if [ -z $1 ]; then
	echo "Usage: $0 [path-to-destination-folder]"
	exit 1
fi

# check destination directory exists
if [ ! -d $1 ]; then
	echo "$1 does not exist. Please create this folder first"
	exit 1
fi

# confirm destination directory contents will be removed
read -p "Replacing contents of $1 - are you sure? [y,N] " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
	# remove existing files
	rm -f $1/*.dcm
	# rename all files in current folder with I0->In
	index=0
	for n in $(ls); do
	       cp $n $1/I$index.dcm
	       index=$((index + 1))
	done
else
	echo "Aborted"
	exit 1
fi
