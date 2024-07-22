#!/system/bin/sh

SCRIPT=`readlink -f "$0"`
DIR=`dirname "$SCRIPT"`
export LD_LIBRARY_PATH="$DIR/lib"
cd "$DIR"
chmod +x hexagon
./hexagon