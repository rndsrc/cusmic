#!/bin/sh
# Collect runtime libraries; the host supplies the NVIDIA driver.
set -eu
root=$1
shift
mkdir -p "$root/etc" "$root/tmp" "$root/usr/share/licenses/cusmic-runtime" \
    "$root/usr/lib" "$root/usr/lib64" "$root/usr/sbin"
# Use Debian's merged paths in both scratch and Python runtime images.
ln -s usr/lib "$root/lib"
ln -s usr/lib64 "$root/lib64"
ln -s usr/sbin "$root/sbin"
triplet=$(gcc -dumpmachine)
ldd "$@" > /tmp/dependencies
if grep -q 'not found' /tmp/dependencies; then
    cat /tmp/dependencies >&2
    exit 1
fi
for lib in $(awk '$2 == "=>" {print $3} $1 ~ /^\// {print $1}' /tmp/dependencies) \
    /usr/lib/$triplet/libdl.so.2 /usr/lib/$triplet/libpthread.so.0 \
    /usr/lib/$triplet/librt.so.1 /usr/lib/$triplet/libz.so.1; do
    case "$lib" in */libcusmic.so) continue ;; esac
    cp -L --parents "$lib" "$root"
    package=$(dpkg-query -S "$(readlink -f "$lib")" | head -1 | cut -d: -f1)
    cp -L "/usr/share/doc/$package/copyright" "$root/usr/share/licenses/cusmic-runtime/$package"
done
cp /sbin/ldconfig "$root/sbin/"
printf '%s\n' /usr/local/lib /usr/local/nvidia/lib /usr/local/nvidia/lib64 \
    /lib/$triplet /usr/lib/$triplet > "$root/etc/ld.so.conf"
chmod 1777 "$root/tmp"
cp -L /usr/share/doc/libc-bin/copyright "$root/usr/share/licenses/cusmic-runtime/libc-bin"
cp -a --parents /usr/share/common-licenses "$root"
