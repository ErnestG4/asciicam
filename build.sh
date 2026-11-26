#!/bin/sh
# Build asciicam_<version>_all.deb. Needs dpkg-deb (run on the device or any
# Debian system). Usage: ./build.sh [version]
set -eu
cd "$(dirname "$0")"
VERSION=${1:-$(sed -n 's/^VERSION = "\(.*\)"/\1/p' asciicam.py)}
[ -n "$VERSION" ] || { echo "no version given and none in asciicam.py"; exit 1; }

PKG=pkg
rm -rf "$PKG"
mkdir -p "$PKG/DEBIAN" "$PKG/usr/bin" \
    "$PKG/usr/share/APPLaunch/applications" \
    "$PKG/usr/share/APPLaunch/share/images" \
    "$PKG/usr/share/doc/asciicam"

install -m 755 asciicam.py "$PKG/usr/bin/asciicam"
install -m 644 asciicam_100.png "$PKG/usr/share/APPLaunch/share/images/asciicam.png"
install -m 644 debian/asciicam.desktop "$PKG/usr/share/APPLaunch/applications/"
install -m 644 debian/copyright "$PKG/usr/share/doc/asciicam/"
sed "s/@VERSION@/$VERSION/" debian/control.in > "$PKG/DEBIAN/control"

fakeroot dpkg-deb --build --root-owner-group "$PKG" "asciicam_${VERSION}_all.deb"
rm -rf "$PKG"
echo "built asciicam_${VERSION}_all.deb"
