#!/bin/bash
# Generates compile_flags.txt for clangd
# Works on Linux, macOS, MSYS2, WSL
# Windows MSVC users: edit compile_flags.txt manually

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "-I."
echo "-I./tests/bench"
echo "-Wall"
echo "-Wextra"

# Eigen
pkg-config --cflags eigen3 2>/dev/null | tr ' ' '\n' | grep -v '^$' || true

# OpenBLAS: prefer local single-threaded build, then pkg-config, then Homebrew
if [ -d "$SCRIPT_DIR/deps/openblas/include" ]; then
  echo "-I$SCRIPT_DIR/deps/openblas/include"
elif pkg-config --exists openblas 2>/dev/null; then
  pkg-config --cflags openblas | tr ' ' '\n' | grep -v '^$'
elif [ -d "/opt/homebrew/opt/openblas/include" ]; then
  echo "-I/opt/homebrew/opt/openblas/include"
elif [ -d "/usr/local/opt/openblas/include" ]; then
  echo "-I/usr/local/opt/openblas/include"
fi
