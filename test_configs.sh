#!/bin/bash
set -e

# Build
echo "Building Rust..."
cd rust_gf && cargo build --release && cd ..
echo "Building Haskell..."
cd haskell_gf && cabal build && cd ..

# Setup Paths
ROOT_DIR=$(pwd)
PERL_BIN="$ROOT_DIR/bin/gf"
RUST_BIN="$ROOT_DIR/rust_gf/target/release/rust_gf"
HASKELL_BIN=$(cd haskell_gf && cabal list-bin haskell-gf)

# Setup Test Corpus
rm -rf config_test_env
mkdir -p config_test_env/home
mkdir -p config_test_env/project

# Create dummy project files
echo "match this" > config_test_env/project/file1.txt
echo "match this" > config_test_env/project/file2.log
echo "ignore this" > config_test_env/project/ignored.txt

# Create a test .gfconf
# Directives supported: target <pattern>, ignore <pattern>
cat <<EOF > config_test_env/home/.gfconf
ignore ignored.txt
target .txt
EOF

echo "Testing configuration parity..."
cd config_test_env/project

# Run tests with fake HOME
export HOME="$ROOT_DIR/config_test_env/home"

echo "--- PERL ---"
$PERL_BIN --search "match"
echo "--- RUST ---"
$RUST_BIN --search "match"
echo "--- HASKELL ---"
$HASKELL_BIN --search "match"

echo ""
echo "Verification complete. All versions should have only found matches in 'file1.txt' (because .log is not in target and ignored.txt is ignored)."
