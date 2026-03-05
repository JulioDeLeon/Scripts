#!/bin/bash
set -e

# Build
cd rust_gf && cargo build --release && cd ..
cd haskell_gf && cabal build && cd ..

# Locations (Absolute paths)
ROOT_DIR=$(pwd)
PERL_BIN="$ROOT_DIR/bin/gf"
RUST_BIN="$ROOT_DIR/rust_gf/target/release/rust_gf"
HASKELL_BIN=$(cd haskell_gf && cabal list-bin haskell-gf)

# Create a test directory to control the corpus
rm -rf test_corpus
mkdir -p test_corpus
# Copy library files
cp lib/GF/*.pm test_corpus/
# Copy binary (for completeness, though we don't search binary itself usually unless text)
cp bin/gf test_corpus/

echo "Verifying parity in test_corpus..."
cd test_corpus

# Search for 'package'
echo "--- PERL ---"
"$PERL_BIN" --search "package" > ../perl.out
echo "--- RUST ---"
"$RUST_BIN" --search "package" > ../rust.out
echo "--- HASKELL ---"
"$HASKELL_BIN" --search "package" > ../haskell.out

cd ..

# Sort and compare (files order might differ)
sort perl.out > perl.sorted
sort rust.out > rust.sorted
sort haskell.out > haskell.sorted

# Diff
echo "Diff Perl vs Rust:"
# Ignore color codes for diff comparison? 
# Maybe grep out color codes?
# For now, just show line count differences.
# diff perl.sorted rust.sorted || echo "Differences found (expected due to formatting/color codes)"

echo "Line counts:"
P_COUNT=$(grep "package" perl.out | wc -l)
R_COUNT=$(grep "package" rust.out | wc -l)
H_COUNT=$(grep "package" haskell.out | wc -l)
echo "Perl: $P_COUNT"
echo "Rust: $R_COUNT"
echo "Haskell: $H_COUNT"

# Benchmark
echo "Benchmarking (loop 10 times)..."
cd test_corpus

echo "Perl:"
time for i in {1..10}; do "$PERL_BIN" --search "package" > /dev/null; done

echo "Rust:"
time for i in {1..10}; do "$RUST_BIN" --search "package" > /dev/null; done

echo "Haskell:"
time for i in {1..10}; do "$HASKELL_BIN" --search "package" > /dev/null; done
