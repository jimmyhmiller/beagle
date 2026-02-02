const fs = require('fs');
const path = require('path');

// Benchmark file I/O operations - synchronous version for fair comparison

console.log("=== Node.js File I/O Benchmark ===");

const iterations = 1000;
const benchDir = '/tmp/node_bench';

// Setup: create test directory
if (!fs.existsSync(benchDir)) {
    fs.mkdirSync(benchDir, { recursive: true });
}

// Benchmark 1: Write small files
let start1 = process.hrtime.bigint();
for (let i = 0; i < iterations; i++) {
    fs.writeFileSync(path.join(benchDir, `test${i}.txt`), "Hello, World!\n");
}
let end1 = process.hrtime.bigint();
let writeTime = Number(end1 - start1) / 1000000;
console.log(`Write ${iterations} small files: ${writeTime.toFixed(0)} ms`);

// Benchmark 2: Read small files
let start2 = process.hrtime.bigint();
for (let i = 0; i < iterations; i++) {
    fs.readFileSync(path.join(benchDir, `test${i}.txt`), 'utf8');
}
let end2 = process.hrtime.bigint();
let readTime = Number(end2 - start2) / 1000000;
console.log(`Read ${iterations} small files: ${readTime.toFixed(0)} ms`);

// Benchmark 3: File exists checks
let start3 = process.hrtime.bigint();
for (let i = 0; i < iterations; i++) {
    fs.existsSync(path.join(benchDir, `test${i}.txt`));
}
let end3 = process.hrtime.bigint();
let existsTime = Number(end3 - start3) / 1000000;
console.log(`Check exists ${iterations} files: ${existsTime.toFixed(0)} ms`);

// Benchmark 4: File size queries
let start4 = process.hrtime.bigint();
for (let i = 0; i < iterations; i++) {
    fs.statSync(path.join(benchDir, `test${i}.txt`)).size;
}
let end4 = process.hrtime.bigint();
let sizeTime = Number(end4 - start4) / 1000000;
console.log(`Get size ${iterations} files: ${sizeTime.toFixed(0)} ms`);

// Benchmark 5: Delete files
let start5 = process.hrtime.bigint();
for (let i = 0; i < iterations; i++) {
    fs.unlinkSync(path.join(benchDir, `test${i}.txt`));
}
let end5 = process.hrtime.bigint();
let deleteTime = Number(end5 - start5) / 1000000;
console.log(`Delete ${iterations} files: ${deleteTime.toFixed(0)} ms`);

// Cleanup
fs.rmdirSync(benchDir);

console.log("=== Benchmark Complete ===");
let total = writeTime + readTime + existsTime + sizeTime + deleteTime;
console.log(`Total time: ${total.toFixed(0)} ms`);
