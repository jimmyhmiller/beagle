const fs = require('fs');

console.log("=== Node.js Medium File Benchmark ===");

// Create 1KB content
const line = "This is a line of text for testing file I/O performance in Beagle.\n";
const content1k = line.repeat(15);
console.log(`Content size: ${content1k.length} bytes`);

const iterations = 500;

// Benchmark: Write 1KB files
let start1 = process.hrtime.bigint();
for (let i = 0; i < iterations; i++) {
    fs.writeFileSync(`/tmp/node_med_${i}.txt`, content1k);
}
let end1 = process.hrtime.bigint();
let writeTime = Number(end1 - start1) / 1000000;
console.log(`Write ${iterations} x 1KB files: ${writeTime.toFixed(0)} ms`);

// Benchmark: Read 1KB files
let start2 = process.hrtime.bigint();
for (let i = 0; i < iterations; i++) {
    fs.readFileSync(`/tmp/node_med_${i}.txt`, 'utf8');
}
let end2 = process.hrtime.bigint();
let readTime = Number(end2 - start2) / 1000000;
console.log(`Read ${iterations} x 1KB files: ${readTime.toFixed(0)} ms`);

// Cleanup
for (let i = 0; i < iterations; i++) {
    fs.unlinkSync(`/tmp/node_med_${i}.txt`);
}

console.log("=== Complete ===");
console.log(`Total: ${(writeTime + readTime).toFixed(0)} ms`);
