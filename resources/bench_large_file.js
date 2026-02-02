const fs = require('fs');

console.log("=== Node.js Large File Benchmark ===");

// Create 100KB content
console.log("Creating 100KB content...");
const line = "This is a line of text that will be repeated many times to create a large file for testing.\n";
const linesNeeded = Math.floor((100 * 1024) / line.length);
let content = "";
for (let i = 0; i < linesNeeded; i++) {
    content += line;
}
console.log(`Content size: ${content.length} bytes`);

const iterations = 100;

// Benchmark: Write large files
let start1 = process.hrtime.bigint();
for (let i = 0; i < iterations; i++) {
    fs.writeFileSync(`/tmp/node_large_${i}.txt`, content);
}
let end1 = process.hrtime.bigint();
let writeTime = Number(end1 - start1) / 1000000;
console.log(`Write ${iterations} x 100KB files: ${writeTime.toFixed(0)} ms`);

// Benchmark: Read large files
let start2 = process.hrtime.bigint();
for (let i = 0; i < iterations; i++) {
    fs.readFileSync(`/tmp/node_large_${i}.txt`, 'utf8');
}
let end2 = process.hrtime.bigint();
let readTime = Number(end2 - start2) / 1000000;
console.log(`Read ${iterations} x 100KB files: ${readTime.toFixed(0)} ms`);

// Cleanup
for (let i = 0; i < iterations; i++) {
    fs.unlinkSync(`/tmp/node_large_${i}.txt`);
}

console.log("=== Complete ===");
console.log(`Total: ${(writeTime + readTime).toFixed(0)} ms`);
