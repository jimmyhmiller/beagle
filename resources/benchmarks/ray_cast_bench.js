// Node.js port of ray_cast_bench.bg.
//
// Mirrors the Beagle version line-for-line: same wall geometry,
// same number of rays, same number of frames, same checksum
// arithmetic. Use it to gauge how far the Beagle JIT is from V8
// on the hottest function in beagle-zelda.
//
// Run: node resources/benchmarks/ray_cast_bench.js

"use strict";

const SHADOW_FAR = 5000.0;
const PI = 3.14159265358979;

function makeV(x, y) {
    return { x, y };
}

function makeSeg(a, b) {
    return { a, b };
}

function absF(x) {
    return x < 0.0 ? -x : x;
}

function raySegmentHit(origin, dir, a, b) {
    const sx = b.x - a.x;
    const sy = b.y - a.y;
    const denom = dir.x * sy - dir.y * sx;
    if (absF(denom) < 0.000001) {
        return -1.0;
    }
    const diffx = a.x - origin.x;
    const diffy = a.y - origin.y;
    const t = (diffx * sy - diffy * sx) / denom;
    const u = (diffx * dir.y - diffy * dir.x) / denom;
    if (t >= 0.0 && u >= 0.0 && u <= 1.0) return t;
    return -1.0;
}

function castRay(origin, dir, segments) {
    let best = SHADOW_FAR;
    for (let i = 0; i < segments.length; i++) {
        const s = segments[i];
        const t = raySegmentHit(origin, dir, s.a, s.b);
        if (t > 0.0 && t < best) best = t;
    }
    return best;
}

function buildSegments() {
    const walls = [
        [100.0, 100.0, 60.0, 40.0],
        [-150.0, 80.0, 50.0, 50.0],
        [200.0, -120.0, 40.0, 60.0],
        [-80.0, -200.0, 70.0, 30.0],
        [300.0, 50.0, 50.0, 100.0],
        [-250.0, -50.0, 80.0, 40.0],
        [50.0, 250.0, 100.0, 30.0],
        [0.0, -300.0, 90.0, 50.0],
    ];
    const segs = [];
    for (let i = 0; i < walls.length; i++) {
        const w = walls[i];
        const cx = w[0], cy = w[1], hx = w[2], hy = w[3];
        const tl = makeV(cx - hx, cy - hy);
        const tr = makeV(cx + hx, cy - hy);
        const br = makeV(cx + hx, cy + hy);
        const bl = makeV(cx - hx, cy + hy);
        segs.push(makeSeg(tl, tr));
        segs.push(makeSeg(tr, br));
        segs.push(makeSeg(br, bl));
        segs.push(makeSeg(bl, tl));
    }
    return segs;
}

function buildAngles(n) {
    const twoPi = 2.0 * PI;
    const angles = [];
    for (let i = 0; i < n; i++) {
        angles.push(-PI + (i / n) * twoPi);
    }
    return angles;
}

function doPass(origin, angles, segments) {
    let sum = 0.0;
    for (let i = 0; i < angles.length; i++) {
        const a = angles[i];
        const dir = makeV(Math.cos(a), Math.sin(a));
        sum += castRay(origin, dir, segments);
    }
    return sum;
}

function runBench(frames, origin, angles, segments) {
    let total = 0.0;
    for (let f = 0; f < frames; f++) {
        total += doPass(origin, angles, segments);
    }
    return total;
}

function main() {
    const segments = buildSegments();
    const angles = buildAngles(64);
    const origin = makeV(0.0, 0.0);

    // Warm-up: lets V8's tiering compiler reach the optimized tier.
    const warm = runBench(200, origin, angles, segments);
    console.log("warm checksum: " + warm);

    const frames = 10000;
    const start = process.hrtime.bigint();
    const total = runBench(frames, origin, angles, segments);
    const end = process.hrtime.bigint();

    const elapsedMs = Number(end - start) / 1e6;
    const calls = frames * angles.length * segments.length;

    console.log("frames: " + frames);
    console.log("segments/frame: " + segments.length);
    console.log("rays/frame: " + angles.length);
    console.log("ray_segment_hit calls: " + calls);
    console.log("checksum: " + total);
    console.log("elapsed_ms: " + elapsedMs.toFixed(2));
}

main();
