/* The Computer Language Benchmarks Game
   https://salsa.debian.org/benchmarksgame-team/benchmarksgame/

   Contributed by Jesse Millikan
   Modified by Matt Baker
   Single-threaded version
*/

'use strict';

var rd = require('readline');

function RefNum(num){ this.num = num; }
RefNum.prototype.toString = function() { return this.num.toString(); }

function frequency(seq, length){
    var freq = new Map(), n = seq.length-length+1, key, cur, i = 0;
    for(; i<n; ++i){
        key = seq.substr(i, length);
        cur = freq.get(key);
        cur === undefined ? freq.set(key, new RefNum(1)) : ++cur.num;
    }
    return freq;
}

function sort(seq, length){
    var f = frequency(seq, length), keys = Array.from(f.keys()),
        n = seq.length-length+1, res = '';
    keys.sort((a, b)=>f.get(b)-f.get(a));
    for (var key of keys) res +=
        key.toUpperCase()+' '+(f.get(key)*100/n).toFixed(3)+'\n';
    res += '\n';
    return res;
}

function find(seq, s){
    var f = frequency(seq, s.length);
    return (f.get(s) || 0)+"\t"+s.toUpperCase()+'\n';
}

function main() {
    var seq = '', reading = false;
    var lineHandler = function(line){
        if (reading) {
            if (line[0]!=='>') seq += line;
        } else reading = line.substr(0, 6)==='>THREE';
    };
    rd.createInterface(process.stdin, process.stdout)
        .on('line', lineHandler).on('close', function() {
            var res = '';
            res += sort(seq, 1);
            res += sort(seq, 2);
            res += find(seq, "ggt");
            res += find(seq, "ggta");
            res += find(seq, "ggtatt");
            res += find(seq, "ggtattttaatt");
            res += find(seq, "ggtattttaatttatagt");
            process.stdout.write(res);
        });
}

main();
