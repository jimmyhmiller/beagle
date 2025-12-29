(ns clojure-comparison
  "Benchmark for comparing Clojure persistent collections with Beagle.
   Run with: clj -M clojure_comparison.clj")

;; ============================================================================
;; Utility functions
;; ============================================================================

(defn elapsed-ms [start end]
  (/ (- end start) 1000000.0))

(defn run-bench [name f]
  (let [start (System/nanoTime)
        result (f)
        end (System/nanoTime)
        ms (elapsed-ms start end)]
    (println (str name ": " (long ms) " ms"))
    result))

;; ============================================================================
;; VECTOR BENCHMARKS
;; ============================================================================

;; 1. Build vector by pushing
(defn bench-vec-push [n]
  (loop [v [] i 0]
    (if (>= i n)
      v
      (recur (conj v i) (inc i)))))

;; 2. Random access reads
(defn bench-vec-get [v iterations]
  (let [cnt (count v)]
    (loop [i 0 acc 0]
      (if (>= i iterations)
        acc
        (let [idx (mod i cnt)
              val (nth v idx)]
          (recur (inc i) (+ acc val)))))))

;; 3. Sum all elements
(defn bench-vec-sum [v]
  (loop [i 0 acc 0]
    (if (>= i (count v))
      acc
      (recur (inc i) (+ acc (nth v i))))))

;; 4. Filter elements (keep evens)
(defn bench-vec-filter [v]
  (loop [i 0 result []]
    (if (>= i (count v))
      result
      (let [val (nth v i)]
        (if (zero? (mod val 2))
          (recur (inc i) (conj result val))
          (recur (inc i) result))))))

;; 5. Map over elements (double each)
(defn bench-vec-map [v]
  (loop [i 0 result []]
    (if (>= i (count v))
      result
      (let [val (nth v i)]
        (recur (inc i) (conj result (* val 2)))))))

;; 6. Concatenate vectors
(defn build-chunks [n chunk-size]
  (loop [result [] i 0]
    (if (>= i n)
      result
      (let [chunk (bench-vec-push chunk-size)]
        (recur (conj result chunk) (inc i))))))

(defn bench-vec-concat [n chunk-size]
  (let [chunks (build-chunks n chunk-size)]
    (loop [result [] i 0]
      (if (>= i (count chunks))
        result
        (let [chunk (nth chunks i)]
          (recur (into result chunk) (inc i)))))))

;; ============================================================================
;; MAP BENCHMARKS
;; ============================================================================

;; 1. Build map with integer keys
(defn bench-map-assoc [n]
  (loop [m {} i 0]
    (if (>= i n)
      m
      (recur (assoc m i (* i 10)) (inc i)))))

;; 2. Random access lookups
(defn bench-map-get [m cnt iterations]
  (loop [i 0 acc 0]
    (if (>= i iterations)
      acc
      (let [key (mod i cnt)
            val (get m key)]
        (if val
          (recur (inc i) (+ acc val))
          (recur (inc i) acc))))))

;; 3. Build map with string keys
(defn bench-map-string-assoc [n]
  (loop [m {} i 0]
    (if (>= i n)
      m
      (let [key (str "key-" i)]
        (recur (assoc m key i) (inc i))))))

;; 4. Update existing keys
(defn bench-map-update [m cnt iterations]
  (loop [m m i 0]
    (if (>= i iterations)
      m
      (let [key (mod i cnt)
            old-val (get m key)
            new-val (if old-val (inc old-val) 0)]
        (recur (assoc m key new-val) (inc i))))))

;; ============================================================================
;; REAL-WORLD WORKLOADS
;; ============================================================================

;; 1. Word frequency counting
(defn build-word-list [n]
  (loop [words [] i 0]
    (if (>= i n)
      words
      (let [word-idx (mod i 100)
            word (str "word" word-idx)]
        (recur (conj words word) (inc i))))))

(defn bench-word-count [n]
  (let [words (build-word-list n)]
    (loop [i 0 counts {}]
      (if (>= i (count words))
        counts
        (let [word (nth words i)
              current (get counts word)
              new-count (if current (inc current) 1)]
          (recur (inc i) (assoc counts word new-count)))))))

;; 2. Group by key
(defn bench-group-by [v]
  (loop [i 0 groups {}]
    (if (>= i (count v))
      groups
      (let [item (nth v i)
            key (mod item 10)
            existing (get groups key)
            new-group (if existing
                        (conj existing item)
                        [item])]
        (recur (inc i) (assoc groups key new-group))))))

;; 3. Nested data structure traversal
(defn build-nested [depth breadth current-depth]
  (if (>= current-depth depth)
    current-depth
    (loop [m {} i 0]
      (if (>= i breadth)
        m
        (let [child (build-nested depth breadth (inc current-depth))
              key (str "child" i)]
          (recur (assoc m key child) (inc i)))))))

(defn bench-nested-build [depth breadth]
  (build-nested depth breadth 0))

;; 5. Merge maps
(defn bench-map-merge [n]
  (let [m1 (bench-map-assoc n)
        half-n (quot n 2)
        m2 (loop [m {} i half-n]
             (if (>= i (+ n half-n))
               m
               (recur (assoc m i (* i 10)) (inc i))))]
    ;; Iterate through m2's keys and merge into m1
    (loop [result m1 ks (keys m2)]
      (if (empty? ks)
        result
        (let [k (first ks)
              v (get m2 k)]
          (recur (assoc result k v) (rest ks)))))))

;; ============================================================================
;; MAIN
;; ============================================================================

(defn warmup []
  (println "Warming up JVM...")
  ;; Run each benchmark multiple times to trigger JIT compilation
  (dotimes [_ 5]
    (let [v (bench-vec-push 10000)]
      (bench-vec-get v 100000)
      (bench-vec-sum v)
      (bench-vec-filter v)
      (bench-vec-map v))
    (bench-vec-concat 10 100)
    (let [m (bench-map-assoc 10000)]
      (bench-map-get m 10000 100000)
      (bench-map-update m 10000 10000))
    (bench-map-string-assoc 5000)
    (bench-word-count 10000)
    (let [v (bench-vec-push 10000)]
      (bench-group-by v))
    (bench-nested-build 4 4)
    (bench-map-merge 5000))
  (println "Warmup complete.")
  (println ""))

(defn -main []
  (println "==============================================")
  (println "Clojure Collections Benchmark")
  (println "(For comparison with Beagle)")
  (println "==============================================")
  (println "")

  (warmup)

  ;; Vector benchmarks
  (println "--- VECTOR OPERATIONS ---")
  (println "")

  (run-bench "Vec push (100k)" #(bench-vec-push 100000))

  (let [test-vec (bench-vec-push 100000)]
    (run-bench "Vec get (1M lookups)" #(bench-vec-get test-vec 1000000))
    (run-bench "Vec sum (100k elements)" #(bench-vec-sum test-vec))
    (run-bench "Vec filter (100k elements)" #(bench-vec-filter test-vec))
    (run-bench "Vec map (100k elements)" #(bench-vec-map test-vec)))

  (run-bench "Vec concat (100 x 1k)" #(bench-vec-concat 100 1000))

  (println "")
  (println "--- MAP OPERATIONS ---")
  (println "")

  (run-bench "Map assoc int keys (100k)" #(bench-map-assoc 100000))

  (let [test-map (bench-map-assoc 100000)]
    (run-bench "Map get (1M lookups)" #(bench-map-get test-map 100000 1000000))
    (run-bench "Map update (100k updates)" #(bench-map-update test-map 100000 100000)))

  (run-bench "Map assoc string keys (50k)" #(bench-map-string-assoc 50000))

  (println "")
  (println "--- REAL-WORLD WORKLOADS ---")
  (println "")

  (run-bench "Word count (100k words)" #(bench-word-count 100000))

  (let [test-vec (bench-vec-push 100000)]
    (run-bench "Group by (100k items)" #(bench-group-by test-vec)))

  (run-bench "Nested structure (depth=6, breadth=10)" #(bench-nested-build 6 10))
  (run-bench "Map merge (50k + 50k)" #(bench-map-merge 50000))

  (println "")
  (println "==============================================")
  (println "Benchmark complete")
  (println "=============================================="))

(-main)
