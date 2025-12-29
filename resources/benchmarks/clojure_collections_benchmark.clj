(ns collections-benchmark
  "Benchmark for Clojure persistent collections.
   Run with: clj -M resources/benchmarks/clojure_collections_benchmark.clj
   Or: clojure -M resources/benchmarks/clojure_collections_benchmark.clj")

;; ============================================================================
;; VECTOR BENCHMARKS
;; ============================================================================

;; --- Push benchmark ---
(defn bench-vec-push [n]
  (let [start (System/nanoTime)]
    (loop [v [] i 0]
      (if (>= i n)
        (let [end (System/nanoTime)]
          (/ (- end start) 1000000.0))
        (recur (conj v i) (inc i))))))

;; --- Get benchmark (random access) ---
(defn bench-vec-get [v iterations]
  (let [cnt (count v)
        start (System/nanoTime)]
    (loop [i 0]
      (if (>= i iterations)
        (let [end (System/nanoTime)]
          (/ (- end start) 1000000.0))
        (do
          (nth v (mod i cnt))
          (recur (inc i)))))))

;; --- Assoc benchmark (update) ---
(defn bench-vec-assoc [v iterations]
  (let [cnt (count v)
        start (System/nanoTime)]
    (loop [v v i 0]
      (if (>= i iterations)
        (let [end (System/nanoTime)]
          (/ (- end start) 1000000.0))
        (let [idx (mod i cnt)]
          (recur (assoc v idx (* i 2)) (inc i)))))))

;; ============================================================================
;; MAP BENCHMARKS
;; ============================================================================

;; --- Assoc benchmark (insert with integer keys) ---
(defn bench-map-assoc [n]
  (let [start (System/nanoTime)]
    (loop [m {} i 0]
      (if (>= i n)
        (let [end (System/nanoTime)]
          (/ (- end start) 1000000.0))
        (recur (assoc m i (* i 10)) (inc i))))))

;; --- Get benchmark (lookup) ---
(defn bench-map-get [m cnt iterations]
  (let [start (System/nanoTime)]
    (loop [i 0]
      (if (>= i iterations)
        (let [end (System/nanoTime)]
          (/ (- end start) 1000000.0))
        (do
          (get m (mod i cnt))
          (recur (inc i)))))))

;; --- String key benchmark ---
(defn bench-map-string-assoc [n]
  (let [start (System/nanoTime)]
    (loop [m {} i 0]
      (if (>= i n)
        (let [end (System/nanoTime)]
          (/ (- end start) 1000000.0))
        (let [k (str "key" i)]
          (recur (assoc m k i) (inc i)))))))

;; ============================================================================
;; MAIN - Run all benchmarks
;; ============================================================================

(defn print-result [name ms]
  (println name)
  (println (str "  Time: " (format "%.2f" ms) " ms"))
  (println))

(defn -main [& args]
  (println "==============================================")
  (println "Clojure Collections Benchmark")
  (println "==============================================")
  (println)

  ;; Warm up JVM
  (println "Warming up JVM...")
  (dotimes [_ 3]
    (bench-vec-push 1000)
    (bench-map-assoc 1000))
  (println "Warmup complete.")
  (println)

  ;; Vector benchmarks
  (let [vec-size 100000
        get-iterations 1000000
        assoc-iterations 100000]

    (println (str "--- Vector Push (" vec-size " elements) ---"))
    (print-result "Vector Push" (bench-vec-push vec-size))

    ;; Create vector for get/assoc benchmarks
    (let [v (vec (range vec-size))]
      (println (str "--- Vector Get (" get-iterations " lookups) ---"))
      (print-result "Vector Get" (bench-vec-get v get-iterations))

      (println (str "--- Vector Assoc (" assoc-iterations " updates) ---"))
      (print-result "Vector Assoc" (bench-vec-assoc v assoc-iterations))))

  ;; Map benchmarks
  (let [map-size 100000
        map-get-iterations 1000000]

    (println (str "--- Map Assoc/Integer Keys (" map-size " entries) ---"))
    (print-result "Map Assoc (int keys)" (bench-map-assoc map-size))

    ;; Create map for get benchmarks
    (let [m (into {} (map (fn [i] [i (* i 10)]) (range map-size)))]
      (println (str "--- Map Get (" map-get-iterations " lookups) ---"))
      (print-result "Map Get" (bench-map-get m map-size map-get-iterations)))

    (let [string-map-size 50000]
      (println (str "--- Map Assoc/String Keys (" string-map-size " entries) ---"))
      (print-result "Map Assoc (string keys)" (bench-map-string-assoc string-map-size))))

  (println "==============================================")
  (println "Benchmark complete")
  (println "=============================================="))

;; Run main when script is executed
(-main)
