# Beagle Standard Library Reference

Auto-generated catalog. Import with `use beagle.<name> as <alias>`.

**50 modules, 1193+ public functions.**
## beagle.ansi  (51 fns)
`wrap`, `style`, `bold`, `dim`, `italic`, `underline`, `blink`, `reverse`, `hidden`, `strikethrough`, `black`, `red`, `green`, `yellow`, `blue`, `magenta`, `cyan`, `white`, `bright-black`, `bright-red`, `bright-green`, `bright-yellow`, `bright-blue`, `bright-magenta`, `bright-cyan`, `bright-white`, `gray`, `grey`, `bg-black`, `bg-red`, `bg-green`, `bg-yellow`, `bg-blue`, `bg-magenta`, `bg-cyan`, `bg-white`, `bg-bright-black`, `bg-bright-red`, `bg-bright-green`, `bg-bright-yellow`, `bg-bright-blue`, `bg-bright-magenta`, `bg-bright-cyan`, `bg-bright-white`, `fg-256`, `bg-256`, `rgb`, `bg-rgb`, `combine`, `sgr-param-code?`, `strip-ansi`

## beagle.async  (148 fns)
`make-future`, `future-state`, `resolve-future!`, `reject-future!`, `cancel-future!`, `future-cancelled?`, `make-cancellation-token`, `cancel!`, `cancelled?`, `make-scope`, `scope-add-child!`, `scope-children`, `scope-cancel-all!`, `scope-await-all-children`, `with-scope`, `async-ok`, `async-err`, `async-ok?`, `async-unwrap`, `async-unwrap-or`, `handle-read-file`, `handle-write-file`, `handle-append-file`, `handle-delete-file`, `handle-file-exists`, `handle-file-size`, `handle-is-file`, `handle-is-directory`, `handle-rename-file`, `handle-copy-file`, `handle-read-dir`, `handle-create-dir`, `handle-create-dir-all`, `handle-remove-dir`, `handle-remove-dir-all`, `handle-open`, `handle-close`, `handle-read`, `handle-write`, `handle-read-line`, `handle-flush`, `handle-sleep`, `handle-await-blocking`, `handle-await-all-blocking`, `handle-await-first-blocking`, `handle-cancel`, `handle-spawn-blocking`, `handle-spawn-with-token-blocking`, `handle-spawn-threaded`, `handle-spawn-with-token-threaded`, `handle-await-threaded`, `poll-until-resolved`, `handle-await-all-threaded`, `handle-await-first-threaded`, `handle-sleep-event-loop`, `poll-file-result`, `wait-for-file-result`, `handle-read-file-async`, `handle-write-file-async`, `handle-delete-file-async`, `handle-file-size-async`, `handle-read-dir-async`, `handle-append-file-async`, `handle-file-exists-async`, `handle-rename-file-async`, `handle-copy-file-async`, `handle-create-dir-async`, `handle-create-dir-all-async`, `handle-remove-dir-async`, `handle-remove-dir-all-async`, `handle-is-directory-async`, `handle-is-file-async`, `handle-open-async`, `handle-close-async`, `handle-handle-read-async`, `handle-handle-write-async`, `handle-handle-readline-async`, `handle-handle-flush-async`, `read-file`, `write-file`

## beagle.bail  (11 fns)
`add`, `sub`, `mul`, `div`, `modulo`, `lt`, `lte`, `gt`, `gte`, `eq`, `ne`

## beagle.base64  (10 fns)
`to-bytes`, `encode-with`, `decode-char-value`, `decode-any`, `encode`, `decode`, `encode-url`, `decode-url`, `bytes-to-string`, `main`

## beagle.bigint  (20 fns)
`trim-limbs`, `make`, `zero`, `from-int`, `from-string`, `is-zero?`, `cmp-mag`, `add-mag`, `sub-mag`, `mul-mag`, `compare`, `negate`, `add`, `sub`, `mul`, `append-padded-limb!`, `to-string`, `check`, `to-string-any`, `main`

## beagle.channel  (19 fns)
`make-channel`, `send!`, `send-loop`, `try-receive`, `try-receive-loop`, `receive`, `receive-spin`, `spin-backoff`, `channel-count`, `channel-empty?`, `make-mutex`, `acquire!`, `acquire-spin`, `release!`, `with-lock`, `make-counter`, `incr!`, `add!`, `counter-value`

## beagle.cli  (28 fns)
`make-spec`, `add-flag`, `add-option`, `add-positional`, `find-flag-by-name`, `find-option-by-name`, `find-flag-by-short`, `find-option-by-short`, `seed-defaults`, `st-new`, `st-set-option`, `st-add-positional`, `st-add-error`, `parse`, `is-long-arg`, `is-short-arg`, `parse-long`, `apply-long-option`, `parse-short`, `parse-short-cluster`, `finalize`, `check-required-options`, `check-required-positionals`, `help-text`, `build-usage-line`, `build-positionals-section`, `build-options-section`, `build-flags-section`

## beagle.containers  (28 fns)
`deque-new`, `deque-count`, `deque-empty?`, `coll-reverse`, `coll-subvec`, `push-front`, `push-back`, `deque-rebalance`, `pop-front`, `pop-back`, `deque-to-vec`, `dm-new`, `dm-get`, `dm-assoc`, `dm-update`, `dm-count`, `om-new`, `om-get`, `om-has-key?`, `om-assoc`, `om-keys-in-order`, `om-pairs`, `om-vals-in-order`, `om-count`, `vec-member?`, `set-union-vec`, `set-intersection-vec`, `set-difference-vec`

## beagle.csv  (7 fns)
`parse`, `parse-with-header`, `needs-quoting?`, `write-field!`, `write-row!`, `write`, `write-with-header`

## beagle.date  (18 fns)
`from-epoch`, `now`, `to-epoch`, `is-leap-year?`, `weekday`, `add-seconds`, `add-days`, `diff-seconds`, `pad-num`, `format-iso8601`, `days-in-month`, `parse-digits`, `expect-sep`, `parse-iso8601`, `dt-equal?`, `check`, `check-true`, `main`

## beagle.effect  (1 fns)
`resume-tail`

## beagle.ffi  (11 fns)
`call-variadic`, `size-of`, `read-at`, `write-at`, `cell`, `cell-get`, `cell-set`, `array`, `array-get`, `array-set`, `forget`

## beagle.fs  (58 fns)
`read-file`, `write-file`, `append-file`, `delete-file`, `exists?`, `file-size`, `is-file?`, `rename`, `copy`, `read-dir`, `create-dir`, `create-dir-all`, `remove-dir`, `remove-dir-all`, `is-directory?`, `open`, `close`, `read`, `write`, `read-line`, `flush`, `handle-read-file`, `handle-write-file`, `handle-append-file`, `handle-delete-file`, `handle-file-exists`, `handle-file-size`, `handle-is-file`, `handle-is-directory`, `handle-rename-file`, `handle-copy-file`, `handle-read-dir`, `handle-create-dir`, `handle-create-dir-all`, `handle-remove-dir`, `handle-remove-dir-all`, `handle-open`, `handle-close`, `handle-read`, `handle-write`, `handle-read-line`, `handle-flush`, `run-blocking`, `blocking-read-file`, `blocking-write-file`, `blocking-append-file`, `blocking-delete-file`, `blocking-exists?`, `blocking-file-size`, `blocking-is-file?`, `blocking-is-directory?`, `blocking-rename`, `blocking-copy`, `blocking-read-dir`, `blocking-create-dir`, `blocking-create-dir-all`, `blocking-remove-dir`, `blocking-remove-dir-all`

## beagle.glob  (10 fns)
`code-at`, `is-slash-code`, `class-match`, `is-double-star`, `match-from`, `match?`, `translate`, `translate-class`, `find-walk`, `find`

## beagle.hash  (17 fns)
`to-bytes`, `add32`, `rotl32`, `rotr32`, `append-hex32!`, `append-hex8!`, `pad-message`, `word-be`, `sha1`, `sha256-digest-words`, `words-to-hex`, `sha256`, `sha256-k`, `words-to-bytes`, `concat-bytes`, `hmac-sha256`, `main`

## beagle.hex  (5 fns)
`to-bytes`, `nibble-to-code`, `code-to-nibble`, `encode`, `decode`

## beagle.http  (36 fns)
`reason-phrase`, `header`, `set-header`, `has-header?`, `make-response`, `ok`, `ok-with`, `html`, `json`, `not-found`, `status-response`, `redirect`, `conn-buf`, `byte-slice`, `cb-fill!`, `read-line`, `strip-cr`, `read-n`, `parse-request-line`, `parse-header-line`, `read-headers`, `read-request`, `serialize-response`, `byte-length`, `route`, `router`, `default-not-found`, `handle-connection`, `serve`, `serve-routes`, `read-response`, `read-until-eof`, `request`, `parse-url`, `get-url`, `main`

## beagle.ini  (12 fns)
`ws?`, `skip-ws-forward`, `trim-ws-backward`, `trim-range`, `unquote-value`, `find-separator`, `parse`, `parse-line`, `value-needs-quoting?`, `write-value!`, `write-section-body!`, `stringify`

## beagle.io  (20 fns)
`get-libc-path`, `io-err`, `is-null`, `string-to-buffer`, `open`, `close`, `read-bytes`, `read-char`, `write-bytes`, `write-string`, `flush`, `eof?`, `read-line`, `read-stdin`, `read-stdin-string`, `write-stdout`, `write-stdout-buffer`, `write-stderr`, `write-stdout-buffer-offset`, `read-stdin-line`

## beagle.ip  (7 fns)
`parse-octet`, `parse-ipv4`, `ipv4-to-string`, `prefix-to-mask`, `cidr-network`, `ipv4-in-cidr?`, `is-private-ipv4?`

## beagle.iter  (24 fns)
`iter-to-vec`, `mapcat`, `reductions`, `take-while`, `drop-while`, `partition`, `partition-all`, `chunk-by`, `zip`, `zip-with`, `interpose`, `flatten`, `iter-vector?`, `distinct`, `frequencies`, `group-by`, `count-by`, `repeat`, `cycle-take`, `enumerate`, `sum`, `product`, `min-by`, `max-by`

## beagle.json  (33 fns)
`cur-code`, `at-end?`, `ws-code?`, `skip-ws`, `parse-error`, `parse-document`, `parse-value`, `parse-literal`, `parse-object`, `parse-array`, `parse-string`, `parse-escape`, `parse-unicode-escape`, `read-hex4`, `hex-digit-value`, `encode-utf8`, `digit-code?`, `parse-number`, `parse-int-text`, `to-float-checked`, `parse-float-text`, `parse`, `parse-with-string-keys`, `stringify`, `stringify-pretty`, `write-value`, `write-newline-indent`, `write-array`, `write-object`, `key->string`, `write-string`, `write-control-escape`, `hex-char`

## beagle.log  (18 fns)
`level-name`, `normalize-level`, `set-level!`, `current-level`, `enabled?`, `format-fields`, `format-fields-helper`, `format-line`, `format-line-fields`, `build-emit-line`, `emit-line!`, `log-at`, `log`, `debug`, `info`, `warn`, `error`, `main`

## beagle.mathx  (18 fns)
`pi`, `e`, `abs-int`, `sign`, `gcd`, `lcm`, `clamp`, `lerp`, `min-of`, `max-of`, `is-even?`, `is-odd?`, `factorial`, `pow-int`, `deg->rad`, `rad->deg`, `floor-div`, `floor-mod`

## beagle.mutable-array  (18 fns)
`allocate-array-unsafe`, `read-field-unsafe`, `write-field-unsafe`, `is-array`, `panic-if-not-array`, `write-field`, `read-field`, `get`, `swap`, `new-array`, `copy-array`, `unsafe-copy-from-array-to`, `copy-from-array-to`, `copy-range`, `allocate-array-and-return`, `count`, `mod`, `main`

## beagle.os  (10 fns)
`getpid`, `getppid`, `getuid`, `getcwd`, `getenv`, `setenv`, `unsetenv`, `hostname`, `process-exit`, `main`

## beagle.path  (16 fns)
`slash-at?`, `is-absolute?`, `strip-trailing-slashes`, `last-slash-index`, `basename`, `dirname`, `join`, `join-all`, `ext-dot-index`, `extension`, `strip-extension`, `split-path`, `rebuild-path`, `normalize`, `check`, `main`

## beagle.priorityqueue  (15 fns)
`default-compare`, `pq-new`, `pq-new-with`, `pq-new-max`, `pq-count`, `pq-empty?`, `pq-swap`, `pq-sift-up`, `pq-sift-down`, `pq-push`, `pq-peek`, `pq-pop`, `pq-from-vec`, `pq-from-vec-with`, `pq-to-sorted-vec`

## beagle.process  (8 fns)
`get-libc-path`, `is-null-ptr`, `run-capture`, `read-all`, `run-capture-trim`, `system`, `exit-status`, `success?`

## beagle.random  (25 fns)
`normalize-seed`, `xorshift-step`, `next-int`, `next-float`, `int-range`, `choice`, `shuffle`, `sample`, `builtin-allocate-filled`, `primitive-read`, `primitive-write`, `nibble-to-code`, `append-random-nibbles!`, `uuid-v4`, `check`, `check-true`, `multiset-equal`, `main`, `contains-elem`, `distinct-count`, `assert-throws-bool`, `uuid-shape-ok`, `variant-ok`, `is-hex-char`, `uuid-hex-ok`

## beagle.regex-wrapper  (13 fns)
`as-regex`, `compile`, `regex?`, `match?`, `matches-full?`, `find`, `find-all`, `find-all-positions`, `split`, `captures`, `replace-all`, `replace-first`, `replace-with`

## beagle.repl-interactive  (4 fns)
`main`, `repl-loop`, `resume-prompt`, `handle-command`

## beagle.repl-main  (15 fns)
`process-eval-queue`, `process-requests`, `process-single-request`, `send-response`, `handle-eval`, `handle-interrupt`, `handle-describe`, `handle-ls-sessions`, `handle-close`, `handle-message`, `extract-lines`, `handle-client`, `start`, `run-with-repl`, `run-with-recovery`

## beagle.repl-session  (13 fns)
`create-session`, `rest-or-empty`, `pop-or-empty`, `session-eval-loop`, `build-responses`, `process-eval-request`, `session-eval`, `session-resume`, `session-abort`, `session-interrupt`, `session-close`, `session-busy?`, `session-suspend-depth`

## beagle.repl  (19 fns)
`register-main-crash-atom`, `get-or-create-session`, `remove-session`, `send-response`, `parse-json-line`, `handle-eval`, `handle-resume`, `handle-abort`, `handle-interrupt`, `handle-main-status`, `handle-main-resume`, `handle-main-abort`, `handle-describe`, `handle-ls-sessions`, `handle-close`, `handle-message`, `extract-lines`, `handle-client`, `start-repl-server`

## beagle.semver  (25 fns)
`digit?`, `alpha?`, `ident-char?`, `all-digits?`, `all-ident-chars?`, `valid-num-part?`, `valid-prerelease?`, `valid-build?`, `parse`, `parse-validated`, `valid?`, `cmp-int`, `cmp-string`, `cmp-pre-identifier`, `cmp-prerelease`, `cmp-pre-parts`, `compare-semver`, `compare`, `caret-upper`, `tilde-upper`, `ge?`, `lt?`, `satisfies?`, `require-parse`, `satisfies-parsed?`

## beagle.socket  (8 fns)
`listen`, `connect`, `accept`, `read`, `write`, `close`, `close-listener`, `on-connection`

## beagle.spawn  (9 fns)
`spawn`, `spawn-with-token`, `handle-spawn-blocking`, `handle-spawn-with-token-blocking`, `handle-spawn-threaded`, `handle-spawn-with-token-threaded`, `run-blocking`, `blocking-spawn`, `blocking-spawn-with-token`

## beagle.stats  (17 fns)
`as-vector`, `to-float`, `require-non-empty`, `sum`, `mean`, `min`, `max`, `range-of`, `median`, `mode`, `sum-squared-deviations`, `variance`, `variance-sample`, `stdev`, `stdev-sample`, `percentile`, `float-floor-to-int`

## beagle.stream  (39 fns)
`from-source`, `ensure-closed`, `next`, `close`, `map`, `filter`, `take`, `take-while`, `skip`, `flat-map`, `collect`, `reduce`, `fold`, `for-each`, `find`, `any?`, `all?`, `count`, `file-stream`, `file-stream-sized`, `file-stream-sync`, `file-stream-sync-sized`, `file-stream-sync`, `file-stream-sync-sized`, `read-dir-stream`, `split-on`, `lines`, `by-size`, `from-generator`, `range`, `repeat`, `from-vector`, `merge`, `zip`, `buffered`, `catch-default`, `retry`, `socket-stream`, `socket-stream-sized`

## beagle.streamtest  (24 fns)
`from-source`, `ensure-closed`, `stream-next`, `next`, `close`, `map`, `filter`, `take`, `collect`, `count`, `for-each`, `from-generator`, `from-vector`, `range`, `file-stream`, `file-stream-sized`, `read-dir-stream`, `socket-stream`, `socket-stream-sized`, `find-separator`, `extract-substring`, `split-on`, `lines`, `by-size`

## beagle.string-builder  (19 fns)
`new`, `append-byte!`, `append-char!`, `append!`, `append-range!`, `append-range-filter-byte!`, `append-range-uppercase!`, `append-builder-range!`, `index-byte`, `append-int!`, `append-float!`, `length`, `capacity`, `clear!`, `byte-at`, `set-byte-at!`, `reverse!`, `to-string`, `main`

## beagle.template  (24 fns)
`ws-code?`, `strip`, `matches-at?`, `find-from`, `tokenize`, `flush-literal`, `parse-path`, `parse-expr`, `tag-keyword`, `tag-rest`, `parse-body`, `is-stop?`, `parse-if`, `parse-for`, `parse`, `lookup`, `truthy?`, `html-escape`, `render-value`, `render-body`, `render-node`, `render-for`, `render`, `main`

## beagle.test  (15 fns)
`new-results`, `results-snapshot`, `record-pass`, `record-fail`, `assert-eq`, `assert-true`, `assert-throws`, `print-failures`, `print-summary`, `run-tests-loop`, `run-tests`, `demo-test-passing`, `demo-test-failing`, `demo-test-throws`, `main`

## beagle.text  (29 fns)
`code-at`, `ws-code?`, `digit-code?`, `repeat-string`, `pad-left`, `pad-right`, `clip-fill`, `capitalize`, `title-case`, `char-set-contains?`, `trim-chars`, `trim-chars-left`, `trim-chars-right`, `replace-all`, `split-with-limit`, `starts-with-any?`, `ends-with-any?`, `contains-any?`, `parse-float`, `hex-digit`, `int->hex`, `format`, `format-one`, `apply-pad`, `count-occurrences`, `reverse-string`, `string->bytes`, `bytes->string`, `ellipsize`

## beagle.time  (16 fns)
`libc-path`, `epoch-seconds`, `epoch-millis`, `floor-div`, `floor-mod`, `days-from-civil`, `civil-from-days`, `days-since-epoch`, `weekday`, `civil-from-epoch`, `append-2!`, `append-4!`, `http-date`, `check`, `check-true`, `main`

## beagle.timer  (9 fns)
`sleep`, `now`, `timeout`, `deadline`, `deadline-passed?`, `deadline-remaining`, `run-blocking`, `blocking-sleep`, `blocking-now`

## beagle.url  (15 fns)
`hex-digit`, `hex-value`, `unreserved-byte?`, `encode-char-into`, `encode-component`, `decode-component`, `decode-form`, `decode-into`, `append-char-bytes`, `parse-query`, `parse-pair-into`, `index-of-char`, `build-query`, `coerce-string`, `main`

## beagle.ws  (32 fns)
`opcode-continuation`, `opcode-text`, `opcode-binary`, `opcode-close`, `opcode-ping`, `opcode-pong`, `to-buf`, `length-or-zero`, `buf-byte`, `hex-to-bytes`, `hex-digit-value`, `accept-key`, `is-websocket-upgrade?`, `handshake-response`, `parse-frame`, `encode-frame`, `text-frame`, `binary-frame`, `ping-frame`, `pong-frame`, `close-frame`, `text?`, `binary?`, `ping?`, `pong?`, `close?`, `read-exact-buf`, `read-frame`, `do-handshake`, `serve-ws`, `handle-ws-connection`, `main`

## std  (146 fns)
`format-rust-vec-helper`, `format-rust-map-entries`, `deref`, `swap!`, `reset!`, `compare-and-swap!`, `atom`, `instance-of`, `format-persistent-vector-helper`, `join`, `join-helper`, `parse-int`, `parse-int-helper`, `new-string-buffer`, `print`, `println`, `keyword?`, `keyword->string`, `string->keyword`, `range`, `range-step`, `reduce`, `map`, `filter`, `any?`, `all?`, `none?`, `not-every?`, `find`, `find-index`, `take`, `drop`, `take-while`, `drop-while`, `slice`, `enumerate`, `remove-at`, `count`, `empty?`, `first-of`, `last`, `rest`, `butlast`, `nth`, `second`, `third`, `min-of`, `max-of`, `min-by`, `max-by`, `reduce-right`, `concat`, `flatten`, `flat-map`, `zip`, `zip-with`, `zipmap`, `interleave`, `interpose`, `into`, `reverse`, `repeat`, `repeatedly`, `iterate`, `partition`, `partition-by`, `group-by`, `frequencies`, `distinct`, `dedupe`, `tim-allocate-array`, `tim-reverse`, `tim-count-run-and-make-ascending`, `tim-binary-insertion-sort`, `tim-calc-min-run`, `tim-merge`, `tim-merge-lo`, `tim-merge-hi`, `tim-merge-at`, `tim-merge-collapse`
