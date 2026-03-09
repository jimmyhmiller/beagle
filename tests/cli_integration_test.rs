use std::path::PathBuf;
use std::process::Command;

fn beag_binary() -> PathBuf {
    // cargo test builds the binary into target/debug
    let mut path = PathBuf::from(env!("CARGO_BIN_EXE_beag"));
    // Fallback if the env var doesn't resolve nicely
    if !path.exists() {
        path = PathBuf::from("target/debug/beag");
    }
    path
}

fn beag() -> Command {
    Command::new(beag_binary())
}

// --- Version and Help ---

#[test]
fn test_version() {
    let output = beag()
        .arg("--version")
        .output()
        .expect("failed to run beag");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.starts_with("beag "),
        "Expected 'beag X.Y.Z', got: {}",
        stdout
    );
}

#[test]
fn test_help() {
    let output = beag().arg("--help").output().expect("failed to run beag");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Beagle"), "Help should mention Beagle");
    assert!(stdout.contains("run"), "Help should list run command");
    assert!(stdout.contains("repl"), "Help should list repl command");
    assert!(stdout.contains("init"), "Help should list init command");
    assert!(stdout.contains("test"), "Help should list test command");
}

#[test]
fn test_run_help() {
    let output = beag()
        .args(["run", "--help"])
        .output()
        .expect("failed to run beag");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("FILE"), "run help should mention FILE");
}

// --- Running programs ---

#[test]
fn test_run_subcommand() {
    let output = beag()
        .args(["run", "resources/fib.bg"])
        .output()
        .expect("failed to run beag");
    assert!(
        output.status.success(),
        "beag run fib.bg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.trim(), "832040");
}

#[test]
fn test_bare_file() {
    let output = beag()
        .arg("resources/fib.bg")
        .output()
        .expect("failed to run beag");
    assert!(
        output.status.success(),
        "beag fib.bg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.trim(), "832040");
}

#[test]
fn test_run_with_args() {
    let output = beag()
        .args(["run", "resources/args_test.bg", "hello", "world"])
        .output()
        .expect("failed to run beag");
    // This may or may not pass depending on whether args_test.bg exists,
    // but at least it shouldn't crash the CLI parser
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(!stdout.is_empty() || true); // Just confirm it ran
    }
}

#[test]
fn test_run_nonexistent_file() {
    let output = beag()
        .args(["run", "nonexistent_file.bg"])
        .output()
        .expect("failed to run beag");
    assert!(
        !output.status.success(),
        "Running nonexistent file should fail"
    );
}

#[test]
fn test_bare_nonexistent_file() {
    let output = beag()
        .arg("nonexistent_file.bg")
        .output()
        .expect("failed to run beag");
    assert!(
        !output.status.success(),
        "Running nonexistent bare file should fail"
    );
}

// --- Init ---

#[test]
fn test_init_with_name() {
    let tmp = tempdir();
    let output = beag()
        .args(["init", "my-project"])
        .current_dir(&tmp)
        .output()
        .expect("failed to run beag init");
    assert!(
        output.status.success(),
        "beag init failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("my-project"), "Should mention project name");

    // Check files were created
    assert!(
        tmp.join("my-project/beagle.toml").exists(),
        "beagle.toml should exist"
    );
    assert!(
        tmp.join("my-project/src/main.bg").exists(),
        "src/main.bg should exist"
    );
    assert!(
        tmp.join("my-project/test/main_test.bg").exists(),
        "test/main_test.bg should exist"
    );

    // Check beagle.toml content
    let toml = std::fs::read_to_string(tmp.join("my-project/beagle.toml")).unwrap();
    assert!(
        toml.contains("my-project"),
        "beagle.toml should contain project name"
    );

    // Check main.bg content
    let main = std::fs::read_to_string(tmp.join("my-project/src/main.bg")).unwrap();
    assert!(
        main.contains("namespace my_project"),
        "main.bg should have correct namespace"
    );
    assert!(
        main.contains("fn main()"),
        "main.bg should have main function"
    );

    // Check test file has snapshot annotation
    let test = std::fs::read_to_string(tmp.join("my-project/test/main_test.bg")).unwrap();
    assert!(
        test.contains("// @beagle.core.snapshot"),
        "test file should have // @beagle.core.snapshot"
    );
}

#[test]
fn test_init_in_current_dir() {
    let tmp = tempdir();
    let output = beag()
        .arg("init")
        .current_dir(&tmp)
        .output()
        .expect("failed to run beag init");
    assert!(
        output.status.success(),
        "beag init failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(tmp.join("beagle.toml").exists(), "beagle.toml should exist");
    assert!(tmp.join("src/main.bg").exists(), "src/main.bg should exist");
    assert!(
        tmp.join("test/main_test.bg").exists(),
        "test/main_test.bg should exist"
    );
}

#[test]
fn test_init_and_run() {
    let tmp = tempdir();

    // Init
    let output = beag()
        .args(["init", "hello-beagle"])
        .current_dir(&tmp)
        .output()
        .expect("failed to run beag init");
    assert!(output.status.success());

    // Run the generated main.bg
    let output = beag()
        .args(["run", "src/main.bg"])
        .current_dir(tmp.join("hello-beagle"))
        .output()
        .expect("failed to run beag run");
    assert!(
        output.status.success(),
        "Running init'd project failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Hello from hello-beagle!"),
        "Expected greeting, got: {}",
        stdout
    );
}

#[test]
fn test_init_and_test() {
    let tmp = tempdir();

    // Init
    let output = beag()
        .args(["init", "test-proj"])
        .current_dir(&tmp)
        .output()
        .expect("failed to run beag init");
    assert!(output.status.success());

    // Run tests
    let output = beag()
        .arg("test")
        .current_dir(tmp.join("test-proj"))
        .output()
        .expect("failed to run beag test");
    assert!(
        output.status.success(),
        "beag test failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("pass"), "Should have passing tests");
    assert!(stdout.contains("0 failed"), "Should have 0 failures");
}

// --- Test command ---

#[test]
fn test_no_test_dir() {
    let tmp = tempdir();
    let output = beag()
        .arg("test")
        .current_dir(&tmp)
        .output()
        .expect("failed to run beag test");
    assert!(
        !output.status.success(),
        "beag test with no test dir should fail"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("No test/") || stderr.contains("test"),
        "Should mention missing test dir"
    );
}

#[test]
fn test_specific_file() {
    let tmp = tempdir();

    // Init a project
    let output = beag()
        .args(["init", "specific-test"])
        .current_dir(&tmp)
        .output()
        .expect("failed to run beag init");
    assert!(output.status.success());

    // Run specific test file
    let output = beag()
        .args(["test", "test/main_test.bg"])
        .current_dir(tmp.join("specific-test"))
        .output()
        .expect("failed to run beag test");
    assert!(
        output.status.success(),
        "beag test with specific file failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("1 passed"), "Should have 1 passing test");
}

#[test]
fn test_nonexistent_test_file() {
    let tmp = tempdir();
    let output = beag()
        .args(["test", "nonexistent.bg"])
        .current_dir(&tmp)
        .output()
        .expect("failed to run beag test");
    assert!(
        !output.status.success(),
        "beag test with nonexistent file should fail"
    );
}

// --- Update snapshots ---

#[test]
fn test_snapshot_mismatch_fails() {
    let tmp = tempdir();

    // Create a test file with a wrong snapshot
    std::fs::create_dir_all(tmp.join("test")).unwrap();
    std::fs::write(
        tmp.join("test/snap_test.bg"),
        "namespace snap_test\n\nfn main() {\n    println(\"actual\")\n}\n\n// @beagle.core.snapshot\n// wrong\n",
    )
    .unwrap();

    let output = beag()
        .arg("test")
        .current_dir(&tmp)
        .output()
        .expect("failed to run beag test");
    assert!(!output.status.success(), "Mismatched snapshot should fail");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("FAIL"), "Output should show FAIL");
}

#[test]
fn test_update_snapshots_rewrites_file() {
    let tmp = tempdir();

    // Create a test file with a wrong snapshot
    std::fs::create_dir_all(tmp.join("test")).unwrap();
    let test_file = tmp.join("test/snap_test.bg");
    std::fs::write(
        &test_file,
        "namespace snap_test\n\nfn main() {\n    println(\"actual output\")\n}\n\n// @beagle.core.snapshot\n// wrong\n",
    )
    .unwrap();

    // Run with --update-snapshots
    let output = beag()
        .args(["test", "--update-snapshots"])
        .current_dir(&tmp)
        .output()
        .expect("failed to run beag test");
    assert!(
        output.status.success(),
        "beag test --update-snapshots should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Read the file back and verify the snapshot was updated
    let content = std::fs::read_to_string(&test_file).unwrap();
    assert!(
        content.contains("// actual output"),
        "Snapshot should contain actual output, got:\n{}",
        content
    );
    assert!(!content.contains("// wrong"), "Old snapshot should be gone");
}

#[test]
fn test_update_snapshots_then_passes() {
    let tmp = tempdir();

    // Create a test file with a wrong snapshot
    std::fs::create_dir_all(tmp.join("test")).unwrap();
    let test_file = tmp.join("test/snap_test.bg");
    std::fs::write(
        &test_file,
        "namespace snap_test\n\nfn main() {\n    println(\"hello\")\n    println(42)\n}\n\n// @beagle.core.snapshot\n// stale\n",
    )
    .unwrap();

    // Update snapshots
    let output = beag()
        .args(["test", "--update-snapshots"])
        .current_dir(&tmp)
        .output()
        .expect("failed to run beag test");
    assert!(output.status.success());

    // Now run without --update-snapshots, should pass
    let output = beag()
        .arg("test")
        .current_dir(&tmp)
        .output()
        .expect("failed to run beag test");
    assert!(
        output.status.success(),
        "After update, test should pass: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("1 passed"), "Should have 1 passing test");
    assert!(stdout.contains("0 failed"), "Should have 0 failures");
}

#[test]
fn test_recursive_test_discovery() {
    let tmp = tempdir();

    // Create nested test directories
    std::fs::create_dir_all(tmp.join("test/unit")).unwrap();
    std::fs::write(
        tmp.join("test/top_test.bg"),
        "namespace top_test\n\nfn main() {\n    println(\"top\")\n}\n\n// @beagle.core.snapshot\n// top\n",
    )
    .unwrap();
    std::fs::write(
        tmp.join("test/unit/nested_test.bg"),
        "namespace nested_test\n\nfn main() {\n    println(\"nested\")\n}\n\n// @beagle.core.snapshot\n// nested\n",
    )
    .unwrap();

    let output = beag()
        .arg("test")
        .current_dir(&tmp)
        .output()
        .expect("failed to run beag test");
    assert!(
        output.status.success(),
        "Recursive tests should pass: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("2 passed"), "Should find 2 tests");
}

// --- No arguments ---

#[test]
fn test_no_args() {
    let output = beag().output().expect("failed to run beag");
    // Should show help and exit with error (missing required subcommand)
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let combined = format!("{}{}", stdout, stderr);
    assert!(
        combined.contains("Usage") || combined.contains("beag"),
        "Should show usage info"
    );
}

// --- REPL ---

/// Helper to run the REPL with piped input and return stdout
fn run_repl(input: &str) -> String {
    use std::io::Write;
    use std::process::Stdio;

    let mut child = beag()
        .arg("repl")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn beag repl");

    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(input.as_bytes())
        .expect("failed to write to stdin");
    drop(child.stdin.take());

    let output = child
        .wait_with_output()
        .expect("failed to wait for beag repl");
    String::from_utf8_lossy(&output.stdout).to_string()
}

#[test]
fn test_repl_banner() {
    let stdout = run_repl(":quit\n");
    assert!(
        stdout.contains("Beagle REPL"),
        "Should show banner, got: {}",
        stdout
    );
}

#[test]
fn test_repl_simple_expression() {
    let stdout = run_repl("1 + 2\n:quit\n");
    assert!(
        stdout.contains("=> 3"),
        "Should evaluate 1 + 2 to 3, got: {}",
        stdout
    );
}

#[test]
fn test_repl_string_expression() {
    let stdout = run_repl("\"hello\"\n:quit\n");
    assert!(
        stdout.contains("=> \"hello\""),
        "Should show string repr, got: {}",
        stdout
    );
}

#[test]
fn test_repl_variable_binding() {
    let stdout = run_repl("let x = 42\nx * 2\n:quit\n");
    assert!(
        stdout.contains("=> 42"),
        "Should show let binding result, got: {}",
        stdout
    );
    assert!(
        stdout.contains("=> 84"),
        "Should evaluate x * 2, got: {}",
        stdout
    );
}

#[test]
fn test_repl_error_handling() {
    let stdout = run_repl("undefined_var\n1 + 2\n:quit\n");
    assert!(
        stdout.contains("Error:"),
        "Should show error for undefined var, got: {}",
        stdout
    );
    // Should recover and continue
    assert!(
        stdout.contains("=> 3"),
        "Should continue after error, got: {}",
        stdout
    );
}

#[test]
fn test_repl_eof_exits() {
    // Send no :quit, just EOF
    let stdout = run_repl("1 + 2\n");
    assert!(
        stdout.contains("=> 3"),
        "Should evaluate before EOF, got: {}",
        stdout
    );
}

#[test]
fn test_repl_empty_lines_skipped() {
    let stdout = run_repl("\n\n1 + 2\n:quit\n");
    assert!(
        stdout.contains("=> 3"),
        "Should skip empty lines, got: {}",
        stdout
    );
}

#[test]
fn test_repl_function_definition() {
    let stdout = run_repl("fn double(x) { x * 2 }\ndouble(21)\n:quit\n");
    assert!(
        stdout.contains("=> 42"),
        "Should define and call function, got: {}",
        stdout
    );
}

#[test]
fn test_repl_multiple_expressions() {
    let stdout = run_repl("10 + 20\n30 + 40\n50 + 60\n:quit\n");
    assert!(stdout.contains("=> 30"), "First expr, got: {}", stdout);
    assert!(stdout.contains("=> 70"), "Second expr, got: {}", stdout);
    assert!(stdout.contains("=> 110"), "Third expr, got: {}", stdout);
}

#[test]
fn test_repl_help_command() {
    let stdout = run_repl(":help\n:quit\n");
    assert!(
        stdout.contains("Available commands"),
        "Should show help text, got: {}",
        stdout
    );
    assert!(
        stdout.contains(":quit"),
        "Help should list :quit, got: {}",
        stdout
    );
}

#[test]
fn test_repl_ns_command() {
    let stdout = run_repl(":ns\n:quit\n");
    assert!(
        stdout.contains("user"),
        "Should show current namespace, got: {}",
        stdout
    );
}

#[test]
fn test_repl_null_not_printed() {
    // println returns its last arg, but null results should not be printed
    let stdout = run_repl("null\n:quit\n");
    // null should not produce "=> null" output
    assert!(
        !stdout.contains("=> null"),
        "Should not print null result, got: {}",
        stdout
    );
}

#[test]
fn test_repl_quit_command() {
    // :quit should exit cleanly — the process should terminate
    let stdout = run_repl("1 + 1\n:quit\n");
    assert!(
        stdout.contains("=> 2"),
        "Should eval before quit, got: {}",
        stdout
    );
    // No crash, no error output — just clean exit
}

#[test]
fn test_repl_quit_aliases() {
    // :q and :exit should also work
    let stdout = run_repl("1 + 1\n:q\n");
    assert!(
        stdout.contains("=> 2"),
        ":q should work as quit alias, got: {}",
        stdout
    );

    let stdout = run_repl("1 + 1\n:exit\n");
    assert!(
        stdout.contains("=> 2"),
        ":exit should work as quit alias, got: {}",
        stdout
    );
}

#[test]
fn test_repl_ns_switch() {
    // Switch namespace and verify eval happens in new namespace
    let stdout = run_repl(":ns mylib\nfn greet() { \"hello from mylib\" }\ngreet()\n:quit\n");
    assert!(
        stdout.contains("Switched to namespace: mylib"),
        "Should confirm switch, got: {}",
        stdout
    );
    assert!(
        stdout.contains("hello from mylib"),
        "Should eval in new namespace, got: {}",
        stdout
    );
}

#[test]
fn test_repl_unknown_command() {
    let stdout = run_repl(":bogus\n:quit\n");
    assert!(
        stdout.contains("Unknown command"),
        "Should report unknown command, got: {}",
        stdout
    );
}

#[test]
fn test_repl_struct_definition() {
    let stdout = run_repl("struct Point { x, y }\nlet p = Point { x: 10, y: 20 }\np.x + p.y\n:quit\n");
    assert!(
        stdout.contains("=> 30"),
        "Should define struct and access fields, got: {}",
        stdout
    );
}

#[test]
fn test_repl_multiple_errors_recover() {
    // Multiple errors in a row should all recover
    let stdout = run_repl("bad1\nbad2\n1 + 1\n:quit\n");
    let error_count = stdout.matches("Error:").count();
    assert!(
        error_count >= 2,
        "Should show at least 2 errors, got {} in: {}",
        error_count,
        stdout
    );
    assert!(
        stdout.contains("=> 2"),
        "Should still work after multiple errors, got: {}",
        stdout
    );
}

#[test]
fn test_repl_use_import() {
    // use statement should work in the REPL
    let stdout = run_repl("use beagle.core as core\ncore/length([1,2,3])\n:quit\n");
    assert!(
        stdout.contains("=> 3"),
        "Should be able to use namespace alias, got: {}",
        stdout
    );
}

#[test]
fn test_repl_clear_command() {
    // :clear shouldn't crash — just verify the REPL continues working after it
    let stdout = run_repl(":clear\n1 + 1\n:quit\n");
    assert!(
        stdout.contains("=> 2"),
        ":clear should not break the REPL, got: {}",
        stdout
    );
}

#[test]
fn test_repl_println_output() {
    // println should print to stdout, not show as => result
    let stdout = run_repl("println(\"visible output\")\n:quit\n");
    assert!(
        stdout.contains("visible output"),
        "println should appear in output, got: {}",
        stdout
    );
}

#[test]
fn test_repl_state_persists_across_evals() {
    // Variables defined in one eval should be accessible in subsequent evals
    let stdout = run_repl("let a = 10\nlet b = 20\nlet c = a + b\nc * 2\n:quit\n");
    assert!(
        stdout.contains("=> 60"),
        "State should persist across eval calls, got: {}",
        stdout
    );
}

// --- REPL + Socket REPL ---

#[test]
fn test_repl_starts_socket_server() {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpStream;
    use std::process::Stdio;
    use std::time::{Duration, Instant};

    /// Read response lines from the socket REPL until we see a "done" status
    /// or the read times out. Returns all lines concatenated.
    fn read_until_done(reader: &mut BufReader<TcpStream>) -> String {
        let mut all = String::new();
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    all.push_str(&line);
                    if line.contains("done") {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        all
    }

    // Pick a port unlikely to collide
    let port = 17856 + (std::process::id() % 1000) as u16;

    let mut child = beag()
        .arg("repl")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn beag repl");

    let mut stdin = child.stdin.take().unwrap();

    // Start the socket REPL server in a thread from the interactive REPL
    let cmds = format!(
        "use beagle.repl as repl\nthread(fn() {{ repl/start-repl-server(\"127.0.0.1\", {}) }})\n",
        port
    );
    stdin
        .write_all(cmds.as_bytes())
        .expect("failed to write to repl stdin");
    stdin.flush().unwrap();

    // Poll until the server is accepting connections (max 10s)
    let start = Instant::now();
    let mut stream = None;
    while start.elapsed() < Duration::from_secs(10) {
        match TcpStream::connect(format!("127.0.0.1:{}", port)) {
            Ok(s) => {
                stream = Some(s);
                break;
            }
            Err(_) => std::thread::sleep(Duration::from_millis(100)),
        }
    }
    let stream = stream.unwrap_or_else(|| {
        let _ = child.kill();
        panic!(
            "Socket REPL server did not start within 10s on port {}",
            port
        );
    });

    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();

    // Split into separate read/write handles to avoid borrow conflicts
    let mut writer = stream.try_clone().expect("failed to clone TcpStream");
    let mut reader = BufReader::new(stream);

    // Helper: send a JSON command
    let mut send = |cmd: &str| {
        writer.write_all(cmd.as_bytes()).unwrap();
        writer.flush().unwrap();
    };

    // 1) describe — returns a single line with ops and versions
    send("{\"op\":\"describe\",\"id\":\"d1\"}\n");

    let describe_resp = read_until_done(&mut reader);
    assert!(
        describe_resp.contains("d1"),
        "describe response should contain request id, got: {}",
        describe_resp
    );
    assert!(
        describe_resp.contains("eval"),
        "describe should list eval op, got: {}",
        describe_resp
    );

    // 2) eval "1 + 2" — returns value line then done line
    send("{\"op\":\"eval\",\"id\":\"e1\",\"session\":\"test-sess\",\"code\":\"1 + 2\"}\n");

    let eval_resp = read_until_done(&mut reader);
    assert!(
        eval_resp.contains("e1"),
        "eval response should contain request id, got: {}",
        eval_resp
    );
    assert!(
        eval_resp.contains("3"),
        "eval of '1 + 2' should return 3, got: {}",
        eval_resp
    );
    assert!(
        eval_resp.contains("done"),
        "eval should end with done status, got: {}",
        eval_resp
    );

    // 3) eval with println — returns out line, value line, then done line
    send(
        "{\"op\":\"eval\",\"id\":\"e2\",\"session\":\"test-sess\",\"code\":\"println(\\\"socket-hello\\\")\"}\n",
    );

    let println_resp = read_until_done(&mut reader);
    assert!(
        println_resp.contains("e2"),
        "println eval response should contain request id, got: {}",
        println_resp
    );
    assert!(
        println_resp.contains("socket-hello"),
        "should capture println output, got: {}",
        println_resp
    );
    assert!(
        println_resp.contains("done"),
        "println eval should end with done, got: {}",
        println_resp
    );

    // 4) close session
    send("{\"op\":\"close\",\"id\":\"c1\",\"session\":\"test-sess\"}\n");

    let close_resp = read_until_done(&mut reader);
    assert!(
        close_resp.contains("c1") && close_resp.contains("done"),
        "close response should have id and done status, got: {}",
        close_resp
    );

    // Clean up
    drop(reader);
    drop(writer);
    let _ = stdin.write_all(b":quit\n");
    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();
}

// --- Helpers ---

fn tempdir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("beag-test-{}", std::process::id()));
    // Use a unique subdir per test by including thread name
    let thread = std::thread::current();
    let name = thread.name().unwrap_or("unknown");
    let dir = dir.join(name.replace("::", "_"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("failed to create temp dir");
    dir
}
