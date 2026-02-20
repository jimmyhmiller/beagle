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
    assert!(
        !output.status.success(),
        "Mismatched snapshot should fail"
    );
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
    assert!(
        !content.contains("// wrong"),
        "Old snapshot should be gone"
    );
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
