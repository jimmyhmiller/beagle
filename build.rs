use std::fs;
use std::path::Path;
use std::path::PathBuf;

fn get_build_profile_name() -> String {
    // The profile name is always the 3rd last part of the path (with 1 based indexing).
    // e.g. /code/core/target/cli/build/my-build-info-9f91ba6f99d7a061/out
    std::env::var("OUT_DIR")
        .unwrap()
        .split(std::path::MAIN_SEPARATOR)
        .nth_back(3)
        .unwrap_or("unknown")
        .to_string()
}

fn main() {
    // Get the build profile (either "debug" or "release")
    let profile = get_build_profile_name();
    let target_dir = PathBuf::from("target").join(&profile); // Points to "target/debug" or "target/release"

    // Define the source and destination paths
    let source_dir = PathBuf::from("resources");
    let dest_dir = target_dir.join("resources");

    // Copy the entire directory recursively
    if let Err(e) = fs::create_dir_all(&dest_dir) {
        panic!("Failed to create destination directory: {}", e);
    }

    copy_recursively(&source_dir, &dest_dir).unwrap();

    // Also copy standard-library/ next to the binary for development
    let stdlib_source = PathBuf::from("standard-library");
    let stdlib_dest = target_dir.join("standard-library");
    if stdlib_source.exists() {
        if let Err(e) = fs::create_dir_all(&stdlib_dest) {
            panic!("Failed to create stdlib destination directory: {}", e);
        }
        copy_recursively(&stdlib_source, &stdlib_dest).unwrap();
    }
}

// Helper function to recursively copy files
fn copy_recursively(src: &PathBuf, dst: &Path) -> std::io::Result<()> {
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        let dest_path = dst.join(entry.file_name());

        if path.is_dir() {
            fs::create_dir_all(&dest_path)?;
            copy_recursively(&path, &dest_path)?;
        } else {
            fs::copy(&path, &dest_path)?;
        }
    }
    Ok(())
}
