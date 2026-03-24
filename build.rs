fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.lock");

    // When the cef feature is enabled, add the CEF distribution directory
    // to the binary's RPATH so libcef.so can be found at runtime without
    // requiring LD_LIBRARY_PATH.
    if std::env::var("CARGO_FEATURE_CEF").is_ok() {
        if let Ok(out_dir) = std::env::var("OUT_DIR") {
            let build_dir = std::path::Path::new(&out_dir)
                .ancestors()
                .find(|p| p.file_name().map_or(false, |n| n == "build"));

            if let Some(build_dir) = build_dir {
                if let Ok(entries) = std::fs::read_dir(build_dir) {
                    for entry in entries.flatten() {
                        let name = entry.file_name();
                        if name.to_string_lossy().starts_with("cef-dll-sys-") {
                            // Use TARGET arch, not HOST arch, for cross-compilation
                            let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH")
                                .unwrap_or_else(|_| std::env::consts::ARCH.to_string());
                            let cef_dir = entry.path().join("out").join(format!(
                                "cef_{}_{}",
                                std::env::consts::OS,
                                target_arch
                            ));
                            if cef_dir.exists() {
                                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cef_dir.display());
                                strip_cef_libs(&cef_dir);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Strip debug info and symbols from CEF shared libraries in release builds.
/// The CEF "minimal" distribution still ships with full debug info (~1.5GB libcef.so),
/// stripping brings it down to ~237MB.
fn strip_cef_libs(cef_dir: &std::path::Path) {
    let profile = std::env::var("PROFILE").unwrap_or_default();
    if profile != "release" {
        return;
    }

    let marker = cef_dir.join(".stripped");
    if marker.exists() {
        return;
    }

    // Use the appropriate strip for cross-compilation
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let strip_cmd = if target_arch == "aarch64" {
        "aarch64-linux-gnu-strip"
    } else {
        "strip"
    };

    let libs = ["libcef.so", "libEGL.so", "libGLESv2.so", "chrome-sandbox"];

    for lib in &libs {
        let path = cef_dir.join(lib);
        if path.exists() {
            let status = std::process::Command::new(strip_cmd)
                .arg("--strip-all")
                .arg(&path)
                .status();
            match status {
                Ok(s) if s.success() => {
                    println!("cargo:warning=Stripped {}", lib);
                }
                _ => {
                    println!("cargo:warning=Failed to strip {}", lib);
                }
            }
        }
    }

    // Mark as stripped so we don't re-strip on incremental builds
    let _ = std::fs::write(&marker, "");
}
