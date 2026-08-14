//! Locate bundled native runtime libraries relative to the running executable.
//!
//! CEF and pdfium ship as loose shared libraries rather than as system packages with a stable
//! soname on the loader's search path, so every application embedding them has to answer the
//! same question at startup: *where is the distribution?* The answer is a property of how the
//! application was installed, not of the application itself:
//!
//! | layout        | CEF lives at                      |
//! |---------------|-----------------------------------|
//! | FHS (Fedora)  | `/usr/lib/cef`                    |
//! | Nix           | `<store path>/lib/cef`            |
//! | dev build     | wherever `LD_LIBRARY_PATH` points |
//!
//! Resolving against [`install_prefix`] covers the first two with one rule and no environment
//! at all, because `<prefix>/bin/<exe>` implies `<prefix>/lib`. That is what lets the same
//! binary work on both an FHS distribution and a non-FHS one.
//!
//! This crate is deliberately dependency-free so that a consumer needing only the lookup — a
//! PDF viewer wanting pdfium, say — does not have to pull in the webview stack to get it.
//!
//! ```no_run
//! // At startup, before creating a webview:
//! native_lib_paths::ensure_cef_path();
//! ```

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

/// File whose presence identifies a directory as a CEF distribution.
#[cfg(target_os = "linux")]
pub const CEF_LIB: &str = "libcef.so";
#[cfg(target_os = "macos")]
pub const CEF_LIB: &str = "Chromium Embedded Framework";
#[cfg(target_os = "windows")]
pub const CEF_LIB: &str = "libcef.dll";

/// Where the `cef-libs` package puts the distribution on FHS systems.
pub const CEF_FHS_DIR: &str = "/usr/lib/cef";

/// Environment variable the `cef` crate reads to locate the distribution.
pub const CEF_PATH_ENV: &str = "CEF_PATH";

/// Library directories to try under an install prefix, in order.
///
/// Both spellings are needed: Debian-family and Nix use `lib`, Fedora-family uses `lib64`.
pub const LIB_SUBDIRS: [&str; 2] = ["lib", "lib64"];

/// The install prefix of `exe`, i.e. the `<prefix>` of `<prefix>/bin/<name>`.
///
/// `None` when the executable is not in a `bin` directory — a `target/debug` build has no
/// sibling `lib`, and guessing one would find the wrong thing.
#[must_use]
pub fn install_prefix(exe: &Path) -> Option<&Path> {
    let bin = exe.parent()?;
    if bin.file_name()? != "bin" {
        return None;
    }
    bin.parent()
}

/// Where a CEF distribution sits under an install prefix.
const CEF_SUBDIR: &str = "lib/cef";

/// XDG spec default for `XDG_DATA_DIRS`.
const DEFAULT_XDG_DATA_DIRS: &str = "/usr/local/share:/usr/share";

/// The environment a CEF lookup depends on, captured so the search is testable.
///
/// Every field is optional and an absent one simply contributes no candidates, so a caller
/// that does not care about (say) `LD_LIBRARY_PATH` can leave it unset.
#[derive(Debug, Default, Clone)]
pub struct CefSearch {
    /// The running executable, used to derive the install prefix.
    pub exe: Option<PathBuf>,
    /// `$CEF_PATH` — an explicit override, tried first.
    pub cef_path: Option<OsString>,
    /// `$XDG_DATA_DIRS` — covers a CEF that lives in its own prefix (its own Nix store path).
    pub xdg_data_dirs: Option<OsString>,
    /// `$LD_LIBRARY_PATH` — how a development run points at a build-tree CEF.
    pub ld_library_path: Option<OsString>,
}

impl CefSearch {
    /// Capture the search inputs from the current process.
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            exe: std::env::current_exe().ok(),
            cef_path: std::env::var_os(CEF_PATH_ENV),
            xdg_data_dirs: std::env::var_os("XDG_DATA_DIRS"),
            ld_library_path: std::env::var_os("LD_LIBRARY_PATH"),
        }
    }

    /// Directories that may hold a CEF distribution, most specific first.
    ///
    /// Order is deliberate: an explicit `CEF_PATH` wins, then the prefix the binary was
    /// installed into, then each `XDG_DATA_DIRS` entry's prefix (`/usr/share` -> `/usr/lib/cef`,
    /// and on Nix a CEF in its own store path), then `LD_LIBRARY_PATH`, and finally the FHS
    /// location as a last resort so an existing Fedora install keeps resolving exactly as before.
    #[must_use]
    pub fn candidates(&self) -> Vec<PathBuf> {
        let mut dirs: Vec<PathBuf> = Vec::new();

        if let Some(path) = self.cef_path.as_deref().filter(|p| !p.is_empty()) {
            dirs.push(PathBuf::from(path));
        }
        if let Some(prefix) = self.exe.as_deref().and_then(install_prefix) {
            dirs.push(prefix.join(CEF_SUBDIR));
        }

        let data_dirs = self
            .xdg_data_dirs
            .as_deref()
            .filter(|d| !d.is_empty())
            .unwrap_or(OsStr::new(DEFAULT_XDG_DATA_DIRS));
        dirs.extend(
            std::env::split_paths(data_dirs)
                .filter(|p| p.is_absolute())
                .filter_map(|d| d.parent().map(|prefix| prefix.join(CEF_SUBDIR))),
        );

        if let Some(path) = self.ld_library_path.as_deref() {
            dirs.extend(std::env::split_paths(path).filter(|p| !p.as_os_str().is_empty()));
        }
        dirs.push(PathBuf::from(CEF_FHS_DIR));

        dedup_preserving_order(dirs)
    }

    /// The first candidate that actually contains a CEF distribution.
    #[must_use]
    pub fn find(&self) -> Option<PathBuf> {
        self.find_with(&|dir: &Path| dir.join(CEF_LIB).is_file())
    }

    /// [`Self::find`] with the "is this a CEF directory?" test injected, for tests that must
    /// not depend on what happens to be installed on the machine running them.
    #[must_use]
    pub fn find_with(&self, is_cef_dir: &dyn Fn(&Path) -> bool) -> Option<PathBuf> {
        self.candidates().into_iter().find(|dir| is_cef_dir(dir))
    }
}

/// The first directory that actually contains a CEF distribution, searching from the
/// current process environment.
#[must_use]
pub fn find_cef_dir() -> Option<PathBuf> {
    CefSearch::from_env().find()
}

/// Locate CEF and export [`CEF_PATH_ENV`] so the `cef` crate finds the same distribution.
///
/// Returns the directory, or `None` when nothing was found — in which case the environment is
/// left alone. That matters for a binary run out of a build tree: `cef-dll-sys` falls back to
/// the copy unpacked into its `OUT_DIR`, and overwriting `CEF_PATH` would take that away.
///
/// A `CEF_PATH` that is already set and valid is returned untouched. One that points at
/// nothing is treated as stale and the search continues, since falling through beats refusing
/// to start the webview.
///
/// Sets a process-wide environment variable, so **call this from `main` before starting any
/// threads**.
pub fn ensure_cef_path() -> Option<PathBuf> {
    let search = CefSearch::from_env();
    let resolved = search.find()?;

    if search.cef_path.as_deref() != Some(resolved.as_os_str()) {
        // SAFETY: documented as main-thread-before-threads. This crate is edition 2021, where
        // set_var is safe; the requirement is the same either way and is the caller's to meet.
        std::env::set_var(CEF_PATH_ENV, &resolved);
    }

    Some(resolved)
}

/// Candidate paths for a native library file, most specific first.
///
/// `override_path` (typically from an application-specific environment variable) may name
/// either the library file itself or the directory holding it. The current directory is
/// included so a dev run can drop the library next to the build tree, and the install prefix
/// is expanded over [`LIB_SUBDIRS`].
///
/// Callers should treat a bare-soname `dlopen` as the final fallback: on a distribution that
/// packages the library properly, the loader already knows where it is.
#[must_use]
pub fn library_candidates(
    lib_name: &OsStr,
    override_path: Option<&OsStr>,
    exe: Option<&Path>,
) -> Vec<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Some(path) = override_path.filter(|p| !p.is_empty()).map(PathBuf::from) {
        // Accept the library file itself as well as the directory holding it.
        if path.file_name() == Some(lib_name) {
            candidates.push(path);
        } else {
            candidates.push(path.join(lib_name));
        }
    }

    candidates.push(PathBuf::from(".").join(lib_name));

    if let Some(prefix) = exe.and_then(install_prefix) {
        candidates.extend(LIB_SUBDIRS.iter().map(|d| prefix.join(d).join(lib_name)));
    }

    dedup_preserving_order(candidates)
}

/// [`library_candidates`] resolved against the current process, honouring `env_var` as the
/// override. Returns the first path that exists.
#[must_use]
pub fn find_library(lib_name: &OsStr, env_var: &str) -> Option<PathBuf> {
    let exe = std::env::current_exe().ok();
    let override_path: Option<OsString> = std::env::var_os(env_var);

    library_candidates(lib_name, override_path.as_deref(), exe.as_deref())
        .into_iter()
        .find(|p| p.exists())
}

/// Drop repeats while keeping the first occurrence, so precedence is preserved.
fn dedup_preserving_order(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen: Vec<PathBuf> = Vec::with_capacity(paths.len());
    for path in paths {
        if !seen.contains(&path) {
            seen.push(path);
        }
    }
    seen
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    fn os(s: &str) -> OsString {
        OsString::from(s)
    }

    #[test]
    fn install_prefix_requires_a_bin_dir() {
        assert_eq!(
            install_prefix(Path::new("/usr/bin/agentos-mail-suite")),
            Some(Path::new("/usr"))
        );
        assert_eq!(
            install_prefix(Path::new("/nix/store/abc-app-1.0/bin/app")),
            Some(Path::new("/nix/store/abc-app-1.0"))
        );
        // A cargo build tree has no sibling lib/ — guessing one would be wrong.
        assert_eq!(
            install_prefix(Path::new("/home/dev/target/debug/app")),
            None
        );
        assert_eq!(install_prefix(Path::new("app")), None);
    }

    /// Predicate over a fixed set of directories that "contain" a CEF distribution, so the
    /// search is tested against a known layout rather than whatever the host has installed.
    fn present(dirs: &'static [&'static str]) -> impl Fn(&Path) -> bool {
        move |path: &Path| dirs.iter().any(|d| Path::new(d) == path)
    }

    fn search(exe: &str) -> CefSearch {
        CefSearch {
            exe: Some(PathBuf::from(exe)),
            ..CefSearch::default()
        }
    }

    /// Fedora: both the prefix candidate and the XDG default land on the directory the
    /// `cef-libs` package installs into, so behaviour there is unchanged.
    #[test]
    fn fedora_resolves_to_the_usual_cef_dir() {
        let found = search("/usr/bin/app").find_with(&present(&["/usr/lib/cef"]));
        assert_eq!(found, Some(PathBuf::from("/usr/lib/cef")));
    }

    /// Nix, CEF in the application's own store path — resolved with no environment at all.
    #[test]
    fn nix_store_prefix_resolves_without_env() {
        let found = search("/nix/store/abc-app/bin/app")
            .find_with(&present(&["/nix/store/abc-app/lib/cef"]));
        assert_eq!(found, Some(PathBuf::from("/nix/store/abc-app/lib/cef")));
    }

    /// Nix, CEF in its OWN store path — only reachable via XDG_DATA_DIRS. This is the case
    /// the per-app copies disagreed about; two of the four never looked here.
    #[test]
    fn nix_cef_in_its_own_store_path_via_xdg_data_dirs() {
        let s = CefSearch {
            exe: Some(PathBuf::from("/nix/store/abc-app/bin/app")),
            xdg_data_dirs: Some(os("/nix/store/xyz-cef/share:/run/current-system/sw/share")),
            ..CefSearch::default()
        };
        assert_eq!(
            s.find_with(&present(&["/nix/store/xyz-cef/lib/cef"])),
            Some(PathBuf::from("/nix/store/xyz-cef/lib/cef"))
        );
    }

    /// A development run points at a build tree through LD_LIBRARY_PATH.
    #[test]
    fn ld_library_path_is_searched() {
        let s = CefSearch {
            exe: Some(PathBuf::from("/usr/bin/app")),
            ld_library_path: Some(os("/build/cef")),
            ..CefSearch::default()
        };
        assert_eq!(
            s.find_with(&present(&["/build/cef"])),
            Some(PathBuf::from("/build/cef"))
        );
    }

    #[test]
    fn explicit_cef_path_outranks_everything() {
        let s = CefSearch {
            exe: Some(PathBuf::from("/usr/bin/app")),
            cef_path: Some(os("/opt/my-cef")),
            ld_library_path: Some(os("/build/cef")),
            ..CefSearch::default()
        };
        assert_eq!(s.candidates().first(), Some(&PathBuf::from("/opt/my-cef")));
        assert_eq!(
            s.find_with(&present(&["/opt/my-cef", "/build/cef", "/usr/lib/cef"])),
            Some(PathBuf::from("/opt/my-cef"))
        );
    }

    /// A CEF_PATH pointing at nothing must not veto the search — failing to start the webview
    /// because of a stale override is worse than falling through to a working directory.
    #[test]
    fn stale_cef_path_falls_through() {
        let s = CefSearch {
            exe: Some(PathBuf::from("/usr/bin/app")),
            cef_path: Some(os("/gone/cef")),
            ..CefSearch::default()
        };
        assert_eq!(
            s.find_with(&present(&["/usr/lib/cef"])),
            Some(PathBuf::from("/usr/lib/cef"))
        );
    }

    #[test]
    fn empty_env_entries_are_ignored() {
        let s = CefSearch {
            cef_path: Some(os("")),
            ld_library_path: Some(os("/a::/b")),
            xdg_data_dirs: Some(os("")),
            ..CefSearch::default()
        };
        let dirs = s.candidates();
        assert!(!dirs.iter().any(|d| d.as_os_str().is_empty()));
        assert!(dirs.contains(&PathBuf::from("/a")));
        assert!(dirs.contains(&PathBuf::from("/b")));
        // An empty XDG_DATA_DIRS falls back to the spec default, which reaches /usr/lib/cef.
        assert!(dirs.contains(&PathBuf::from("/usr/lib/cef")));
    }

    /// The FHS dir is reachable three ways; it must be probed once and keep its position.
    #[test]
    fn duplicates_collapse_to_the_highest_priority_entry() {
        let s = CefSearch {
            exe: Some(PathBuf::from("/usr/bin/app")),
            cef_path: Some(os("/usr/lib/cef")),
            ..CefSearch::default()
        };
        let dirs = s.candidates();
        assert_eq!(
            dirs.iter()
                .filter(|d| *d == &PathBuf::from("/usr/lib/cef"))
                .count(),
            1
        );
        assert_eq!(dirs.first(), Some(&PathBuf::from("/usr/lib/cef")));
    }

    /// Nothing installed anywhere: the caller must be able to tell, so it can leave CEF_PATH
    /// alone and let a build-tree binary keep its OUT_DIR copy.
    #[test]
    fn nothing_found_is_none() {
        assert_eq!(search("/usr/bin/app").find_with(&present(&[])), None);
    }

    #[test]
    fn library_override_accepts_a_file_or_a_directory() {
        let name = OsString::from("libpdfium.so");

        let as_dir = os("/opt/pdfium");
        assert_eq!(
            library_candidates(&name, Some(&as_dir), None).first(),
            Some(&PathBuf::from("/opt/pdfium/libpdfium.so"))
        );

        let as_file = os("/opt/pdfium/libpdfium.so");
        assert_eq!(
            library_candidates(&name, Some(&as_file), None).first(),
            Some(&PathBuf::from("/opt/pdfium/libpdfium.so"))
        );
    }

    /// Fedora ships pdfium in /usr/lib64; Nix and Debian use lib. Both are probed.
    #[test]
    fn library_candidates_cover_both_lib_subdirs() {
        let name = OsString::from("libpdfium.so");
        let dirs = library_candidates(&name, None, Some(Path::new("/usr/bin/viewer")));
        assert_eq!(
            dirs,
            vec![
                PathBuf::from("./libpdfium.so"),
                PathBuf::from("/usr/lib/libpdfium.so"),
                PathBuf::from("/usr/lib64/libpdfium.so"),
            ]
        );
    }

    #[test]
    fn library_candidates_without_a_prefix_still_probe_the_cwd() {
        let name = OsString::from("libpdfium.so");
        let dirs = library_candidates(&name, None, Some(Path::new("/home/dev/target/debug/app")));
        assert_eq!(dirs, vec![PathBuf::from("./libpdfium.so")]);
    }
}
