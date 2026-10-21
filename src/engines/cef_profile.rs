//! Who owns a CEF profile directory, and clearing what a dead browser left in it.
//!
//! Chromium marks a profile in use with `SingletonLock`, a symlink to
//! `hostname-pid`. It clears a lock whose process on this host is gone, but a
//! lock naming another host stops it cold: it asks the user whether that other
//! computer is still using the profile, an embedded browser cannot show the
//! question, and `initialize()` fails, leaving every view blank. A lock names
//! another host whenever the machine was renamed after a run that did not shut
//! down cleanly. Inside a PID namespace the pid is no better a witness: the app
//! gets the same small pid every run.
//!
//! So every instance also holds a file lock of its own in the profile, which
//! the kernel drops however the process ends. Whoever gets it knows no other
//! instance is running there, and can clear Chromium's files.

use std::fs::{File, OpenOptions};
use std::path::Path;

/// The file a running instance keeps locked inside its profile.
const PROFILE_LOCK: &str = "iced_webview.lock";

/// Chromium's singleton files, all left behind together.
const SINGLETON_FILES: [&str; 3] = ["SingletonLock", "SingletonSocket", "SingletonCookie"];

/// Take the profile at `dir` for as long as the returned file is open, or
/// `None` when a running instance already holds it.
fn claim(dir: &Path) -> Option<File> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(dir.join(PROFILE_LOCK))
        .ok()?;
    file.try_lock().ok()?;
    Some(file)
}

/// Whether the `SingletonLock` pointing at `target` belongs to no running
/// browser, given that this process (`own_pid` on `host`) holds the profile's
/// [`PROFILE_LOCK`].
///
/// One exception keeps the lock: a live process on this host other than us,
/// which could be an older build that predates [`PROFILE_LOCK`]. Chromium then
/// decides for itself, as it always has.
fn singleton_is_stale(target: &str, host: &str, own_pid: u32, alive: impl Fn(u32) -> bool) -> bool {
    // `hostname-pid`, and a hostname may itself contain '-'.
    let Some((lock_host, pid)) = target.rsplit_once('-') else {
        return true;
    };
    let Ok(pid) = pid.parse::<u32>() else {
        return true;
    };
    !(lock_host == host && pid != own_pid && alive(pid))
}

/// Claim the profile at `dir` for the rest of this process and clear the
/// singleton files a dead browser left there.
///
/// When another instance holds the profile, nothing is touched.
pub(crate) fn claim_and_clear(dir: &Path) {
    let Some(file) = claim(dir) else {
        log::debug!(
            "iced_webview: profile {} is held by a running instance, leaving it alone",
            dir.display()
        );
        return;
    };
    // Held until the process exits; the kernel releases it then.
    std::mem::forget(file);

    if let Ok(target) = std::fs::read_link(dir.join("SingletonLock")) {
        let target = target.to_string_lossy();
        let host = host_name();
        let alive = |pid: u32| Path::new(&format!("/proc/{pid}")).exists();
        if !singleton_is_stale(&target, &host, std::process::id(), alive) {
            log::debug!(
                "iced_webview: profile {} is locked by live pid {target}, leaving it to Chromium",
                dir.display()
            );
            return;
        }
        log::info!(
            "iced_webview: clearing the profile lock {target} left in {} (this host is {host})",
            dir.display()
        );
    }
    for name in SINGLETON_FILES {
        let _ = std::fs::remove_file(dir.join(name));
    }
}

/// This host's name as Chromium reads it, for comparing with a lock's.
fn host_name() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|name| name.trim_end().to_owned())
        .unwrap_or_default()
}
