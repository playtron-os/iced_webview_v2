//! CEF Dialog handler — native file picker for `<input type="file">`.
//!
//! We spawn `zenity` as a subprocess instead of linking a dialog library
//! (gtk3, xdg-portal) because both conflict with CEF's headless ozone
//! platform or its internal GTK usage.

use cef::*;

/// Read all strings from a borrowed `CefStringList` via raw FFI.
///
/// The safe wrapper's `From<&mut CefStringList> -> *mut` conversion
/// produces a `BorrowedMut` that gets freed on drop, causing a
/// use-after-free / double-free. Going through raw pointers directly
/// avoids that.
fn read_cef_string_list_raw(list: Option<&mut CefStringList>) -> Vec<String> {
    let Some(list) = list else {
        return Vec::new();
    };
    let raw: *mut cef::sys::_cef_string_list_t = list.into();
    if raw.is_null() {
        return Vec::new();
    }
    let count = unsafe { cef::sys::cef_string_list_size(raw) };
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let mut val: cef::sys::cef_string_t = unsafe { std::mem::zeroed() };
        let ok = unsafe { cef::sys::cef_string_list_value(raw, i, &mut val) };
        if ok > 0 {
            let s = CefString::from(std::ptr::from_ref(&val)).to_string();
            if !s.is_empty() {
                result.push(s);
            }
        }
    }
    result
}

fn run_zenity_file_dialog(
    is_save: bool,
    is_folder: bool,
    is_multi: bool,
    title: &str,
    default_path: &str,
    filters: &[(String, Vec<String>)],
) -> Option<Vec<std::path::PathBuf>> {
    let mut cmd = std::process::Command::new("zenity");
    cmd.arg("--file-selection");

    if is_save {
        cmd.arg("--save").arg("--confirm-overwrite");
    }
    if is_folder {
        cmd.arg("--directory");
    }
    if is_multi {
        cmd.arg("--multiple").arg("--separator=\n");
    }
    if !title.is_empty() {
        cmd.arg(format!("--title={title}"));
    }
    if !default_path.is_empty() {
        cmd.arg(format!("--filename={default_path}"));
    }
    for (desc, exts) in filters {
        if exts.is_empty() {
            continue;
        }
        let pattern: String = exts
            .iter()
            .map(|e| {
                let e = e.strip_prefix('.').unwrap_or(e);
                format!("*.{e}")
            })
            .collect::<Vec<_>>()
            .join(" ");
        let label = if desc.is_empty() { "Accepted" } else { desc };
        cmd.arg(format!("--file-filter={label} | {pattern}"));
    }
    if !filters.is_empty() {
        cmd.arg("--file-filter=All files | *");
    }

    let output = cmd.output().ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let paths: Vec<std::path::PathBuf> = stdout
        .lines()
        .filter(|l| !l.is_empty())
        .map(std::path::PathBuf::from)
        .collect();

    if paths.is_empty() {
        None
    } else {
        Some(paths)
    }
}

wrap_dialog_handler! {
    pub(super) struct OsrDialogHandler;

    impl DialogHandler {
        fn on_file_dialog(
            &self,
            _browser: Option<&mut Browser>,
            mode: FileDialogMode,
            title: Option<&CefString>,
            default_file_path: Option<&CefString>,
            _accept_filters: Option<&mut CefStringList>,
            accept_extensions: Option<&mut CefStringList>,
            accept_descriptions: Option<&mut CefStringList>,
            callback: Option<&mut FileDialogCallback>,
        ) -> ::std::os::raw::c_int {
            let Some(cb) = callback else {
                return 0;
            };

            let title_str = title
                .map(|s| s.to_string())
                .unwrap_or_default();

            let default_path = default_file_path
                .map(|s| s.to_string())
                .unwrap_or_default();

            let raw_mode = mode.get_raw();
            let is_save = raw_mode == FileDialogMode::SAVE.get_raw();
            let is_folder = raw_mode == FileDialogMode::OPEN_FOLDER.get_raw();
            let is_multi = raw_mode == FileDialogMode::OPEN_MULTIPLE.get_raw();

            // Read extension and description lists via raw FFI (the safe
            // wrapper's `into()` conversion causes use-after-free).
            let ext_entries = read_cef_string_list_raw(accept_extensions);
            let desc_entries = read_cef_string_list_raw(accept_descriptions);

            let filters: Vec<(String, Vec<String>)> = ext_entries
                .iter()
                .enumerate()
                .map(|(i, exts_str)| {
                    let desc = desc_entries.get(i).cloned().unwrap_or_default();
                    let exts: Vec<String> = exts_str
                        .split(';')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                    (desc, exts)
                })
                .filter(|(_, exts)| !exts.is_empty())
                .collect();

            let result = run_zenity_file_dialog(
                is_save, is_folder, is_multi,
                &title_str, &default_path, &filters,
            );

            match result {
                Some(paths) if !paths.is_empty() => {
                    let mut list = string_list_alloc().expect("failed to allocate CefStringList");
                    for path in &paths {
                        let s = CefString::from(path.to_string_lossy().as_ref());
                        string_list_append(Some(&mut list), Some(&s));
                    }
                    cb.cont(Some(&mut list));
                    string_list_free(Some(&mut list));
                }
                _ => {
                    cb.cancel();
                }
            }

            1 // We handled the dialog
        }
    }
}
