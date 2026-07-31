use std::cell::RefCell;
use std::os::raw::c_int;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use iced::keyboard;
use iced::mouse::{self, Interaction};
use iced::{Point, Size};
use rand::Rng;

use super::{ConsoleMessage, Engine, PageType, PixelFormat, ViewId};
use crate::ImageInfo;

// Pull in all CEF types, traits, and macros. The wrap_*! macros reference
// ImplClient, WrapClient, Client, etc. by unqualified name, so a glob
// import is the simplest way to satisfy them.
use cef::args::Args;
use cef::*;

/// Shared mutable state populated by CEF handler callbacks and drained
/// each `update()` tick.
struct SharedState {
    /// New frame ready for consumption by `update()`. The `Arc` is shared
    /// with the shader pipeline so no copy is needed.
    frame_buffer: Option<(Arc<Vec<u8>>, u32, u32)>,
    /// Persistent pixel buffer reused across on_paint calls. Dirty rects
    /// are blitted into it via `Arc::make_mut` (copy-on-write: only copies
    /// if the shader still holds a reference to the previous frame).
    persistent_buffer: Arc<Vec<u8>>,
    persistent_size: (u32, u32),
    url: Option<String>,
    popup_url: Option<String>,
    title: Option<String>,
    cursor_type: CursorType,
    size: Size<u32>,
    scale_factor: f32,
    /// Set to `true` by the load handler when a page finishes loading.
    page_loaded: bool,
    /// Console messages emitted by the page, drained by the webview layer.
    console_messages: Vec<ConsoleMessage>,
    /// When `true`, the request handler cancels top-level navigations away
    /// from the loaded document and surfaces them via `popup_url` instead.
    /// Opt-in; off by default so the view behaves like a normal browser.
    block_navigation: bool,
    /// `User-Agent` to send, or `None` for the engine default. Applied as a
    /// real request header on every request (see `OsrResourceRequestHandler`).
    user_agent: Option<String>,
}

// -- CEF App handler --

wrap_app! {
    struct OsrApp;

    impl App {
        fn on_before_command_line_processing(
            &self,
            process_type: Option<&CefString>,
            command_line: Option<&mut CommandLine>,
        ) {
            let Some(cmd) = command_line else { return };

            // Forcing `ozone-platform=headless` + `disable-gpu` +
            // `in-process-gpu` dodges a GLX BadMatch against the host wgpu
            // context, but drops WebGL onto SwiftShader. Set
            // `ICED_WEBVIEW_DISABLE_GPU=1` to restore those if a host hits the
            // conflict.
            let disable_gpu = std::env::var("ICED_WEBVIEW_DISABLE_GPU")
                .is_ok_and(|v| matches!(v.trim(), "1" | "true" | "yes"));

            if disable_gpu {
                cmd.append_switch_with_value(
                    Some(&CefString::from("ozone-platform")),
                    Some(&CefString::from("headless")),
                );
                cmd.append_switch(Some(&CefString::from("disable-gpu")));
                cmd.append_switch(Some(&CefString::from("in-process-gpu")));
            } else {
                cmd.append_switch_with_value(
                    Some(&CefString::from("ozone-platform-hint")),
                    Some(&CefString::from("x11")),
                );
            }

            // Don't probe the system keyring (gnome-keyring / kwallet) over
            // D-Bus during browser-process startup.
            cmd.append_switch_with_value(
                Some(&CefString::from("password-store")),
                Some(&CefString::from("basic")),
            );

            // The rest are browser-process only.
            let is_browser_process = process_type.is_none_or(|t| t.to_string().is_empty());
            if !is_browser_process {
                return;
            }

            // Keeps the OSR texture from glitching on resize — this is
            // what actually fixes the context conflict.
            cmd.append_switch(Some(&CefString::from("disable-gpu-compositing")));
            cmd.append_switch(Some(&CefString::from("disable-gpu-shader-disk-cache")));
            cmd.append_switch(Some(&CefString::from("enable-accelerated-2d-canvas")));
            cmd.append_switch_with_value(
                Some(&CefString::from("enable-features")),
                Some(&CefString::from(
                    "CanvasOopRasterization,UserAgentClientHints,GreaseUACH",
                )),
            );
        }
    }
}

// -- CEF handler implementations via wrap macros --

wrap_render_handler! {
    struct OsrRenderHandler {
        shared: Rc<RefCell<SharedState>>,
    }

    impl RenderHandler {
        fn view_rect(&self, _browser: Option<&mut Browser>, rect: Option<&mut Rect>) {
            if let Some(rect) = rect {
                let shared = self.shared.borrow();
                rect.x = 0;
                rect.y = 0;
                rect.width = shared.size.width as c_int;
                rect.height = shared.size.height as c_int;
            }
        }

        fn screen_info(
            &self,
            _browser: Option<&mut Browser>,
            screen_info: Option<&mut ScreenInfo>,
        ) -> c_int {
            if let Some(info) = screen_info {
                let shared = self.shared.borrow();
                info.device_scale_factor = shared.scale_factor;
                return 1;
            }
            0
        }

        fn on_paint(
            &self,
            _browser: Option<&mut Browser>,
            _type_: PaintElementType,
            dirty_rects: Option<&[Rect]>,
            buffer: *const u8,
            width: c_int,
            height: c_int,
        ) {
            let w = width as u32;
            let h = height as u32;
            let stride = (w as usize) * 4;
            let total = stride * h as usize;
            let src = unsafe { std::slice::from_raw_parts(buffer, total) };

            let mut shared = self.shared.borrow_mut();

            // Resize persistent buffer when dimensions change.
            if shared.persistent_size != (w, h) {
                shared.persistent_buffer = Arc::new(src.to_vec());
                shared.persistent_size = (w, h);
            } else if let Some(rects) = dirty_rects {
                // Copy only dirty regions. `Arc::make_mut` gives us
                // exclusive write access — if the shader still holds a
                // reference to the previous frame, this triggers a CoW
                // copy; otherwise we mutate in-place with zero allocation.
                let dst = Arc::make_mut(&mut shared.persistent_buffer);
                for rect in rects {
                    let rx = (rect.x as usize).min(w as usize);
                    let ry = (rect.y as usize).min(h as usize);
                    let rw = (rect.width as usize).min(w as usize - rx);
                    let rh = (rect.height as usize).min(h as usize - ry);
                    for row in 0..rh {
                        let y = ry + row;
                        let offset = y * stride + rx * 4;
                        let len = rw * 4;
                        dst[offset..offset + len].copy_from_slice(&src[offset..offset + len]);
                    }
                }
            } else {
                // No dirty rects — full copy fallback.
                Arc::make_mut(&mut shared.persistent_buffer).copy_from_slice(src);
            }

            shared.frame_buffer = Some((Arc::clone(&shared.persistent_buffer), w, h));
        }
    }
}

wrap_display_handler! {
    struct OsrDisplayHandler {
        shared: Rc<RefCell<SharedState>>,
    }

    impl DisplayHandler {
        fn on_address_change(
            &self,
            _browser: Option<&mut Browser>,
            _frame: Option<&mut Frame>,
            url: Option<&CefString>,
        ) {
            if let Some(url) = url {
                self.shared.borrow_mut().url = Some(url.to_string());
            }
        }

        fn on_title_change(
            &self,
            _browser: Option<&mut Browser>,
            title: Option<&CefString>,
        ) {
            if let Some(title) = title {
                self.shared.borrow_mut().title = Some(title.to_string());
            }
        }

        // Capture page `console.log`/`warn`/`error` output into the shared
        // queue. The webview layer drains it each tick and hands each message
        // to the consumer's `on_console_message` callback.
        fn on_console_message(
            &self,
            _browser: Option<&mut Browser>,
            level: LogSeverity,
            message: Option<&CefString>,
            source: Option<&CefString>,
            line: c_int,
        ) -> c_int {
            self.shared.borrow_mut().console_messages.push(ConsoleMessage {
                level: level.get_raw() as i32,
                message: message.map(|m| m.to_string()).unwrap_or_default(),
                source: source.map(|s| s.to_string()).unwrap_or_default(),
                line: line as i32,
            });
            0 // allow default handling
        }

        fn on_cursor_change(
            &self,
            _browser: Option<&mut Browser>,
            _cursor: std::os::raw::c_ulong,
            type_: CursorType,
            _custom_cursor_info: Option<&CursorInfo>,
        ) -> c_int {
            self.shared.borrow_mut().cursor_type = type_;
            0
        }
    }
}

wrap_life_span_handler! {
    struct OsrLifeSpanHandler {
        shared: Rc<RefCell<SharedState>>,
    }

    impl LifeSpanHandler {
        fn on_after_created(&self, _browser: Option<&mut Browser>) {}
        fn on_before_close(&self, _browser: Option<&mut Browser>) {}
        fn on_before_popup(
            &self,
            browser: Option<&mut Browser>,
            _frame: Option<&mut Frame>,
            _popup_id: c_int,
            target_url: Option<&CefString>,
            _target_frame_name: Option<&CefString>,
            _target_disposition: WindowOpenDisposition,
            _user_gesture: c_int,
            _popup_features: Option<&PopupFeatures>,
            _window_info: Option<&mut WindowInfo>,
            _client: Option<&mut Option<Client>>,
            _settings: Option<&mut BrowserSettings>,
            _extra_info: Option<&mut Option<DictionaryValue>>,
            _no_javascript_access: Option<&mut c_int>,
        ) -> c_int {
            let Some(url) = target_url else {
                return 1; // nothing to open
            };

            // Single-document viewers opt out of navigating at all: hand the
            // URL to the host so it can open it externally.
            if self.shared.borrow().block_navigation {
                self.shared.borrow_mut().popup_url = Some(url.to_string());
                return 1;
            }

            // Otherwise load the target here rather than dropping it: a
            // cancelled popup with no follow-up makes every `window.open` /
            // `target="_blank"` control a dead button. Opener semantics
            // (`window.close()`, `postMessage`) are not preserved.
            //
            // Deliberately not also setting `popup_url` — that is the host's
            // "open this elsewhere" channel, and doing both loads the page
            // twice.
            match browser.and_then(|b| b.main_frame()) {
                Some(frame) => {
                    log::debug!("iced_webview: popup navigated in place: {url}");
                    frame.load_url(Some(url));
                }
                // No frame to navigate — fall back to handing it to the host
                // rather than silently dropping it.
                None => self.shared.borrow_mut().popup_url = Some(url.to_string()),
            }
            1 // the popup itself is cancelled; we handled the URL
        }
    }
}

wrap_load_handler! {
    struct OsrLoadHandler {
        shared: Rc<RefCell<SharedState>>,
    }

    impl LoadHandler {
        fn on_loading_state_change(
            &self,
            _browser: Option<&mut Browser>,
            is_loading: c_int,
            _can_go_back: c_int,
            _can_go_forward: c_int,
        ) {
            if is_loading == 0 {
                self.shared.borrow_mut().page_loaded = true;
            }
        }

        // Off-screen rendering has no window manager handing focus back, so
        // without this a page reached by redirect drops keystrokes until the
        // user clicks inside it.
        fn on_load_end(
            &self,
            browser: Option<&mut Browser>,
            _frame: Option<&mut Frame>,
            _http_status_code: c_int,
        ) {
            if let Some(host) = browser.and_then(|b| b.host()) {
                host.set_focus(1);
            }
        }

        // Render the error inline; otherwise a failed load is a blank
        // rectangle with nothing to retry.
        fn on_load_error(
            &self,
            _browser: Option<&mut Browser>,
            frame: Option<&mut Frame>,
            error_code: Errorcode,
            error_text: Option<&CefString>,
            failed_url: Option<&CefString>,
        ) {
            let text = error_text.map(ToString::to_string).unwrap_or_default();
            let url = failed_url.map(ToString::to_string).unwrap_or_default();
            log::warn!("iced_webview: load failed ({error_code:?}) for {url}: {text}");

            let Some(frame) = frame else { return };
            let body = format!(
                "<html><body style=\"font-family:sans-serif;padding:2rem\">\
                 <h2>Failed to load</h2><p>{}</p><p style=\"color:#666\">{}</p></body></html>",
                html_escape(&url),
                html_escape(&text),
            );
            let data_url = format!(
                "data:text/html;charset=utf-8,{}",
                urlencoding::encode(&body)
            );
            frame.load_url(Some(&CefString::from(data_url.as_str())));
        }
    }
}

/// Minimal HTML escaping for text interpolated into the error page.
fn html_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

wrap_request_handler! {
    struct OsrRequestHandler {
        shared: Rc<RefCell<SharedState>>,
    }

    impl RequestHandler {
        // Optionally prevent the view from navigating away from its loaded
        // document. This is opt-in per view (see `Engine::set_block_navigation`)
        // and is meant for single-document viewers such as an email renderer:
        // the page is loaded as a `data:` URL, and any other top-level
        // navigation (a clicked link) is cancelled here and routed to the host
        // via `popup_url`, which the consumer typically opens in the external
        // browser — matching how `on_before_popup` handles `target="_blank"`.
        //
        // When the flag is unset (the default) this is a no-op and the view
        // navigates like a normal browser, so other consumers are unaffected.
        fn resource_request_handler(
            &self,
            _browser: Option<&mut Browser>,
            _frame: Option<&mut Frame>,
            _request: Option<&mut Request>,
            _is_navigation: c_int,
            _is_download: c_int,
            _request_initiator: Option<&CefString>,
            _disable_default_handling: Option<&mut c_int>,
        ) -> Option<ResourceRequestHandler> {
            Some(OsrResourceRequestHandler::new(Rc::clone(&self.shared)))
        }

        fn on_before_browse(
            &self,
            _browser: Option<&mut Browser>,
            frame: Option<&mut Frame>,
            request: Option<&mut Request>,
            _user_gesture: c_int,
            _is_redirect: c_int,
        ) -> c_int {
            if !self.shared.borrow().block_navigation {
                return 0; // navigation allowed (default browser behavior)
            }

            // Only constrain the main frame. Sub-frames (e.g. an iframe
            // embedded in the document) are allowed to load their own content.
            let is_main = frame.map(|f| f.is_main() != 0).unwrap_or(true);
            if !is_main {
                return 0; // allow sub-frame navigation
            }

            let Some(request) = request else {
                return 0;
            };
            let url = CefString::from(&request.url()).to_string();

            // The document itself is a `data:` URL — allow it (and in-page
            // anchor jumps, which keep the same `data:` prefix). `about:blank`
            // is the browser's initial placeholder document.
            if url.is_empty() || url.starts_with("data:") || url == "about:blank" {
                return 0; // allow
            }

            // A real navigation (link click). Cancel it and hand the URL to
            // the host to open externally.
            self.shared.borrow_mut().popup_url = Some(url);
            1 // non-zero cancels the navigation
        }
    }
}

wrap_resource_request_handler! {
    struct OsrResourceRequestHandler {
        shared: Rc<RefCell<SharedState>>,
    }

    impl ResourceRequestHandler {
        // Set the User-Agent as a genuine request header, on every request.
        //
        // DevTools `Emulation.setUserAgentOverride` also works, but drives
        // Chromium's *emulation* path: it leaves `navigator.userAgentData` /
        // Sec-CH-UA reporting the real browser while the UA string claims
        // otherwise, and it needs the DevTools protocol attached.
        fn on_before_resource_load(
            &self,
            _browser: Option<&mut Browser>,
            _frame: Option<&mut Frame>,
            request: Option<&mut Request>,
            _callback: Option<&mut Callback>,
        ) -> ReturnValue {
            if let (Some(request), Some(ua)) =
                (request, self.shared.borrow().user_agent.as_deref())
            {
                let name = CefString::from("User-Agent");
                let value = CefString::from(ua);
                // `overwrite = 1`: replace Chromium's own header rather than
                // appending a second one.
                request.set_header_by_name(Some(&name), Some(&value), 1);
            }
            ReturnValue::CONTINUE
        }
    }
}

use super::cef_dialog::OsrDialogHandler;

wrap_client! {
    struct OsrClient {
        render_handler: RenderHandler,
        display_handler: DisplayHandler,
        life_span_handler: LifeSpanHandler,
        load_handler: LoadHandler,
        request_handler: RequestHandler,
        dialog_handler: DialogHandler,
    }

    impl Client {
        fn render_handler(&self) -> Option<RenderHandler> {
            Some(self.render_handler.clone())
        }

        fn display_handler(&self) -> Option<DisplayHandler> {
            Some(self.display_handler.clone())
        }

        fn life_span_handler(&self) -> Option<LifeSpanHandler> {
            Some(self.life_span_handler.clone())
        }

        fn load_handler(&self) -> Option<LoadHandler> {
            Some(self.load_handler.clone())
        }

        fn request_handler(&self) -> Option<RequestHandler> {
            Some(self.request_handler.clone())
        }

        fn dialog_handler(&self) -> Option<DialogHandler> {
            Some(self.dialog_handler.clone())
        }
    }
}

struct CefView {
    id: ViewId,
    browser: Browser,
    shared: Rc<RefCell<SharedState>>,
    url: String,
    title: String,
    cursor: Interaction,
    last_frame: ImageInfo,
    needs_render: bool,
    size: Size<u32>,
    /// Last click position + timestamp for multi-click detection.
    last_click: Option<(Point, std::time::Instant)>,
    click_count: c_int,
    /// CEF event-flag bitmask for currently held mouse buttons.
    pressed_buttons: u32,
}

/// Minimal state kept for a suspended view so it can be resumed later
/// without tearing down the entire CEF engine.
struct ParkedView {
    id: ViewId,
    last_frame: ImageInfo,
}

/// Full browser engine backed by [CEF/Chromium](https://github.com/tauri-apps/cef-rs)
/// (HTML5, CSS3, JS).
///
/// CEF handles its own networking, scrolling, and JavaScript execution.
/// Rendering is off-screen (windowless), producing BGRA pixel buffers that
/// are uploaded to a persistent GPU texture via iced's shader widget.
///
/// ## Subprocess requirement
///
/// CEF uses multi-process mode — helper sub-processes (renderer, GPU,
/// utility) are spawned from the same binary. Call [`cef_subprocess_check`]
/// at the very top of `main()` — if it returns `true`, the process is a
/// CEF subprocess and should exit immediately.
///
/// On non-FHS systems (Guix, Nix), run inside an FHS-emulated container
/// so subprocesses can discover `.pak` resources, `icudtl.dat`, and shared
/// libraries at standard paths.
///
/// ```rust,ignore
/// fn main() -> iced::Result {
///     if iced_webview::cef_subprocess_check() {
///         return Ok(());
///     }
///     // ... iced application setup ...
/// }
/// ```
/// Global flag: CEF has been successfully initialized in this process.
/// CEF does not support being initialized more than once per process,
/// nor re-initialization after `shutdown()`. This flag ensures we call
/// `initialize()` exactly once and never call `shutdown()`.
static CEF_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Stores the initialization error from the first (and only) attempt.
static CEF_INIT_ERROR: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();

/// Run CEF global initialization exactly once. Returns `(success, error)`.
fn ensure_cef_initialized() -> (bool, Option<&'static str>) {
    let error = CEF_INIT_ERROR.get_or_init(|| {
        let _ = api_hash(cef::sys::CEF_API_VERSION_LAST, 0);
        let args = Args::new();

        let cache_dir = std::env::var("XDG_CACHE_HOME").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            format!("{home}/.cache")
        });
        let cef_cache = format!("{cache_dir}/iced_webview_cef");
        let _ = std::fs::create_dir_all(&cef_cache);

        // Remove stale singleton lock files from previous crashed runs.
        for name in ["SingletonLock", "SingletonSocket", "SingletonCookie"] {
            let lock = std::path::Path::new(&cef_cache).join(name);
            let _ = std::fs::remove_file(&lock);
        }

        let cef_dir = match cef::sys::get_cef_dir() {
            Some(dir) => dir,
            None => {
                log::error!(
                    "iced_webview: CEF distribution directory not found. \
                     Webview will be unavailable."
                );
                return Some("CEF distribution directory not found".to_string());
            }
        };
        let cef_dir_str = cef_dir.to_string_lossy();

        let locales_dir = cef_dir.join("locales");
        let locales_str = locales_dir.to_string_lossy();

        let settings = Settings {
            windowless_rendering_enabled: 1,
            external_message_pump: 1,
            no_sandbox: 1,
            root_cache_path: CefString::from(cef_cache.as_str()),
            framework_dir_path: CefString::from(cef_dir_str.as_ref()),
            resources_dir_path: CefString::from(cef_dir_str.as_ref()),
            locales_dir_path: CefString::from(locales_str.as_ref()),
            ..Default::default()
        };

        let mut app = OsrApp::new();

        let result = initialize(
            Some(args.as_main_args()),
            Some(&settings),
            Some(&mut app),
            std::ptr::null_mut(),
        );

        if result != 1 {
            let msg = format!(
                "CEF initialize() returned {result} (expected 1). \
                 cef_dir={cef_dir_str}, cache={cef_cache}"
            );
            log::error!("iced_webview: {msg}");
            return Some(msg);
        }

        // CEF's initialize() installs its own signal handlers that swallow
        // SIGINT — restore the default so a single Ctrl+C terminates the app.
        unsafe {
            libc::signal(libc::SIGINT, libc::SIG_DFL);
            libc::signal(libc::SIGTERM, libc::SIG_DFL);
        }

        CEF_INITIALIZED.store(true, Ordering::Release);
        log::info!("iced_webview: CEF initialized successfully");
        None // no error
    });

    let ok = CEF_INITIALIZED.load(Ordering::Acquire);
    (ok, error.as_deref())
}

pub struct Cef {
    views: Vec<CefView>,
    parked_views: Vec<ParkedView>,
    scale_factor: f32,
    initialized: bool,
    init_error: Option<String>,
    /// Default background color for new browser views (ARGB u32, CEF format).
    /// If `None`, CEF uses its default (opaque white with alpha=0 → black on
    /// windowless). Set via `Engine::set_background_color`.
    background_color: Option<u32>,
    /// When `true`, new views block top-level navigation away from their
    /// loaded document (see `set_block_navigation`). Off by default.
    block_navigation: bool,
    /// `User-Agent` to present, or `None` for the CEF default.
    /// Applied per view (see `Engine::set_user_agent`).
    user_agent: Option<String>,
}

impl Default for Cef {
    fn default() -> Self {
        // NB: do NOT initialize CEF here. The engine struct may be constructed on
        // a different thread (iced's `boot()` on the main thread) than the one
        // that pumps CEF (`update()` on the render worker thread under
        // `off-thread-render`). Init is deferred to first use via
        // `ensure_initialized()` so it lands on the pump thread.
        Self {
            views: Vec::new(),
            parked_views: Vec::new(),
            scale_factor: 1.0,
            initialized: false,
            init_error: None,
            background_color: None,
            block_navigation: false,
            user_agent: None,
        }
    }
}

/// Ensure the CEF distribution directory is on `LD_LIBRARY_PATH` so that
/// subprocesses (GPU, renderer, utility) can find `libEGL.so`,
/// `libGLESv2.so`, and other CEF shared libraries at runtime.
fn ensure_cef_lib_path() {
    if let Some(cef_dir) = cef::sys::get_cef_dir() {
        let cef_dir_str = cef_dir.to_string_lossy().to_string();
        let ld_path = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
        if !ld_path.contains(&cef_dir_str) {
            let new_path = if ld_path.is_empty() {
                cef_dir_str
            } else {
                format!("{cef_dir_str}:{ld_path}")
            };
            unsafe { std::env::set_var("LD_LIBRARY_PATH", new_path) };
        }
    }
}

/// Check if the current process is a CEF subprocess.
///
/// Must be called at the very top of `main()`. Returns `true` if this
/// process is a CEF helper (renderer, GPU, utility) — in that case,
/// exit immediately without starting the iced application.
pub fn cef_subprocess_check() -> bool {
    ensure_cef_lib_path();
    let _ = api_hash(cef::sys::CEF_API_VERSION_LAST, 0);
    let args = Args::new();

    let cmd_line = args.as_cmd_line();
    let is_browser = if let Some(cmd) = &cmd_line {
        let switch = CefString::from("type");
        cmd.has_switch(Some(&switch)) != 1
    } else {
        true
    };

    // Browser process — no subprocess work needed. Return immediately
    // without calling execute_process(), which would set up CEF global
    // state that interferes with the later initialize() call.
    if is_browser {
        return false;
    }

    let mut app = OsrApp::new();
    let ret = execute_process(
        Some(args.as_main_args()),
        Some(&mut app),
        std::ptr::null_mut(),
    );

    ret >= 0
}

impl Cef {
    /// Lazily run CEF global initialization on the *current* thread — the one
    /// that pumps the message loop (`update`) and makes all browser calls.
    ///
    /// CEF requires `CefInitialize`, `CefDoMessageLoopWork` and every browser API
    /// call to happen on a single, consistent thread. With iced's
    /// `off-thread-render`, the engine struct is constructed during `boot()` on
    /// the main thread but `update()`/`new_view()` run on the render worker
    /// thread — so initializing in `Default` would bind CEF to the wrong thread.
    /// Deferring init to first use keeps it on the pump thread. Idempotent and
    /// cheap after the first call (the global init is `OnceLock`-guarded).
    fn ensure_initialized(&mut self) {
        if self.initialized || self.init_error.is_some() {
            return;
        }
        let (initialized, init_error) = ensure_cef_initialized();
        self.initialized = initialized;
        self.init_error = init_error.map(|s| s.to_string());
    }

    fn find_view(&self, id: ViewId) -> Option<&CefView> {
        self.views.iter().find(|v| v.id == id)
    }

    fn find_view_mut(&mut self, id: ViewId) -> Option<&mut CefView> {
        self.views.iter_mut().find(|v| v.id == id)
    }

    /// Create a browser and its CefView, returning None if CEF isn't
    /// initialized or browser creation fails. Optionally reuses a previous
    /// frame so the view doesn't flash to blank on resume.
    fn create_browser_view(
        &self,
        id: ViewId,
        size: Size<u32>,
        last_frame: Option<ImageInfo>,
    ) -> Option<CefView> {
        if !self.initialized {
            return None;
        }

        let w = size.width.max(1);
        let h = size.height.max(1);
        let size = Size::new(w, h);

        let shared = Rc::new(RefCell::new(SharedState {
            frame_buffer: None,
            persistent_buffer: Arc::new(Vec::new()),
            persistent_size: (0, 0),
            url: None,
            popup_url: None,
            title: None,
            cursor_type: CursorType::POINTER,
            size,
            scale_factor: self.scale_factor,
            page_loaded: false,
            console_messages: Vec::new(),
            block_navigation: self.block_navigation,
            user_agent: self.user_agent.clone(),
        }));

        let render_handler = OsrRenderHandler::new(Rc::clone(&shared));
        let display_handler = OsrDisplayHandler::new(Rc::clone(&shared));
        let life_span_handler = OsrLifeSpanHandler::new(Rc::clone(&shared));
        let load_handler = OsrLoadHandler::new(Rc::clone(&shared));
        let request_handler = OsrRequestHandler::new(Rc::clone(&shared));
        let dialog_handler = OsrDialogHandler::new();
        let mut client = OsrClient::new(
            render_handler,
            display_handler,
            life_span_handler,
            load_handler,
            request_handler,
            dialog_handler,
        );

        let window_info = WindowInfo::default().set_as_windowless(0);
        let browser_settings = BrowserSettings {
            windowless_frame_rate: 60,
            background_color: self.background_color.unwrap_or(0xFFFFFFFF),
            ..Default::default()
        };

        let initial_url = CefString::from("about:blank");
        let browser = browser_host_create_browser_sync(
            Some(&window_info),
            Some(&mut client),
            Some(&initial_url),
            Some(&browser_settings),
            None,
            None,
        )?;

        // Give the new browser host focus immediately. Off-screen rendering has
        // no window manager to do it, so without this the renderer discards
        // every key event until the user physically clicks inside the view —
        // i.e. a portal you are supposed to type an email into looks dead.
        if let Some(host) = browser.host() {
            host.set_focus(1);
        }

        Some(CefView {
            id,
            browser,
            shared,
            url: String::new(),
            title: String::new(),
            cursor: Interaction::Idle,
            last_frame: last_frame.unwrap_or_else(|| ImageInfo::blank(w, h)),
            needs_render: true,
            size,
            last_click: None,
            click_count: 0,
            pressed_buttons: 0,
        })
    }
}

fn cursor_type_to_interaction(cursor: CursorType) -> Interaction {
    match cursor {
        CursorType::POINTER => Interaction::Idle,
        CursorType::IBEAM => Interaction::Text,
        CursorType::CROSS => Interaction::Crosshair,
        CursorType::HAND => Interaction::Pointer,
        CursorType::GRAB => Interaction::Grab,
        CursorType::GRABBING => Interaction::Grabbing,
        CursorType::NOTALLOWED => Interaction::NotAllowed,
        CursorType::EASTWESTRESIZE
        | CursorType::EASTRESIZE
        | CursorType::WESTRESIZE
        | CursorType::COLUMNRESIZE => Interaction::ResizingHorizontally,
        CursorType::NORTHSOUTHRESIZE
        | CursorType::NORTHRESIZE
        | CursorType::SOUTHRESIZE
        | CursorType::ROWRESIZE => Interaction::ResizingVertically,
        CursorType::ZOOMIN => Interaction::ZoomIn,
        CursorType::ZOOMOUT => Interaction::ZoomOut,
        _ => Interaction::Idle,
    }
}

impl Engine for Cef {
    fn handles_urls(&self) -> bool {
        true
    }

    fn update(&mut self) {
        // First use pumps init onto the pump thread (see `ensure_initialized`).
        self.ensure_initialized();
        if !self.initialized {
            return;
        }

        do_message_loop_work();

        for view in &mut self.views {
            let mut shared = view.shared.borrow_mut();

            if let Some((pixels, w, h)) = shared.frame_buffer.take() {
                let t0 = std::time::Instant::now();
                view.last_frame = ImageInfo::from_arc(pixels, PixelFormat::Bgra, w, h);
                view.needs_render = false;
                let elapsed = t0.elapsed();
                if elapsed.as_millis() > 2 {
                    eprintln!(
                        "[cef] slow frame {}×{} took {}ms",
                        w,
                        h,
                        elapsed.as_millis()
                    );
                }
            }
            if let Some(url) = shared.url.take() {
                view.url = url;
            }
            if let Some(title) = shared.title.take() {
                view.title = title;
            }
            view.cursor = cursor_type_to_interaction(shared.cursor_type);
        }
    }

    fn render(&mut self, _size: Size<u32>) {
        // CEF renders asynchronously via on_paint — nothing to do here.
    }

    fn request_render(&mut self, _id: ViewId, _size: Size<u32>) {
        // CEF renders asynchronously via on_paint — nothing to do here.
    }

    fn new_view(&mut self, size: Size<u32>, content: Option<PageType>) -> ViewId {
        // A browser may be created before the first `update()`; make sure CEF is
        // initialized on this (the pump) thread first.
        self.ensure_initialized();
        let id = rand::thread_rng().gen();

        if let Some(view) = self.create_browser_view(id, size, None) {
            self.views.push(view);
            if let Some(page_type) = content {
                self.goto(id, page_type);
            }
        }

        id
    }

    fn remove_view(&mut self, id: ViewId) {
        if let Some(pos) = self.views.iter().position(|v| v.id == id) {
            let view = &self.views[pos];
            if let Some(host) = view.browser.host() {
                host.close_browser(1);
            }
            self.views.remove(pos);
            // Pump the message loop so CEF processes the close and
            // renderer subprocesses can terminate.
            for _ in 0..10 {
                do_message_loop_work();
            }
        }
    }

    fn has_view(&self, id: ViewId) -> bool {
        self.views.iter().any(|v| v.id == id) || self.parked_views.iter().any(|v| v.id == id)
    }

    fn suspend_view(&mut self, id: ViewId) {
        if let Some(pos) = self.views.iter().position(|v| v.id == id) {
            let view = self.views.remove(pos);
            let parked = ParkedView {
                id: view.id,
                last_frame: view.last_frame,
            };
            if let Some(host) = view.browser.host() {
                host.close_browser(1);
            }
            self.parked_views.push(parked);
            // Pump the message loop so CEF processes the close and
            // renderer subprocesses can terminate.
            for _ in 0..10 {
                do_message_loop_work();
            }
        }
    }

    fn resume_view(&mut self, id: ViewId, size: Size<u32>, content: Option<PageType>) {
        let last_frame = if let Some(pos) = self.parked_views.iter().position(|v| v.id == id) {
            Some(self.parked_views.remove(pos).last_frame)
        } else {
            None
        };

        if let Some(view) = self.create_browser_view(id, size, last_frame) {
            self.views.push(view);
            if let Some(page_type) = content {
                self.goto(id, page_type);
            }
        }
    }

    fn view_ids(&self) -> Vec<ViewId> {
        self.views.iter().map(|v| v.id).collect()
    }

    fn focus(&mut self) {
        if let Some(view) = self.views.last() {
            if let Some(host) = view.browser.host() {
                host.set_focus(1);
            }
        }
    }

    fn unfocus(&self) {
        if let Some(view) = self.views.last() {
            if let Some(host) = view.browser.host() {
                host.set_focus(0);
            }
        }
    }

    fn resize(&mut self, size: Size<u32>) {
        let w = size.width.max(1);
        let h = size.height.max(1);
        let new_size = Size::new(w, h);
        for view in &mut self.views {
            view.size = new_size;
            view.shared.borrow_mut().size = new_size;
            if let Some(host) = view.browser.host() {
                host.was_resized();
            }
            view.needs_render = true;
        }
    }

    fn set_scale_factor(&mut self, scale: f32) {
        if (self.scale_factor - scale).abs() < f32::EPSILON {
            return;
        }
        self.scale_factor = scale;
        for view in &mut self.views {
            view.shared.borrow_mut().scale_factor = scale;
            if let Some(host) = view.browser.host() {
                host.notify_screen_info_changed();
                host.was_resized();
            }
            view.needs_render = true;
        }
    }

    fn set_background_color(&mut self, color: u32) {
        self.background_color = Some(color);
    }

    fn set_user_agent(&mut self, user_agent: Option<String>) {
        self.user_agent = user_agent;
    }

    fn set_block_navigation(&mut self, block: bool) {
        self.block_navigation = block;
        // Propagate to any already-created views so the flag can be toggled
        // after construction, not just at view-creation time.
        for view in &mut self.views {
            view.shared.borrow_mut().block_navigation = block;
        }
    }

    fn handle_keyboard_event(&mut self, id: ViewId, event: keyboard::Event) {
        let Some(view) = self.find_view_mut(id) else {
            return;
        };
        let Some(host) = view.browser.host() else {
            return;
        };

        match &event {
            keyboard::Event::KeyPressed {
                key,
                text,
                modifiers,
                physical_key,
                location,
                repeat,
                ..
            } => {
                let cef_modifiers =
                    iced_modifiers_to_cef_key(*modifiers) | key_location_flags(*location, *repeat);
                let native = physical_key_to_x11(physical_key);
                // Send RAWKEYDOWN with the unmodified key's virtual key code
                if let Some((vk, unmod_char)) = iced_key_to_cef(key) {
                    let ke = KeyEvent {
                        size: std::mem::size_of::<KeyEvent>(),
                        type_: KeyEventType::RAWKEYDOWN,
                        modifiers: cef_modifiers,
                        windows_key_code: vk as c_int,
                        native_key_code: native,
                        is_system_key: 0,
                        character: unmod_char,
                        unmodified_character: unmod_char,
                        focus_on_editable_field: 0,
                    };
                    host.send_key_event(Some(&ke));
                }
                // Send CHAR event with the actual text character produced.
                //
                // Fall back to the key's own character when iced reports no
                // text: numpad digits arrive that way, and without a CHAR event
                // nothing is inserted — the key looks dead while the top-row
                // digits work fine.
                let char_code = text
                    .as_ref()
                    .and_then(|t| t.chars().next())
                    .map(|c| c as u16)
                    .or_else(|| {
                        iced_key_to_cef(key)
                            .map(|(_, ch)| ch)
                            .filter(|ch| *ch >= 0x20)
                    });
                if let Some(ch) = char_code {
                    let char_event = KeyEvent {
                        size: std::mem::size_of::<KeyEvent>(),
                        type_: KeyEventType::CHAR,
                        modifiers: cef_modifiers,
                        windows_key_code: ch as c_int,
                        native_key_code: native,
                        is_system_key: 0,
                        character: ch,
                        unmodified_character: ch,
                        focus_on_editable_field: 0,
                    };
                    host.send_key_event(Some(&char_event));
                }
            }
            keyboard::Event::KeyReleased {
                key,
                modifiers,
                physical_key,
                location,
                ..
            } => {
                // A keyup whose `code`/location disagrees with its keydown is
                // itself anomalous, so mirror them here.
                let cef_modifiers =
                    iced_modifiers_to_cef_key(*modifiers) | key_location_flags(*location, false);
                let native = physical_key_to_x11(physical_key);
                if let Some((vk, character)) = iced_key_to_cef(key) {
                    let ke = KeyEvent {
                        size: std::mem::size_of::<KeyEvent>(),
                        type_: KeyEventType::KEYUP,
                        modifiers: cef_modifiers,
                        windows_key_code: vk as c_int,
                        native_key_code: native,
                        is_system_key: 0,
                        character,
                        unmodified_character: character,
                        focus_on_editable_field: 0,
                    };
                    host.send_key_event(Some(&ke));
                }
            }
            _ => {}
        }
    }

    fn handle_mouse_event(
        &mut self,
        id: ViewId,
        point: Point,
        event: mouse::Event,
        modifiers: keyboard::Modifiers,
    ) {
        let Some(view) = self.find_view_mut(id) else {
            return;
        };
        let Some(host) = view.browser.host() else {
            return;
        };

        let cef_modifiers = iced_modifiers_to_cef(modifiers) | view.pressed_buttons;

        match event {
            mouse::Event::ButtonPressed(button) => {
                // Notify CEF that the browser has host-level focus (required
                // for input fields to activate in OSR mode).
                host.set_focus(1);

                // Set the button flag *before* building MouseEvent so the
                // press itself already carries the held-button flag.
                view.pressed_buttons |= mouse_button_event_flag(button);
                let cef_modifiers = cef_modifiers | view.pressed_buttons;
                let me = MouseEvent {
                    x: point.x as c_int,
                    y: point.y as c_int,
                    modifiers: cef_modifiers,
                };
                // Multi-click detection: same button within 500ms and 4px radius
                let now = std::time::Instant::now();
                let is_repeat = view
                    .last_click
                    .as_ref()
                    .is_some_and(|(prev_pt, prev_time)| {
                        now.duration_since(*prev_time).as_millis() < 500
                            && (point.x - prev_pt.x).abs() < 4.0
                            && (point.y - prev_pt.y).abs() < 4.0
                    });
                view.click_count = if is_repeat {
                    (view.click_count % 3) + 1 // cycle 1 → 2 → 3 → 1
                } else {
                    1
                };
                view.last_click = Some((point, now));

                if let Some(cef_btn) = iced_button_to_cef(button) {
                    host.send_mouse_click_event(Some(&me), cef_btn, 0, view.click_count);
                }
            }
            mouse::Event::ButtonReleased(button) => {
                // Clear the button flag *after* sending the release event
                // so the release itself still carries the button flag.
                let me = MouseEvent {
                    x: point.x as c_int,
                    y: point.y as c_int,
                    modifiers: cef_modifiers,
                };
                if let Some(cef_btn) = iced_button_to_cef(button) {
                    host.send_mouse_click_event(Some(&me), cef_btn, 1, view.click_count.max(1));
                }
                view.pressed_buttons &= !mouse_button_event_flag(button);
            }
            mouse::Event::CursorMoved { .. } => {
                let me = MouseEvent {
                    x: point.x as c_int,
                    y: point.y as c_int,
                    modifiers: cef_modifiers,
                };
                host.send_mouse_move_event(Some(&me), 0);
            }
            mouse::Event::WheelScrolled { delta } => {
                drop(host);
                self.scroll(id, point, delta);
            }
            mouse::Event::CursorLeft => {
                let me = MouseEvent {
                    x: point.x as c_int,
                    y: point.y as c_int,
                    modifiers: cef_modifiers,
                };
                host.send_mouse_move_event(Some(&me), 1);
            }
            _ => {}
        }
    }

    fn scroll(&mut self, id: ViewId, point: Point, delta: mouse::ScrollDelta) {
        let Some(view) = self.find_view_mut(id) else {
            return;
        };
        let Some(host) = view.browser.host() else {
            return;
        };

        let me = MouseEvent {
            x: point.x as c_int,
            y: point.y as c_int,
            modifiers: 0,
        };

        let (dx, dy) = match delta {
            mouse::ScrollDelta::Lines { x, y } => ((x * 53.0) as c_int, (y * 53.0) as c_int),
            mouse::ScrollDelta::Pixels { x, y } => (x as c_int, y as c_int),
        };

        host.send_mouse_wheel_event(Some(&me), dx, dy);
    }

    fn goto(&mut self, id: ViewId, page_type: PageType) {
        let Some(view) = self.find_view_mut(id) else {
            return;
        };
        let Some(frame) = view.browser.main_frame() else {
            return;
        };

        match page_type {
            PageType::Url(url) => {
                view.url = url.clone();
                let cef_url = CefString::from(url.as_str());
                frame.load_url(Some(&cef_url));
            }
            PageType::Html(html) => {
                let data_url = format!(
                    "data:text/html;charset=utf-8,{}",
                    urlencoding::encode(&html)
                );
                let cef_url = CefString::from(data_url.as_str());
                frame.load_url(Some(&cef_url));
            }
        }
    }

    fn execute_javascript(&mut self, id: ViewId, code: &str) {
        let Some(view) = self.find_view_mut(id) else {
            return;
        };
        let Some(frame) = view.browser.main_frame() else {
            return;
        };
        let cef_code = CefString::from(code);
        frame.execute_java_script(Some(&cef_code), None, 0);
    }

    fn refresh(&mut self, id: ViewId) {
        if let Some(view) = self.find_view(id) {
            view.browser.reload();
        }
    }

    fn go_forward(&mut self, id: ViewId) {
        if let Some(view) = self.find_view(id) {
            view.browser.go_forward();
        }
    }

    fn go_back(&mut self, id: ViewId) {
        if let Some(view) = self.find_view(id) {
            view.browser.go_back();
        }
    }

    fn get_url(&self, id: ViewId) -> String {
        let Some(view) = self.find_view(id) else {
            return "about:blank".to_string();
        };
        // Use view.url which is kept up-to-date by on_address_change callback.
        // Avoids expensive frame.url() UTF-16→UTF-8 conversion every tick.
        if view.url.is_empty() {
            "about:blank".to_string()
        } else {
            view.url.clone()
        }
    }

    fn take_popup_url(&mut self, id: ViewId) -> Option<String> {
        self.find_view(id)
            .and_then(|view| view.shared.borrow_mut().popup_url.take())
    }

    fn take_page_loaded(&mut self, id: ViewId) -> bool {
        self.find_view(id)
            .map(|view| {
                let mut shared = view.shared.borrow_mut();
                let loaded = shared.page_loaded;
                shared.page_loaded = false;
                loaded
            })
            .unwrap_or(false)
    }

    fn take_console_messages(&mut self, id: ViewId) -> Vec<ConsoleMessage> {
        self.find_view(id)
            .map(|view| std::mem::take(&mut view.shared.borrow_mut().console_messages))
            .unwrap_or_default()
    }

    fn get_title(&self, id: ViewId) -> String {
        self.find_view(id)
            .map(|v| v.title.clone())
            .unwrap_or_default()
    }

    fn get_cursor(&self, id: ViewId) -> Interaction {
        self.find_view(id)
            .map(|v| v.cursor)
            .unwrap_or(Interaction::Idle)
    }

    fn get_view(&self, id: ViewId) -> &ImageInfo {
        static BLANK: std::sync::LazyLock<ImageInfo> =
            std::sync::LazyLock::new(|| ImageInfo::blank(1, 1));
        self.find_view(id)
            .map(|v| &v.last_frame)
            .or_else(|| {
                self.parked_views
                    .iter()
                    .find(|v| v.id == id)
                    .map(|v| &v.last_frame)
            })
            .unwrap_or(&BLANK)
    }

    fn initialization_error(&self) -> Option<&str> {
        self.init_error.as_deref()
    }
}

impl Drop for Cef {
    fn drop(&mut self) {
        // Close all open browsers so CEF subprocesses can exit cleanly.
        for view in &self.views {
            if let Some(host) = view.browser.host() {
                host.close_browser(1);
            }
        }
        self.views.clear();

        // Pump the message loop to let close events propagate.
        // We intentionally never call shutdown() — CEF does not support
        // re-initialization after shutdown, and multiple Cef instances
        // share the same global engine. The OS reclaims all resources
        // when the process exits.
        if self.initialized {
            for _ in 0..10 {
                do_message_loop_work();
            }
        }
    }
}

fn iced_button_to_cef(button: mouse::Button) -> Option<MouseButtonType> {
    match button {
        mouse::Button::Left => Some(MouseButtonType::LEFT),
        mouse::Button::Middle => Some(MouseButtonType::MIDDLE),
        mouse::Button::Right => Some(MouseButtonType::RIGHT),
        _ => None,
    }
}

/// Map a physical key to the X11 hardware keycode CEF expects in
/// `native_key_code` on Linux.
///
/// Chromium derives `KeyboardEvent.code` — the *physical* key, independent of
/// layout — from this field. Sending 0 leaves `event.code` empty on every
/// keystroke, which breaks sites that bind to physical keys and is an obvious
/// automation signal: real Chrome never reports an empty code.
///
/// Values are standard X11 keycodes (evdev scancode + 8) for a PC105 layout.
/// Unknown keys fall back to 0, i.e. exactly the previous behaviour.
fn physical_key_to_x11(physical: &keyboard::key::Physical) -> c_int {
    use keyboard::key::{Code, Physical};
    let Physical::Code(code) = physical else {
        return 0;
    };
    match code {
        Code::Escape => 9,
        Code::Digit1 => 10,
        Code::Digit2 => 11,
        Code::Digit3 => 12,
        Code::Digit4 => 13,
        Code::Digit5 => 14,
        Code::Digit6 => 15,
        Code::Digit7 => 16,
        Code::Digit8 => 17,
        Code::Digit9 => 18,
        Code::Digit0 => 19,
        Code::Minus => 20,
        Code::Equal => 21,
        Code::Backspace => 22,
        Code::Tab => 23,
        Code::KeyQ => 24,
        Code::KeyW => 25,
        Code::KeyE => 26,
        Code::KeyR => 27,
        Code::KeyT => 28,
        Code::KeyY => 29,
        Code::KeyU => 30,
        Code::KeyI => 31,
        Code::KeyO => 32,
        Code::KeyP => 33,
        Code::BracketLeft => 34,
        Code::BracketRight => 35,
        Code::Enter => 36,
        Code::ControlLeft => 37,
        Code::KeyA => 38,
        Code::KeyS => 39,
        Code::KeyD => 40,
        Code::KeyF => 41,
        Code::KeyG => 42,
        Code::KeyH => 43,
        Code::KeyJ => 44,
        Code::KeyK => 45,
        Code::KeyL => 46,
        Code::Semicolon => 47,
        Code::Quote => 48,
        Code::Backquote => 49,
        Code::ShiftLeft => 50,
        Code::Backslash => 51,
        Code::KeyZ => 52,
        Code::KeyX => 53,
        Code::KeyC => 54,
        Code::KeyV => 55,
        Code::KeyB => 56,
        Code::KeyN => 57,
        Code::KeyM => 58,
        Code::Comma => 59,
        Code::Period => 60,
        Code::Slash => 61,
        Code::ShiftRight => 62,
        Code::NumpadMultiply => 63,
        Code::AltLeft => 64,
        Code::Space => 65,
        Code::CapsLock => 66,
        Code::F1 => 67,
        Code::F2 => 68,
        Code::F3 => 69,
        Code::F4 => 70,
        Code::F5 => 71,
        Code::F6 => 72,
        Code::F7 => 73,
        Code::F8 => 74,
        Code::F9 => 75,
        Code::F10 => 76,
        Code::NumLock => 77,
        Code::ScrollLock => 78,
        Code::Numpad7 => 79,
        Code::Numpad8 => 80,
        Code::Numpad9 => 81,
        Code::NumpadSubtract => 82,
        Code::Numpad4 => 83,
        Code::Numpad5 => 84,
        Code::Numpad6 => 85,
        Code::NumpadAdd => 86,
        Code::Numpad1 => 87,
        Code::Numpad2 => 88,
        Code::Numpad3 => 89,
        Code::Numpad0 => 90,
        Code::NumpadDecimal => 91,
        Code::F11 => 95,
        Code::F12 => 96,
        Code::NumpadEnter => 104,
        Code::ControlRight => 105,
        Code::NumpadDivide => 106,
        Code::AltRight => 108,
        Code::Home => 110,
        Code::ArrowUp => 111,
        Code::PageUp => 112,
        Code::ArrowLeft => 113,
        Code::ArrowRight => 114,
        Code::End => 115,
        Code::ArrowDown => 116,
        Code::PageDown => 117,
        Code::Insert => 118,
        Code::Delete => 119,
        Code::SuperLeft => 133,
        Code::SuperRight => 134,
        Code::ContextMenu => 135,
        _ => 0,
    }
}

/// Extra CEF event flags describing *where* a key is, and whether it repeated.
///
/// Without these, numpad digits report `KeyboardEvent.location === 0` instead of
/// 3, left and right modifiers are indistinguishable, and a held key reports
/// `event.repeat === false` forever.
fn key_location_flags(location: keyboard::Location, repeat: bool) -> u32 {
    let mut flags = 0;
    match location {
        keyboard::Location::Numpad => flags |= 1 << 9, // EVENTFLAG_IS_KEY_PAD
        keyboard::Location::Left => flags |= 1 << 10,  // EVENTFLAG_IS_LEFT
        keyboard::Location::Right => flags |= 1 << 11, // EVENTFLAG_IS_RIGHT
        keyboard::Location::Standard => {}
    }
    if repeat {
        flags |= 1 << 13; // EVENTFLAG_IS_REPEAT
    }
    flags
}

/// Return the CEF event-flag bit for a held mouse button.
///
/// Values are `cef_event_flags_t` from CEF's `internal/cef_types.h`. They were
/// previously each one bit too high, so a left press arrived as
/// `EVENTFLAG_MIDDLE_MOUSE_BUTTON`: the page saw `event.button === 0` (left)
/// together with `event.buttons === 4` (middle), a combination no real mouse
/// can produce and a cheap automation signal for bot detection.
fn mouse_button_event_flag(button: mouse::Button) -> u32 {
    match button {
        mouse::Button::Left => 1 << 4,   // EVENTFLAG_LEFT_MOUSE_BUTTON
        mouse::Button::Middle => 1 << 5, // EVENTFLAG_MIDDLE_MOUSE_BUTTON
        mouse::Button::Right => 1 << 6,  // EVENTFLAG_RIGHT_MOUSE_BUTTON
        _ => 0,
    }
}

/// Convert iced keyboard modifiers to CEF event flag bitmask.
fn iced_modifiers_to_cef(modifiers: keyboard::Modifiers) -> u32 {
    let mut flags: u32 = 0;
    if modifiers.shift() {
        flags |= 2; // EVENTFLAG_SHIFT_DOWN
    }
    if modifiers.control() {
        flags |= 4; // EVENTFLAG_CONTROL_DOWN
    }
    if modifiers.alt() {
        flags |= 8; // EVENTFLAG_ALT_DOWN
    }
    flags
}

fn iced_modifiers_to_cef_key(modifiers: keyboard::Modifiers) -> u32 {
    let mut flags: u32 = 0;
    if modifiers.shift() {
        flags |= 2; // EVENTFLAG_SHIFT_DOWN
    }
    if modifiers.control() {
        flags |= 4; // EVENTFLAG_CONTROL_DOWN
    }
    if modifiers.alt() {
        flags |= 8; // EVENTFLAG_ALT_DOWN
    }
    flags
}

fn iced_key_to_cef(key: &keyboard::Key) -> Option<(i32, u16)> {
    use keyboard::key::Named;

    match key {
        keyboard::Key::Character(s) => {
            let ch = s.chars().next()?;
            let vk = if ch.is_ascii_alphabetic() {
                ch.to_ascii_uppercase() as i32
            } else if ch.is_ascii_digit() {
                ch as i32 // 0x30..=0x39 — same as VK_0..VK_9
            } else {
                // Map punctuation to Windows OEM VK codes to avoid
                // collisions with control keys (e.g. '.'=46=VK_DELETE).
                match ch {
                    '.' | '>' => 0xBE,  // VK_OEM_PERIOD
                    ',' | '<' => 0xBC,  // VK_OEM_COMMA
                    ';' | ':' => 0xBA,  // VK_OEM_1
                    '/' | '?' => 0xBF,  // VK_OEM_2
                    '`' | '~' => 0xC0,  // VK_OEM_3
                    '[' | '{' => 0xDB,  // VK_OEM_4
                    '\\' | '|' => 0xDC, // VK_OEM_5
                    ']' | '}' => 0xDD,  // VK_OEM_6
                    '\'' | '"' => 0xDE, // VK_OEM_7
                    '-' | '_' => 0xBD,  // VK_OEM_MINUS
                    '=' | '+' => 0xBB,  // VK_OEM_PLUS
                    ' ' => 0x20,        // VK_SPACE
                    _ => ch as i32,     // fallback for other chars
                }
            };
            Some((vk, ch as u16))
        }
        keyboard::Key::Named(named) => {
            let (vk, ch) = match named {
                Named::Enter => (0x0D, 0x0D),
                Named::Tab => (0x09, 0x09),
                Named::Space => (0x20, 0x20),
                Named::Backspace => (0x08, 0x08),
                Named::Delete => (0x2E, 0),
                Named::Escape => (0x1B, 0x1B),
                Named::Insert => (0x2D, 0),
                Named::Home => (0x24, 0),
                Named::End => (0x23, 0),
                Named::PageUp => (0x21, 0),
                Named::PageDown => (0x22, 0),
                Named::ArrowUp => (0x26, 0),
                Named::ArrowDown => (0x28, 0),
                Named::ArrowLeft => (0x25, 0),
                Named::ArrowRight => (0x27, 0),
                Named::F1 => (0x70, 0),
                Named::F2 => (0x71, 0),
                Named::F3 => (0x72, 0),
                Named::F4 => (0x73, 0),
                Named::F5 => (0x74, 0),
                Named::F6 => (0x75, 0),
                Named::F7 => (0x76, 0),
                Named::F8 => (0x77, 0),
                Named::F9 => (0x78, 0),
                Named::F10 => (0x79, 0),
                Named::F11 => (0x7A, 0),
                Named::F12 => (0x7B, 0),
                Named::Shift => (0x10, 0),
                Named::Control => (0x11, 0),
                Named::Alt => (0x12, 0),
                Named::Super => (0x5B, 0),
                Named::CapsLock => (0x14, 0),
                _ => return None,
            };
            Some((vk, ch))
        }
        _ => None,
    }
}
