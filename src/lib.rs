//! A library to embed web views in iced applications.
//!
//! Supports [Blitz](https://github.com/DioxusLabs/blitz) (Rust-native, modern CSS),
//! [litehtml](https://github.com/franzos/litehtml-rs) (lightweight, CPU-based), and
//! [Servo](https://servo.org/) (full browser: HTML5, CSS3, JS).
//!
//! Has two separate widgets: Basic, and Advanced.
//! The basic widget is simple to implement — use abstractions like `CloseCurrent` and `ChangeView`.
//! The advanced widget gives you direct `ViewId` control for multiple simultaneous views.
//!
//! # Basic usage
//!
//! ```rust,ignore
//! enum Message {
//!    WebView(iced_webview::Action),
//!    Update,
//! }
//!
//! struct State {
//!    webview: iced_webview::WebView<iced_webview::Blitz, Message>,
//! }
//! ```
//!
//! Then call the usual `view/update` methods — see
//! [examples](https://github.com/franzos/iced_webview_v2/tree/main/examples) for full working code.
//!
use std::sync::Arc;

use iced::widget::image;

/// Engine Trait and Engine implementations
pub mod engines;
pub use engines::{Engine, PageType, PixelFormat, ViewId};

mod webview;
pub use basic::{Action, WebView};
pub use webview::{advanced, basic};

#[cfg(feature = "blitz")]
pub use engines::blitz::Blitz;

#[cfg(feature = "litehtml")]
pub use engines::litehtml::Litehtml;

#[cfg(feature = "servo")]
pub use engines::servo::Servo;

#[cfg(feature = "cef")]
pub use engines::cef_engine::{cef_subprocess_check, Cef};

pub(crate) mod util;

#[cfg(any(feature = "litehtml", feature = "blitz"))]
pub(crate) mod fetch;

/// Image details for passing the view around
#[derive(Clone, Debug)]
pub struct ImageInfo {
    width: u32,
    height: u32,
    /// Original pixel format (RGBA or BGRA). Stored so the shader path
    /// can pick the matching GPU texture format and avoid a CPU swizzle.
    pixel_format: PixelFormat,
    /// Lazily created on first `as_handle()` call. Shader-path engines
    /// (CEF, Servo) never call `as_handle()`, so the Handle allocation
    /// and the pixel-buffer clone it requires are avoided entirely.
    handle: std::sync::OnceLock<image::Handle>,
    raw_pixels: Arc<Vec<u8>>,
}

impl Default for ImageInfo {
    fn default() -> Self {
        let pixels = vec![255; (Self::WIDTH as usize * Self::HEIGHT as usize) * 4];
        let raw_pixels = Arc::new(pixels);
        Self {
            width: Self::WIDTH,
            height: Self::HEIGHT,
            pixel_format: PixelFormat::Rgba,
            handle: std::sync::OnceLock::new(),
            raw_pixels,
        }
    }
}

impl ImageInfo {
    // The default dimensions
    const WIDTH: u32 = 800;
    const HEIGHT: u32 = 800;

    fn new(pixels: Vec<u8>, format: PixelFormat, width: u32, height: u32) -> Self {
        // R, G, B, A
        assert_eq!(pixels.len() % 4, 0);

        // Store pixels in their native format — no CPU swizzle.
        // The shader path picks the matching GPU texture format
        // (Bgra8UnormSrgb vs Rgba8UnormSrgb) so the GPU handles
        // channel reordering for free.
        // The Handle path (as_handle()) swizzles lazily on first call.
        let raw_pixels = Arc::new(pixels);
        Self {
            width,
            height,
            pixel_format: format,
            handle: std::sync::OnceLock::new(),
            raw_pixels,
        }
    }

    /// Get the image handle for direct rendering.
    ///
    /// Creates the Handle lazily on first call. Shader-path engines
    /// (CEF, Servo) never call this, avoiding the clone entirely.
    pub fn as_handle(&self) -> image::Handle {
        self.handle
            .get_or_init(|| {
                let mut pixels = (*self.raw_pixels).clone();
                // The Handle path requires RGBA. Swizzle if source is BGRA.
                if let PixelFormat::Bgra = self.pixel_format {
                    pixels.chunks_mut(4).for_each(|chunk| chunk.swap(0, 2));
                }
                image::Handle::from_rgba(self.width, self.height, pixels)
            })
            .clone()
    }

    /// Image width.
    pub fn image_width(&self) -> u32 {
        self.width
    }

    /// Image height.
    pub fn image_height(&self) -> u32 {
        self.height
    }

    /// Native pixel format of the stored data.
    pub fn pixel_format(&self) -> &PixelFormat {
        &self.pixel_format
    }

    /// Raw pixel data for direct GPU upload (shader widget path).
    /// Format matches `pixel_format()` — no CPU swizzle applied.
    pub fn pixels(&self) -> Arc<Vec<u8>> {
        Arc::clone(&self.raw_pixels)
    }

    fn blank(width: u32, height: u32) -> Self {
        let pixels = vec![255; (width as usize * height as usize) * 4];
        let raw_pixels = Arc::new(pixels);
        Self {
            width,
            height,
            pixel_format: PixelFormat::Rgba,
            handle: std::sync::OnceLock::new(),
            raw_pixels,
        }
    }
}
