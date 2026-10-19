use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;

use iced::keyboard;
use iced::mouse::{self, Interaction};
use iced::{Point, Size};
use rand::Rng;
use url::Url;

use super::{Engine, PageType, PixelFormat, ViewId};
use crate::ImageInfo;

use litehtml::pixbuf::PixbufContainer;
use litehtml::selection::Selection;
use litehtml::{
    css_escape_ident, BackgroundLayer, BorderRadiuses, Borders, Color, ConicGradient, Document,
    DocumentContainer, DrawContext, FontDescription, FontHandle, FontMetrics, LinearGradient,
    ListMarker, MediaFeatures, Position, RadialGradient, TextTransform,
};

/// Wrapper around `PixbufContainer` that handles CSS import resolution
/// and image baseurl tracking, mirroring litehtml-rs's `BrowseContainer`.
struct WebviewContainer {
    inner: PixbufContainer,
    base_url: String,
    css_cache: RefCell<HashMap<String, String>>,
    /// Maps raw image src → baseurl passed by litehtml, so image fetches
    /// can resolve relative URLs against the correct context (stylesheet
    /// URL, not the page URL).
    image_baseurls: RefCell<HashMap<String, String>>,
}

impl WebviewContainer {
    fn new(width: u32, height: u32, scale: f32) -> Self {
        Self {
            inner: PixbufContainer::new_with_scale(width, height, scale),
            base_url: String::new(),
            css_cache: RefCell::new(HashMap::new()),
            image_baseurls: RefCell::new(HashMap::new()),
        }
    }

    fn inner(&self) -> &PixbufContainer {
        &self.inner
    }

    fn inner_mut(&mut self) -> &mut PixbufContainer {
        &mut self.inner
    }

    fn set_css_cache(&self, cache: HashMap<String, String>) {
        *self.css_cache.borrow_mut() = cache;
    }

    /// Resolve a URL against a given base, falling back to self.base_url.
    fn resolve_against(&self, href: &str, baseurl: &str) -> Option<Url> {
        // Already absolute
        if let Ok(u) = Url::parse(href) {
            return Some(u);
        }
        // Resolve against the provided base context (e.g. stylesheet URL)
        if !baseurl.is_empty() {
            if let Ok(base) = Url::parse(baseurl) {
                if let Ok(u) = base.join(href) {
                    return Some(u);
                }
            }
        }
        // Fall back to page base URL
        if !self.base_url.is_empty() {
            if let Ok(base) = Url::parse(&self.base_url) {
                return base.join(href).ok();
            }
        }
        None
    }
}

// Delegate everything to inner, override import_css, set_base_url, load_image
impl DocumentContainer for WebviewContainer {
    fn create_font(&mut self, descr: &FontDescription) -> (FontHandle, FontMetrics) {
        self.inner.create_font(descr)
    }
    fn delete_font(&mut self, font: FontHandle) {
        self.inner.delete_font(font);
    }
    fn text_width(&self, text: &str, font: FontHandle) -> f32 {
        self.inner.text_width(text, font)
    }
    fn draw_text(
        &mut self,
        hdc: DrawContext,
        text: &str,
        font: FontHandle,
        color: Color,
        pos: Position,
    ) {
        self.inner.draw_text(hdc, text, font, color, pos);
    }
    fn draw_list_marker(&mut self, hdc: DrawContext, marker: &ListMarker) {
        self.inner.draw_list_marker(hdc, marker);
    }
    fn load_image(&mut self, src: &str, baseurl: &str, redraw_on_ready: bool) {
        // Store the baseurl context so image fetches can resolve correctly
        if !baseurl.is_empty() {
            self.image_baseurls
                .borrow_mut()
                .insert(src.to_string(), baseurl.to_string());
        }
        self.inner.load_image(src, baseurl, redraw_on_ready);
    }
    fn get_image_size(&self, src: &str, baseurl: &str) -> litehtml::Size {
        self.inner.get_image_size(src, baseurl)
    }
    fn draw_image(&mut self, hdc: DrawContext, layer: &BackgroundLayer, url: &str, base_url: &str) {
        self.inner.draw_image(hdc, layer, url, base_url);
    }
    fn draw_solid_fill(&mut self, hdc: DrawContext, layer: &BackgroundLayer, color: Color) {
        self.inner.draw_solid_fill(hdc, layer, color);
    }
    fn draw_linear_gradient(
        &mut self,
        hdc: DrawContext,
        layer: &BackgroundLayer,
        gradient: &LinearGradient,
    ) {
        self.inner.draw_linear_gradient(hdc, layer, gradient);
    }
    fn draw_radial_gradient(
        &mut self,
        hdc: DrawContext,
        layer: &BackgroundLayer,
        gradient: &RadialGradient,
    ) {
        self.inner.draw_radial_gradient(hdc, layer, gradient);
    }
    fn draw_conic_gradient(
        &mut self,
        hdc: DrawContext,
        layer: &BackgroundLayer,
        gradient: &ConicGradient,
    ) {
        self.inner.draw_conic_gradient(hdc, layer, gradient);
    }
    fn draw_borders(
        &mut self,
        hdc: DrawContext,
        borders: &Borders,
        draw_pos: Position,
        root: bool,
    ) {
        self.inner.draw_borders(hdc, borders, draw_pos, root);
    }
    fn set_caption(&mut self, caption: &str) {
        self.inner.set_caption(caption);
    }
    fn set_base_url(&mut self, base_url: &str) {
        // Update our stored base URL for resolve_against fallback
        self.base_url = base_url.to_string();
        self.inner.set_base_url(base_url);
    }
    fn on_anchor_click(&mut self, url: &str) {
        self.inner.on_anchor_click(url);
    }
    fn set_cursor(&mut self, cursor: &str) {
        self.inner.set_cursor(cursor);
    }
    fn transform_text(&self, text: &str, tt: TextTransform) -> String {
        self.inner.transform_text(text, tt)
    }
    fn import_css(&self, url: &str, baseurl: &str) -> (String, Option<String>) {
        // Resolve against the baseurl parameter (stylesheet context)
        let resolved = match self.resolve_against(url, baseurl) {
            Some(u) => u,
            None => return (String::new(), None),
        };
        let key = resolved.to_string();
        if let Some(cached) = self.css_cache.borrow().get(&key) {
            return (cached.clone(), Some(key));
        }
        // Not in cache — return empty (the CSS wasn't pre-fetched)
        (String::new(), None)
    }
    fn set_clip(&mut self, pos: Position, radius: BorderRadiuses) {
        self.inner.set_clip(pos, radius);
    }
    fn del_clip(&mut self) {
        self.inner.del_clip();
    }
    fn get_viewport(&self) -> Position {
        self.inner.get_viewport()
    }
    fn get_media_features(&self) -> MediaFeatures {
        self.inner.get_media_features()
    }
}

/// Persistent document and selection state for a view.
///
/// # Safety
///
/// The `doc` field borrows from the `Box<WebviewContainer>` in the parent
/// `LitehtmlView`. The container is heap-allocated for address stability.
/// `doc_state` is always dropped before the container is modified or dropped
/// (field drop order: `doc_state` is declared before `container`).
/// A page longer than this is laid out away from the window: its start at
/// once, so the top shows straight away, and the rest on a thread of its own.
const BACKGROUND_BYTES: usize = 256 * 1024;
/// How much of a long page is laid out at once.
const FIRST_BYTES: usize = 64 * 1024;
/// The step between the start and the whole page, so a reader who scrolls
/// on finds more before all of it is ready.
const SECOND_BYTES: usize = 1024 * 1024;
/// A frame holding the whole page past this many bytes of pixels holds only
/// the part around where the page is scrolled to.
const FULL_FRAME_BYTES: u64 = 64 * 1024 * 1024;
/// How many bytes of pixels that part may take. litehtml walks the whole page
/// on every draw, so the part is drawn again rarely, and each time costs that
/// walk plus this much rastering.
const WINDOW_FRAME_BYTES: u64 = 16 * 1024 * 1024;

/// The start of `html` through the end of a closing tag at or after `bytes`,
/// outside any comment, script or style, so what is cut off is whole
/// elements; litehtml closes what is still open. `None` when no such tag
/// comes after `bytes`.
fn prefix(html: &str, bytes: usize) -> Option<&str> {
    let raw = html.as_bytes();
    let lower = |at: usize, word: &str| {
        raw.get(at..at + word.len())
            .is_some_and(|got| got.eq_ignore_ascii_case(word.as_bytes()))
    };
    let mut i = 0;
    while i < raw.len() {
        if raw[i] != b'<' {
            i += 1;
            continue;
        }
        if lower(i, "<!--") {
            i = html[i..].find("-->").map_or(raw.len(), |end| i + end + 3);
            continue;
        }
        let raw_text = ["script", "style", "textarea", "title"]
            .into_iter()
            .find(|tag| lower(i + 1, tag));
        if let Some(tag) = raw_text {
            let close = format!("</{tag}");
            let found = html[i..]
                .to_ascii_lowercase()
                .find(&close)
                .map(|end| i + end + close.len());
            i = found.unwrap_or(raw.len());
            continue;
        }
        let closing = raw.get(i + 1) == Some(&b'/');
        let end = html[i..].find('>').map_or(raw.len(), |end| i + end + 1);
        if closing && end > bytes && end < raw.len() {
            return Some(&html[..end]);
        }
        i = end;
    }
    None
}

/// A document laid out on a thread of its own, with the container it borrows.
/// Field order is drop order: the document before its container.
struct Built {
    state: Option<DocumentState>,
    container: Box<WebviewContainer>,
    content_height: f32,
    generation: u64,
    /// The whole page, not a start of it.
    complete: bool,
}

// SAFETY: a `Built` is made on one thread and then only ever used on the one
// it is sent to. Its `Rc`s (the font system the container and the measuring
// closure share) and the litehtml document all travel inside it together, so
// nothing it holds is reachable from two threads at once.
unsafe impl Send for Built {}

/// Lay out `html` in `container`, at `w` by `h`. With `second_pass`, a page
/// taller than the first viewport is laid out again with the viewport as tall
/// as the page, so `100vh` covers it; a long page is laid out once.
fn lay_out(
    container: &mut Box<WebviewContainer>,
    html: &str,
    w: u32,
    h: u32,
    second_pass: bool,
) -> (Option<DocumentState>, f32) {
    if w == 0 || h == 0 || html.is_empty() {
        return (None, 0.0);
    }
    // Pass 1: use a tall viewport so CSS `100vh` doesn't cap content height.
    let layout_h = h.max(10_000);
    container.inner_mut().resize(w, layout_h);
    let measure = container.inner().text_measure_fn();

    // SAFETY: Manual lifetime extension is required here due to litehtml API constraints.
    //
    // The litehtml Document<'a> type is invariant over its lifetime parameter and
    // requires a mutable borrow of the container. This makes it incompatible with
    // self-referential struct crates like ouroboros or self_cell, which cannot handle:
    //   1. Lifetime invariance (they require covariance)
    //   2. Multiple mutable borrows from the same field (Document and Selection)
    //
    // The unsafe lifetime extension to 'static is safe because:
    //   1. container is Box<WebviewContainer> — heap-allocated with a stable address
    //   2. the document is always stored beside its container and dropped first
    //   3. the document is dropped before any container modification or drop
    //   4. The Document never outlives the container it borrows from
    let container_ptr = &mut **container as *mut WebviewContainer;
    let container_ref: &'static mut WebviewContainer = unsafe { &mut *container_ptr };
    let mut doc = match Document::from_html(html, container_ref, None, None) {
        Ok(doc) => doc,
        Err(e) => {
            eprintln!("litehtml: from_html failed: {e:?}");
            return (None, 0.0);
        }
    };
    let _ = doc.render(w as f32);
    let measured = doc.height();
    if !second_pass || measured <= layout_h as f32 {
        let state = DocumentState {
            doc,
            measure: Box::new(measure),
            selection: Selection::new(),
        };
        return (Some(state), measured);
    }

    // Pass 2: re-layout so `100vh` covers the full content and overflow clips
    // don't cut it off. Drop the document BEFORE resizing: resizing while it
    // holds a &mut borrow of the container would make two live &mut references.
    drop(doc);
    container.inner_mut().resize(w, measured.ceil() as u32);
    let measure = container.inner().text_measure_fn();
    let container_ptr = &mut **container as *mut WebviewContainer;
    let container_ref: &'static mut WebviewContainer = unsafe { &mut *container_ptr };
    match Document::from_html(html, container_ref, None, None) {
        Err(e) => {
            eprintln!("litehtml: from_html pass 2 failed: {e:?}");
            (None, 0.0)
        }
        Ok(mut doc) => {
            let _ = doc.render(w as f32);
            let height = doc.height();
            let state = DocumentState {
                doc,
                measure: Box::new(measure),
                selection: Selection::new(),
            };
            (Some(state), height)
        }
    }
}

struct DocumentState {
    doc: Document<'static>,
    #[allow(clippy::type_complexity)]
    measure: Box<dyn Fn(&str, FontHandle) -> f32>,
    selection: Selection<'static>,
}

struct LitehtmlView {
    id: ViewId,
    // IMPORTANT: doc_state must be declared before container so it drops first.
    doc_state: Option<DocumentState>,
    container: Box<WebviewContainer>,
    html: String,
    url: String,
    title: String,
    cursor: Interaction,
    last_frame: ImageInfo,
    needs_render: bool,
    /// Fetched image bytes waiting to be flushed into the container.
    /// Accumulated between render cycles so multiple images cause only
    /// one document rebuild instead of one per image.
    staged_images: Vec<(String, Vec<u8>, bool)>,
    /// Selection highlight rects in logical coords, scroll-adjusted.
    /// Drawn as iced quads by the widget so the base image Handle stays stable.
    selection_rects: Vec<[f32; 4]>,
    scroll_y: f32,
    content_height: f32,
    /// The part of the page `last_frame` holds, in logical pixels: all of
    /// it, or for a long page the part around `scroll_y`.
    frame_top: f32,
    frame_height: f32,
    /// Which way the page last scrolled: a long page's next frame reaches
    /// ahead that way.
    scrolling_up: bool,
    size: Size<u32>,
    drag_origin: Option<(f32, f32)>,
    drag_active: bool,
    /// Every image loaded into the page so far, for a layout made on another
    /// thread to start with.
    images: Vec<(String, Vec<u8>)>,
    /// Images that came while a long page was laid out elsewhere; it is laid
    /// out once more with them.
    images_since_layout: bool,
    /// Where a long page's layouts arrive, while one is still to come.
    building: Option<Receiver<Built>>,
    /// The newest layout asked for; an older one landing is dropped.
    latest: Arc<AtomicU64>,
}

/// CPU-based HTML rendering engine backed by litehtml.
///
/// No URL navigation, no keyboard input, no JavaScript.
/// Uses `litehtml::pixbuf::PixbufContainer` for software rasterization.
pub struct Litehtml {
    views: Vec<LitehtmlView>,
    scale_factor: f32,
}

impl Default for Litehtml {
    fn default() -> Self {
        Self {
            views: Vec::new(),
            scale_factor: 1.0,
        }
    }
}

impl Litehtml {
    fn find_view(&self, id: ViewId) -> &LitehtmlView {
        self.views
            .iter()
            .find(|v| v.id == id)
            .expect("The requested View id was not found")
    }

    fn find_view_mut(&mut self, id: ViewId) -> &mut LitehtmlView {
        self.views
            .iter_mut()
            .find(|v| v.id == id)
            .expect("The requested View id was not found")
    }
}

/// Lay the view's page out. A short page is laid out here and now. A long
/// one has its start laid out here, so its top shows at once, and the rest on
/// a thread of its own (see [`lay_out_in_background`]).
fn rebuild_document(view: &mut LitehtmlView) {
    view.doc_state = None;

    // Flush any staged images while doc_state is None (safe — no Document
    // holds a borrow of the container).
    if !view.staged_images.is_empty() {
        for (url, bytes, _) in view.staged_images.drain(..) {
            view.container.inner_mut().load_image_data(&url, &bytes);
            view.images.push((url, bytes));
        }
    }

    let (w, h) = (view.size.width, view.size.height);
    if view.html.len() <= BACKGROUND_BYTES {
        view.latest.fetch_add(1, Ordering::SeqCst);
        view.building = None;
        let (state, height) = lay_out(&mut view.container, &view.html, w, h, true);
        view.doc_state = state;
        view.content_height = height;
        return;
    }
    let start = prefix(&view.html, FIRST_BYTES).unwrap_or_default();
    let (state, height) = lay_out(&mut view.container, start, w, h, false);
    view.doc_state = state;
    view.content_height = height;
    lay_out_in_background(view);
}

/// Lay a long page out on a thread of its own: its first megabyte, then all
/// of it, each sent back as it is done. A newer layout asked for meanwhile
/// makes this one stop at its next step.
fn lay_out_in_background(view: &mut LitehtmlView) {
    let generation = view.latest.fetch_add(1, Ordering::SeqCst) + 1;
    let latest = Arc::clone(&view.latest);
    let (tx, rx) = mpsc::channel();
    let html = view.html.clone();
    let (w, h) = (view.size.width, view.size.height);
    let scale = view.container.inner().scale_factor();
    let base_url = view.container.base_url.clone();
    let css = view.container.css_cache.borrow().clone();
    let images = view.images.clone();
    view.images_since_layout = false;
    std::thread::spawn(move || {
        for limit in [SECOND_BYTES, usize::MAX] {
            if latest.load(Ordering::SeqCst) != generation {
                return;
            }
            let complete = html.len() <= limit;
            let text = if complete {
                html.as_str()
            } else if html.len() > limit * 2 {
                match prefix(&html, limit) {
                    Some(start) => start,
                    None => continue,
                }
            } else {
                continue;
            };
            let mut container = Box::new(WebviewContainer::new(w, h, scale));
            container.base_url = base_url.clone();
            container.set_css_cache(css.clone());
            for (url, bytes) in &images {
                container.inner_mut().load_image_data(url, bytes);
            }
            let (state, content_height) = lay_out(&mut container, text, w, h, false);
            let built = Built {
                state,
                container,
                content_height,
                generation,
                complete,
            };
            if latest.load(Ordering::SeqCst) != generation || tx.send(built).is_err() {
                return;
            }
            if complete {
                return;
            }
        }
    });
    view.building = Some(rx);
}

/// Put in place what a background layout sent, newest first wins; once the
/// whole page has landed, lay it out again if images came meanwhile.
fn take_built(view: &mut LitehtmlView) {
    let Some(rx) = view.building.as_ref() else {
        return;
    };
    let mut landed = Vec::new();
    let mut finished = false;
    loop {
        match rx.try_recv() {
            Ok(built) => landed.push(built),
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                finished = true;
                break;
            }
        }
    }
    let current = view.latest.load(Ordering::SeqCst);
    for built in landed {
        if built.generation != current {
            continue;
        }
        let Built {
            state,
            container,
            content_height,
            complete,
            ..
        } = built;
        // The old document goes before the container it borrows.
        view.doc_state = None;
        view.container = container;
        view.doc_state = state;
        view.content_height = content_height;
        let max_scroll = (content_height - view.size.height as f32).max(0.0);
        view.scroll_y = view.scroll_y.clamp(0.0, max_scroll);
        view.needs_render = true;
        finished |= complete;
    }
    if finished {
        view.building = None;
        if view.images_since_layout {
            lay_out_in_background(view);
        }
    }
}

/// The part of the page a frame holds, as `(top, height)` in logical pixels:
/// all of it, or, when that is more than [`FULL_FRAME_BYTES`], as much as
/// [`WINDOW_FRAME_BYTES`] allows (three windows at least), one window behind
/// where the page is scrolled to and the rest ahead, the way it scrolls.
fn frame_window(view: &LitehtmlView) -> (u32, u32) {
    let full_h = (view.content_height.ceil() as u32).max(view.size.height);
    let scale = f64::from(view.container.inner().scale_factor());
    let row_bytes = f64::from(view.size.width.max(1)) * scale * scale * 4.0;
    if f64::from(full_h) * row_bytes <= FULL_FRAME_BYTES as f64 {
        return (0, full_h);
    }
    let h = view.size.height;
    let window = (WINDOW_FRAME_BYTES as f64 / row_bytes) as u32;
    let height = window.max(h * 3).min(full_h);
    let behind = if view.scrolling_up { height - 2 * h } else { h };
    let top = (view.scroll_y as u32)
        .saturating_sub(behind)
        .min(full_h - height);
    (top, height)
}

/// Whether the window on screen has left the part of the page the frame holds.
fn outside_frame(view: &LitehtmlView) -> bool {
    view.scroll_y < view.frame_top
        || view.scroll_y + view.size.height as f32 > view.frame_top + view.frame_height
}

/// Draw the document into the pixel buffer and capture `last_frame`.
///
/// Resizes the container to fit the full content height, disables CSS
/// overflow clips, draws, then captures the pixels. The caller provides
/// a raw `container_ptr` when the container must be mutated while the
/// Document holds a borrow (flush path); pass `None` for the normal path
/// where no aliasing is needed.
fn capture_frame(view: &mut LitehtmlView, container_ptr: Option<*mut WebviewContainer>) {
    let w = view.size.width;
    let (top, full_h) = frame_window(view);
    view.frame_top = top as f32;
    view.frame_height = full_h as f32;

    // Resize + set_ignore_overflow_clips: use the raw pointer when provided
    // (Document holds a borrow), otherwise go through the safe reference.
    match container_ptr {
        Some(ptr) => unsafe {
            (*ptr).inner_mut().resize(w, full_h);
            (*ptr).inner_mut().set_ignore_overflow_clips(true);
        },
        None => {
            view.container.inner_mut().resize(w, full_h);
            view.container.inner_mut().set_ignore_overflow_clips(true);
        }
    }

    if let Some(ref mut state) = view.doc_state {
        let clip = Position {
            x: 0.0,
            y: 0.0,
            width: w as f32,
            height: full_h as f32,
        };
        state
            .doc
            .draw(DrawContext(0), 0.0, -(top as f32), Some(clip));
    }

    match container_ptr {
        Some(ptr) => unsafe { (*ptr).inner_mut().set_ignore_overflow_clips(false) },
        None => view.container.inner_mut().set_ignore_overflow_clips(false),
    }

    let phys_w = view.container.inner().width();
    let phys_h = view.container.inner().height();
    let pixels = unpremultiply_rgba(view.container.inner().pixels());
    view.last_frame = ImageInfo::new(pixels, PixelFormat::Rgba, phys_w, phys_h);
    view.needs_render = false;
}

/// Render the full document into the pixel buffer and update `last_frame`.
///
/// The buffer covers the entire content height so the widget can scroll
/// by offsetting the draw position — no re-render needed per scroll.
fn draw_view(view: &mut LitehtmlView) {
    capture_frame(view, None);
}

/// Flush staged image bytes into the container, optionally re-layout, and
/// redraw in one pass.
///
/// Avoids re-parsing HTML — only calls `doc.render()` (when needed) which is
/// cheap compared to `Document::from_html`.
///
/// # Safety
///
/// Uses a raw pointer to mutate the container while the Document holds a
/// `&'static mut` borrow. Safe in practice because `load_image_data` only
/// touches the `images` HashMap, which the Document does not access until
/// the next `render()` / `draw()` call that we control.
fn flush_images_and_redraw(view: &mut LitehtmlView) {
    if view.staged_images.is_empty() {
        return;
    }

    let w = view.size.width;
    if w == 0 || view.size.height == 0 {
        return;
    }

    // If ANY image has redraw_on_ready == false, layout may have changed
    // (e.g. <img> without explicit dimensions). Otherwise we can skip
    // the expensive doc.render() and just redraw.
    let needs_render = view.staged_images.iter().any(|(_, _, ror)| !ror);

    // SAFETY: The Document borrows the container but doesn't access `images`
    // right now. We only touch the `images` HashMap via `load_image_data`.
    let container_ptr = &mut *view.container as *mut WebviewContainer;
    for (url, bytes, _) in view.staged_images.drain(..) {
        unsafe { (*container_ptr).inner_mut().load_image_data(&url, &bytes) };
        view.images.push((url, bytes));
    }

    // A long page is laid out again on its own thread rather than here.
    if needs_render && view.html.len() > BACKGROUND_BYTES {
        view.images_since_layout = true;
    } else if needs_render {
        if let Some(ref mut state) = view.doc_state {
            let _ = state.doc.render(w as f32);
            view.content_height = state.doc.height();
        }
    }

    capture_frame(view, Some(container_ptr));
    if view.images_since_layout && view.building.is_none() {
        lay_out_in_background(view);
    }
}

/// Main render entry point: rebuilds the document if needed, then draws.
fn render_view(view: &mut LitehtmlView) {
    let w = view.size.width;
    let h = view.size.height;

    if w == 0 || h == 0 {
        return;
    }

    if view.html.is_empty() {
        let phys_w = view.container.inner().width();
        let phys_h = view.container.inner().height();
        view.last_frame = ImageInfo::blank(phys_w, phys_h);
        view.needs_render = false;
        return;
    }

    if view.doc_state.is_none() {
        rebuild_document(view);
        draw_view(view);
    } else if !view.staged_images.is_empty() {
        // Document exists — inject images, re-layout, and draw in one
        // pass so the container size stays consistent throughout.
        flush_images_and_redraw(view);
    } else {
        draw_view(view);
    }
}

/// Convert premultiplied-alpha RGBA pixels to straight alpha.
///
/// litehtml's pixbuf backend (tiny-skia) stores premultiplied RGBA, but
/// iced's `image::Handle::from_rgba` expects straight (unpremultiplied) alpha.
fn unpremultiply_rgba(pixels: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(pixels.len());
    for chunk in pixels.as_chunks::<4>().0 {
        let a = chunk[3] as u32;
        let r = (chunk[0] as u32 * 255 + a / 2)
            .checked_div(a)
            .unwrap_or(0)
            .min(255) as u8;
        let g = (chunk[1] as u32 * 255 + a / 2)
            .checked_div(a)
            .unwrap_or(0)
            .min(255) as u8;
        let b = (chunk[2] as u32 * 255 + a / 2)
            .checked_div(a)
            .unwrap_or(0)
            .min(255) as u8;
        result.extend_from_slice(&[r, g, b, chunk[3]]);
    }
    result
}

/// Map a CSS cursor value from litehtml to an iced mouse interaction.
fn css_cursor_to_interaction(cursor: &str) -> Interaction {
    match cursor {
        "pointer" => Interaction::Pointer,
        "text" => Interaction::Text,
        "crosshair" => Interaction::Crosshair,
        "grab" => Interaction::Grab,
        "grabbing" => Interaction::Grabbing,
        "not-allowed" | "no-drop" => Interaction::NotAllowed,
        "col-resize" | "ew-resize" => Interaction::ResizingHorizontally,
        "row-resize" | "ns-resize" => Interaction::ResizingVertically,
        _ => Interaction::Idle,
    }
}

/// Store selection rectangles in document coordinates.
/// The widget layer applies the scroll offset when drawing.
fn update_selection_rects(view: &mut LitehtmlView) {
    view.selection_rects.clear();
    if let Some(ref state) = view.doc_state {
        for r in state.selection.rectangles() {
            view.selection_rects.push([r.x, r.y, r.width, r.height]);
        }
    }
}

impl Engine for Litehtml {
    fn handles_urls(&self) -> bool {
        false
    }

    fn update(&mut self) {
        for view in &mut self.views {
            take_built(view);
        }
    }

    fn render(&mut self, _size: Size<u32>) {
        for view in &mut self.views {
            if view.needs_render {
                render_view(view);
            }
        }
    }

    fn request_render(&mut self, id: ViewId, _size: Size<u32>) {
        let view = self.find_view_mut(id);
        if view.needs_render {
            render_view(view);
        }
    }

    fn flush_staged_images(&mut self, id: ViewId, _size: Size<u32>) {
        let view = self.find_view_mut(id);
        if !view.staged_images.is_empty() {
            render_view(view);
        }
    }

    fn new_view(&mut self, size: Size<u32>, content: Option<PageType>) -> ViewId {
        let id = rand::thread_rng().gen();
        let w = size.width.max(1);
        let h = size.height.max(1);

        let html = match &content {
            Some(PageType::Html(html)) => html.clone(),
            _ => String::new(),
        };
        let url = match &content {
            Some(PageType::Url(url)) => url.clone(),
            _ => String::new(),
        };

        let mut view = LitehtmlView {
            id,
            doc_state: None,
            container: Box::new(WebviewContainer::new(w, h, self.scale_factor)),
            html,
            url,
            title: String::new(),
            cursor: Interaction::Idle,
            last_frame: ImageInfo::blank(w, h),
            needs_render: true,
            staged_images: Vec::new(),
            selection_rects: Vec::new(),
            scroll_y: 0.0,
            content_height: 0.0,
            frame_top: 0.0,
            frame_height: 0.0,
            scrolling_up: false,
            size,
            drag_origin: None,
            drag_active: false,
            images: Vec::new(),
            images_since_layout: false,
            building: None,
            latest: Arc::new(AtomicU64::new(0)),
        };

        render_view(&mut view);
        self.views.push(view);
        id
    }

    fn remove_view(&mut self, id: ViewId) {
        self.views.retain(|v| v.id != id);
    }

    fn has_view(&self, id: ViewId) -> bool {
        self.views.iter().any(|v| v.id == id)
    }

    fn view_ids(&self) -> Vec<ViewId> {
        self.views.iter().map(|v| v.id).collect()
    }

    fn focus(&mut self) {
        // No-op: litehtml has no focus model.
    }

    fn unfocus(&self) {
        // No-op: litehtml has no focus model.
    }

    fn resize(&mut self, size: Size<u32>) {
        for view in &mut self.views {
            let width_changed = view.size.width != size.width;
            view.size = size;
            view.needs_render = true;
            // A long page stays on screen at its old width until its new
            // layout lands; a short one is laid out again now.
            if view.html.len() > BACKGROUND_BYTES && view.doc_state.is_some() {
                if width_changed {
                    lay_out_in_background(view);
                }
                continue;
            }
            view.doc_state = None;
        }
    }

    fn set_scale_factor(&mut self, scale: f32) {
        if (self.scale_factor - scale).abs() < f32::EPSILON {
            return;
        }
        self.scale_factor = scale;
        for view in &mut self.views {
            view.doc_state = None;

            view.container
                .inner_mut()
                .resize_with_scale(view.size.width, view.size.height, scale);
            view.needs_render = true;
        }
    }

    fn handle_keyboard_event(&mut self, _id: ViewId, _event: keyboard::Event) {
        // No-op: litehtml has no keyboard interaction.
    }

    fn handle_mouse_event(
        &mut self,
        id: ViewId,
        point: Point,
        event: mouse::Event,
        _modifiers: keyboard::Modifiers,
    ) {
        match event {
            mouse::Event::WheelScrolled { delta } => {
                self.scroll(id, point, delta);
            }
            mouse::Event::ButtonPressed(mouse::Button::Left) => {
                let view = self.find_view_mut(id);
                view.drag_origin = Some((point.x, point.y));
                view.drag_active = false;
                if let Some(ref mut state) = view.doc_state {
                    let doc_y = point.y + view.scroll_y;
                    state.doc.on_lbutton_down(point.x, doc_y, point.x, point.y);
                    state.selection.clear();
                }
                view.selection_rects.clear();
            }
            mouse::Event::CursorMoved { .. } => {
                let view = self.find_view_mut(id);

                // Notify litehtml of mouse movement for :hover and cursor updates
                if let Some(ref mut state) = view.doc_state {
                    let doc_y = point.y + view.scroll_y;
                    state.doc.on_mouse_over(point.x, doc_y, point.x, point.y);
                }
                view.cursor = css_cursor_to_interaction(view.container.inner().cursor());

                if let Some((ox, oy)) = view.drag_origin {
                    let dx = point.x - ox;
                    let dy = point.y - oy;

                    if !view.drag_active && (dx * dx + dy * dy).sqrt() >= 4.0 {
                        view.drag_active = true;
                        if let Some(ref mut state) = view.doc_state {
                            let doc_y = oy + view.scroll_y;
                            state.selection.start_at(
                                &state.doc,
                                &*state.measure,
                                ox,
                                doc_y,
                                ox,
                                oy,
                            );
                        }
                    }

                    if view.drag_active {
                        if let Some(ref mut state) = view.doc_state {
                            let doc_y = point.y + view.scroll_y;
                            state.selection.extend_to(
                                &state.doc,
                                &*state.measure,
                                point.x,
                                doc_y,
                                point.x,
                                point.y,
                            );
                        }
                        update_selection_rects(view);
                    }
                }
            }
            mouse::Event::ButtonReleased(mouse::Button::Left) => {
                let view = self.find_view_mut(id);
                let was_dragging = view.drag_active;
                view.drag_active = false;
                view.drag_origin = None;

                // Always notify litehtml to clear :active pseudo-class state
                if let Some(ref mut state) = view.doc_state {
                    let doc_y = point.y + view.scroll_y;
                    state.doc.on_lbutton_up(point.x, doc_y, point.x, point.y);
                }
                // Discard anchor clicks produced during text selection
                if was_dragging {
                    view.container.inner_mut().take_anchor_click();
                }
            }
            mouse::Event::CursorLeft => {
                let view = self.find_view_mut(id);
                if let Some(ref mut state) = view.doc_state {
                    state.doc.on_mouse_leave();
                }
                view.cursor = Interaction::Idle;
            }
            _ => {}
        }
    }

    fn scroll(&mut self, id: ViewId, _point: Point, delta: mouse::ScrollDelta) {
        let view = self.find_view_mut(id);
        let was = view.scroll_y;
        match delta {
            mouse::ScrollDelta::Lines { y, .. } => {
                view.scroll_y -= y * 40.0;
            }
            mouse::ScrollDelta::Pixels { y, .. } => {
                view.scroll_y -= y;
            }
        }
        let max_scroll = (view.content_height - view.size.height as f32).max(0.0);
        view.scroll_y = view.scroll_y.clamp(0.0, max_scroll);
        if view.scroll_y != was {
            view.scrolling_up = view.scroll_y < was;
        }
        if outside_frame(view) {
            view.needs_render = true;
        }
    }

    fn goto(&mut self, id: ViewId, page_type: PageType) {
        let view = self.find_view_mut(id);
        match page_type {
            PageType::Html(html) => {
                view.doc_state = None;
                // Clear image state from the previous page so stale fetches
                // don't interfere and new images are discovered fresh.
                view.staged_images.clear();
                view.container.inner_mut().clear_pending_images();
                // Clear image baseurls from the previous page
                view.container.image_baseurls.borrow_mut().clear();
                view.images.clear();
                view.images_since_layout = false;

                view.html = html;
                view.scroll_y = 0.0;
                view.needs_render = true;
            }
            PageType::Url(url) => {
                view.container.base_url = url.clone();
                view.url = url;
            }
        }
    }

    fn refresh(&mut self, id: ViewId) {
        let view = self.find_view_mut(id);
        view.doc_state = None;

        view.needs_render = true;
    }

    fn go_forward(&mut self, _id: ViewId) {
        // No-op: no navigation history.
    }

    fn go_back(&mut self, _id: ViewId) {
        // No-op: no navigation history.
    }

    fn get_url(&self, id: ViewId) -> String {
        let url = &self.find_view(id).url;
        if url.is_empty() {
            "about:blank".to_string()
        } else {
            url.clone()
        }
    }

    fn get_title(&self, id: ViewId) -> String {
        self.find_view(id).title.clone()
    }

    fn get_cursor(&self, id: ViewId) -> Interaction {
        self.find_view(id).cursor
    }

    fn get_view(&self, id: ViewId) -> &ImageInfo {
        &self.find_view(id).last_frame
    }

    fn get_scroll_y(&self, id: ViewId) -> f32 {
        self.find_view(id).scroll_y
    }

    fn get_content_height(&self, id: ViewId) -> f32 {
        self.find_view(id).content_height
    }

    fn get_frame_rect(&self, id: ViewId) -> (f32, f32) {
        let view = self.find_view(id);
        (view.frame_top, view.frame_height)
    }

    fn layout_pending(&self, id: ViewId) -> bool {
        self.find_view(id).building.is_some()
    }

    fn get_selected_text(&self, id: ViewId) -> Option<String> {
        self.find_view(id)
            .doc_state
            .as_ref()?
            .selection
            .selected_text()
    }

    fn get_selection_rects(&self, id: ViewId) -> &[[f32; 4]] {
        &self.find_view(id).selection_rects
    }

    fn take_anchor_click(&mut self, id: ViewId) -> Option<String> {
        self.find_view_mut(id)
            .container
            .inner_mut()
            .take_anchor_click()
    }

    fn take_pending_images(&mut self) -> Vec<(ViewId, String, String, bool)> {
        let mut result = Vec::new();
        for view in &mut self.views {
            for (src, redraw_on_ready) in view.container.inner_mut().take_pending_images() {
                let baseurl = view
                    .container
                    .image_baseurls
                    .borrow()
                    .get(&src)
                    .cloned()
                    .unwrap_or_default();
                result.push((view.id, src, baseurl, redraw_on_ready));
            }
        }
        result
    }

    fn load_image_from_bytes(
        &mut self,
        id: ViewId,
        url: &str,
        bytes: &[u8],
        redraw_on_ready: bool,
    ) {
        let view = self.find_view_mut(id);
        view.staged_images
            .push((url.to_string(), bytes.to_vec(), redraw_on_ready));
    }

    fn set_css_cache(&mut self, id: ViewId, cache: HashMap<String, String>) {
        let view = self.find_view_mut(id);
        view.container.set_css_cache(cache);
    }

    fn scroll_to_fragment(&mut self, id: ViewId, fragment: &str) -> bool {
        let view = self.find_view_mut(id);
        let state = match view.doc_state.as_ref() {
            Some(s) => s,
            None => return false,
        };
        let root = match state.doc.root() {
            Some(r) => r,
            None => return false,
        };

        // Try #id first, then [name="fragment"] (matches browser behavior).
        // Escape CSS meta-characters so fragments like "foo.bar" don't get
        // misinterpreted as compound selectors.
        let escaped = css_escape_ident(fragment);
        let id_selector = format!("#{escaped}");
        let el = root.select_one(&id_selector).or_else(|| {
            let quoted = fragment.replace('\\', "\\\\").replace('"', "\\\"");
            let name_selector = format!("[name=\"{quoted}\"]");
            root.select_one(&name_selector)
        });

        if let Some(el) = el {
            let pos = el.placement();
            let max_scroll = (view.content_height - view.size.height as f32).max(0.0);
            view.scroll_y = pos.y.clamp(0.0, max_scroll);
            if outside_frame(view) {
                view.needs_render = true;
            }
            true
        } else {
            false
        }
    }
}
