//! Zero-copy frame delivery for off-screen rendering.
//!
//! The CPU path has Chromium rasterize on the GPU, read the result back into
//! system memory, hand us a BGRA buffer, and then upload that buffer to the
//! GPU again — the whole page crosses the bus twice per frame, 33 MB each way
//! at 4K. This path skips both crossings: the frame arrives as a dmabuf that
//! is already GPU memory, and we blit it into a texture we own without it ever
//! leaving the device.
//!
//! The blit is not avoidable. CEF pools these buffers and reclaims them the
//! moment the paint callback returns (`cef_render_handler.h`: "the handle's
//! resource cannot be cached and cannot be accessed outside of this
//! callback"), so keeping the imported texture around and sampling it later
//! would race Chromium drawing the *next* frame into the same memory. We copy
//! into our own texture and block until the copy has actually run, which is
//! what makes CEF's reuse of the buffer safe.
//!
//! This module deliberately knows nothing about CEF: the engine translates a
//! paint callback into a [`DmabufFrame`], and everything here is plain wgpu.

use std::os::fd::{BorrowedFd, IntoRawFd, RawFd};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use iced::wgpu;

/// "The layout of this buffer is not described by a modifier."
///
/// Importing one requires letting the driver infer the layout, which the
/// explicit-modifier path cannot do — see the check in [`AcceleratedSurface::accept_frame`].
const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;

/// Hands out texture generations that are unique across every surface in the
/// process. See [`AcceleratedSurface::generation`].
fn next_generation() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Whether any GPU on this machine can import browser frames.
///
/// Asked before the first browser is created, because that is when the choice
/// gets baked in — see `accelerated_paint_enabled` in the CEF engine. Costs one
/// adapter enumeration, once per process.
///
/// Note this is necessary rather than sufficient: it establishes that the
/// driver can import dmabufs at all, not that every buffer the browser produces
/// will import. In particular the wgpu revision pinned here reports this
/// feature without also requiring `VK_EXT_image_drm_format_modifier`, which the
/// import actually depends on.
pub fn can_import_dmabuf() -> bool {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN))
        .iter()
        .any(|adapter| {
            adapter
                .features()
                .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
        })
}

/// Drives one of wgpu's native futures to completion.
///
/// They are already complete when returned on native targets, so pulling in an
/// async runtime for a single startup query would not earn its keep.
fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    let mut fut = std::pin::pin!(fut);
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    loop {
        match fut.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::yield_now(),
        }
    }
}

/// One frame's worth of dmabuf, as handed over by the browser engine.
///
/// Only single-plane RGB formats are described here. That is all an OSR page
/// produces — the multi-planar cases in the underlying API are for video.
pub struct DmabufFrame {
    /// Borrowed for the duration of the call only; this type never owns it.
    pub fd: RawFd,
    pub stride: u32,
    pub offset: u32,
    /// Where the page content starts inside the buffer.
    ///
    /// Non-zero while the browser's capture resolution is catching up with a
    /// resize: it letterboxes the content inside the frame, and copying from
    /// the buffer's origin would take the bars instead of the page.
    pub src_x: u32,
    pub src_y: u32,
    /// Size of the whole allocation, not of the plane. Zero when unknown, in
    /// which case Vulkan's own memory requirements are used instead.
    pub total_size: u64,
    pub modifier: u64,
    /// Dimensions of the whole buffer, which is what gets imported. Larger
    /// than the content while a resize is settling.
    pub buffer_width: u32,
    pub buffer_height: u32,
    /// Dimensions of the page content within it — what we actually keep.
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
}

/// The GPU to import into, learned from the renderer rather than created here.
struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Format for the texture we own. Chosen by the renderer, because only it
    /// knows whether the surface it draws into is sRGB — see `to_wgpu_format`
    /// in the shader widget. The imported frame keeps the browser's own
    /// format; copying between the two is legal because they differ only in
    /// sRGB-ness, and reinterpreting the bytes is exactly what we want.
    target_format: wgpu::TextureFormat,
}

/// The texture we own and copy frames into.
struct Target {
    texture: Arc<wgpu::Texture>,
    size: (u32, u32),
    format: wgpu::TextureFormat,
}

/// A page's rendered frames, delivered as GPU textures.
///
/// Shared between the browser's paint callback (which fills it) and the
/// renderer (which samples it), so every method takes `&self`.
#[derive(Default)]
pub struct AcceleratedSurface {
    gpu: Mutex<Option<Gpu>>,
    target: Mutex<Option<Target>>,
    /// Changes whenever the texture *identity* does, so a caller caching a bind
    /// group knows when to rebuild it. Frame contents changing does not change
    /// it — we copy into the same texture.
    ///
    /// Drawn from a process-wide counter rather than counting up per surface.
    /// The renderer keys its cached bind group on this value, and its pipeline
    /// storage is shared by every webview in the process, so two surfaces that
    /// both numbered their first texture "1" would look identical to it — the
    /// second would skip rebinding and draw the first one's page.
    generation: AtomicU64,
    /// Frames accepted so far. The host uses this to tell whether a redraw is
    /// worth scheduling.
    frames: AtomicU64,
    /// Time spent importing and copying, summed between reports.
    import_nanos: AtomicU64,
    /// Last frame size reported, packed as `(w << 32) | h`, so a size change
    /// can be logged — that is the signal that says whether the view settled
    /// at the size it was asked for.
    last_reported: AtomicU64,
}

impl std::fmt::Debug for AcceleratedSurface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AcceleratedSurface")
            .field("attached", &self.has_gpu())
            .field("frames", &self.frame_count())
            .finish()
    }
}

impl AcceleratedSurface {
    pub fn new() -> Self {
        Self::default()
    }

    /// Give the surface the GPU it should import into, and the format the
    /// renderer wants to sample.
    ///
    /// Called from the renderer, which is the only place a device exists. The
    /// first device wins — re-importing onto a different one mid-flight would
    /// invalidate every texture already handed out — but a later change of
    /// format is honoured, since that only means dropping our own texture and
    /// making the next frame rebuild it.
    pub fn attach_gpu(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_format: wgpu::TextureFormat,
    ) {
        let mut gpu = self.gpu.lock().unwrap_or_else(|e| e.into_inner());
        match gpu.as_mut() {
            None => {
                *gpu = Some(Gpu {
                    device: device.clone(),
                    queue: queue.clone(),
                    target_format,
                });
            }
            Some(existing) if existing.target_format != target_format => {
                existing.target_format = target_format;
                *self.target.lock().unwrap_or_else(|e| e.into_inner()) = None;
                self.generation.store(next_generation(), Ordering::Relaxed);
            }
            Some(_) => {}
        }
    }

    /// Whether a device has been attached yet.
    ///
    /// Frames that arrive before the first render are dropped, so the engine
    /// uses this to know it must ask for a repaint once the GPU shows up.
    pub fn has_gpu(&self) -> bool {
        self.gpu.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }

    /// Number of frames accepted so far.
    pub fn frame_count(&self) -> u64 {
        self.frames.load(Ordering::Relaxed)
    }

    /// The texture to sample, with the generation it belongs to.
    pub fn current(&self) -> Option<(Arc<wgpu::Texture>, u64)> {
        let target = self.target.lock().unwrap_or_else(|e| e.into_inner());
        let target = target.as_ref()?;
        Some((
            Arc::clone(&target.texture),
            self.generation.load(Ordering::Relaxed),
        ))
    }

    /// Import one frame and copy it into our own texture.
    ///
    /// Returns `false` if the frame could not be taken — no device yet, or the
    /// import failed — and the caller should fall back to the CPU path rather
    /// than show nothing.
    ///
    /// Blocks until the copy has executed. See the module docs: the caller's
    /// buffer goes back to the browser's pool as soon as this returns.
    pub fn accept_frame(&self, frame: &DmabufFrame) -> bool {
        let gpu = self.gpu.lock().unwrap_or_else(|e| e.into_inner());
        let Some(gpu) = gpu.as_ref() else {
            return false;
        };

        if frame.width == 0 || frame.height == 0 {
            return false;
        }

        // A buffer whose layout the compositor would not name cannot be
        // imported through the explicit-modifier path, which is the only one
        // wgpu offers — it would describe the memory wrongly rather than fail,
        // and the GPU faults on the first read. Observed for real under
        // `--ozone-platform=x11`, which hands over DRM_FORMAT_MOD_INVALID and
        // took out the whole device: amdgpu GPUVM page fault, coredump, gfx
        // ring reset, wgpu queue dead for the rest of the process.
        if frame.modifier == DRM_FORMAT_MOD_INVALID {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                log::warn!(
                    "iced_webview: browser frames arrived with an unspecified layout \
                     (DRM_FORMAT_MOD_INVALID) and cannot be imported; the view will \
                     stay blank"
                );
            }
            return false;
        }

        let started = std::time::Instant::now();

        // Vulkan takes ownership of the fd it imports, but this one belongs to
        // the browser's buffer pool. Hand the import a duplicate so closing it
        // does not close the pool's.
        let dup = match unsafe { BorrowedFd::borrow_raw(frame.fd) }.try_clone_to_owned() {
            Ok(fd) => fd.into_raw_fd(),
            Err(e) => {
                log::warn!("iced_webview: could not dup dmabuf fd: {e}");
                return false;
            }
        };

        let Some(imported) = (unsafe { import_dmabuf(&gpu.device, dup, frame) }) else {
            return false;
        };

        let mut target = self.target.lock().unwrap_or_else(|e| e.into_inner());
        let needs_new = match target.as_ref() {
            Some(t) => t.size != (frame.width, frame.height) || t.format != gpu.target_format,
            None => true,
        };
        if needs_new {
            *target = Some(Target {
                texture: Arc::new(create_target(
                    &gpu.device,
                    frame.width,
                    frame.height,
                    gpu.target_format,
                )),
                size: (frame.width, frame.height),
                format: gpu.target_format,
            });
            // Identity changed — anything caching a view of it must rebuild.
            self.generation.store(next_generation(), Ordering::Relaxed);
        }
        let Some(dst) = target.as_ref() else {
            return false;
        };

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("accelerated_paint_copy"),
            });
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &imported,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: frame.src_x,
                    y: frame.src_y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &dst.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: frame.width,
                height: frame.height,
                depth_or_array_layers: 1,
            },
        );
        let submission = gpu.queue.submit([encoder.finish()]);

        // The imported texture aliases memory the browser is about to reuse,
        // so the copy has to be finished, not merely queued, before we return.
        if let Err(e) = gpu.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        }) {
            log::warn!("iced_webview: waiting for the frame copy failed: {e:?}");
            return false;
        }

        let n = self.frames.fetch_add(1, Ordering::Relaxed) + 1;
        self.import_nanos
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // Periodic rather than per-frame: the point is the steady-state cost,
        // and logging every frame would itself show up in it. The early
        // reports exist because a static page paints only a handful of times,
        // so a purely periodic one would never appear.
        let packed = ((frame.width as u64) << 32) | frame.height as u64;
        let resized = self.last_reported.swap(packed, Ordering::Relaxed) != packed;

        const REPORT_AT: [u64; 4] = [1, 10, 60, 120];
        if resized || REPORT_AT.contains(&n) || n.is_multiple_of(120) {
            let total = self.import_nanos.swap(0, Ordering::Relaxed);
            let window = if n <= 1 { 1 } else { n.min(120) };
            log::debug!(
                "iced_webview: gpu frame {n} {}x{} avg import+copy {:.2}ms \
                 (modifier {:#x}, stride {})",
                frame.width,
                frame.height,
                total as f64 / window as f64 / 1_000_000.0,
                frame.modifier,
                frame.stride,
            );
        }
        true
    }
}

/// Our own copy of a frame, which outlives the browser's buffer.
fn create_target(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("accelerated_paint_target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// Wrap a dmabuf as a texture on `device`.
///
/// # Safety
///
/// `fd` must be a dmabuf matching `frame`'s dimensions, format and modifier.
/// Ownership of `fd` transfers to Vulkan, which closes it — including on the
/// failure paths below.
unsafe fn import_dmabuf(
    device: &wgpu::Device,
    fd: RawFd,
    frame: &DmabufFrame,
) -> Option<wgpu::Texture> {
    use wgpu::hal::api::Vulkan;

    // The import must describe the whole buffer, not just the part we keep —
    // the plane's stride belongs to the full width, and a sub-region copy
    // still reads through the full-size image.
    let size = wgpu::Extent3d {
        width: frame.buffer_width,
        height: frame.buffer_height,
        depth_or_array_layers: 1,
    };

    // The imported image is only ever a copy source; asking for more would
    // constrain which modifiers the driver accepts for no benefit.
    let hal_desc = wgpu::hal::TextureDescriptor {
        label: Some("accelerated_paint_import"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: frame.format,
        usage: wgpu::wgt::TextureUses::COPY_SRC,
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: Vec::new(),
    };

    let plane = wgpu::hal::vulkan::DmaBufPlaneInfo {
        drm_modifier: frame.modifier,
        stride: frame.stride,
        offset: frame.offset,
        total_size: frame.total_size,
    };

    let hal_texture = unsafe {
        let vk_device = device.as_hal::<Vulkan>()?;
        match vk_device.texture_from_dmabuf_fd(fd, &hal_desc, &plane) {
            Ok(texture) => texture,
            Err(e) => {
                log::warn!(
                    "iced_webview: dmabuf import failed ({e:?}); \
                     {}x{} stride={} modifier={:#x}",
                    frame.width,
                    frame.height,
                    frame.stride,
                    frame.modifier
                );
                return None;
            }
        }
    };

    Some(unsafe {
        device.create_texture_from_hal::<Vulkan>(
            hal_texture,
            &wgpu::TextureDescriptor {
                label: Some("accelerated_paint_import"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: frame.format,
                usage: wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        )
    })
}
