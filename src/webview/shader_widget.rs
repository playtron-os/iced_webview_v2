use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use iced::mouse::{self, Interaction};
use iced::wgpu;
use iced::widget::shader;
use iced::{keyboard, touch, Event, Point, Rectangle, Size};

use crate::engines::PixelFormat;
use crate::webview::basic::Action;
use crate::webview::{is_touch_release, touch_to_mouse};
use crate::ImageInfo;

/// Shader-based rendering for servo webview content.
///
/// Uses direct GPU texture updates (`queue.write_texture()`) instead of iced's
/// image Handle cache, avoiding the texture allocation churn and visible
/// flickering that happens during rapid frame updates (e.g. scrolling).
pub struct WebViewShaderProgram<'a> {
    image_info: &'a ImageInfo,
    cursor: Interaction,
    detected_scale: Arc<AtomicU32>,
}

impl<'a> WebViewShaderProgram<'a> {
    pub fn new(
        image_info: &'a ImageInfo,
        cursor: Interaction,
        detected_scale: Arc<AtomicU32>,
    ) -> Self {
        Self {
            image_info,
            cursor,
            detected_scale,
        }
    }
}

#[derive(Default)]
pub struct ShaderState {
    bounds: Size<u32>,
    /// Whether the webview has keyboard focus (user clicked inside it).
    focused: bool,
    /// Current keyboard modifier state (shift, ctrl, alt).
    modifiers: keyboard::Modifiers,
    /// Mouse button held inside the webview — continue forwarding events
    /// even when the cursor leaves the widget bounds (scrollbar drag, text
    /// selection, etc.).
    dragging: bool,
}

pub struct WebViewPrimitive {
    pub(crate) pixels: Arc<Vec<u8>>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pixel_format: PixelFormat,
    /// Shared atomic for auto-detecting display scale factor from the GPU viewport.
    pub(crate) detected_scale: Arc<AtomicU32>,
    /// Set when the frame is already a GPU texture, in which case `pixels` is
    /// empty and there is nothing to upload.
    #[cfg(feature = "cef")]
    pub(crate) accelerated: Option<Arc<crate::engines::accelerated::AcceleratedSurface>>,
}

impl std::fmt::Debug for WebViewPrimitive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebViewPrimitive")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
    }
}

pub struct WebViewPipeline {
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    /// Holds the widget/texture size ratio the fragment shader anchors with.
    layout_buffer: wgpu::Buffer,
    /// Last ratio written, so a settled view stops touching the queue.
    last_uv_scale: [f32; 2],
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
    texture_size: (u32, u32),
    /// Current texture pixel format so we can recreate with the right format.
    texture_format: wgpu::TextureFormat,
    /// Whether the render target is an sRGB format. CEF delivers sRGB display
    /// bytes; the uploaded texture's sRGB-ness MUST match the target's so the
    /// bytes reach the screen unchanged (see `to_wgpu_format`).
    target_is_srgb: bool,
    /// Tracks the data pointer of the last uploaded pixel buffer.
    /// When the Arc points to the same allocation, the frame is unchanged
    /// and `write_texture()` can be skipped entirely.
    last_pixels_ptr: usize,
    /// Generation of the GPU surface the bind group currently points at.
    /// Frames arriving into the same texture do not change it — only a resize
    /// does, which is exactly when the bind group has to be rebuilt.
    #[cfg(feature = "cef")]
    accel_generation: Option<u64>,
}

impl WebViewPipeline {
    fn recreate_texture(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) {
        let (texture, texture_view) = create_texture(device, width.max(1), height.max(1), format);

        self.bind_group = make_bind_group(
            device,
            &self.bind_group_layout,
            &texture_view,
            &self.sampler,
            &self.layout_buffer,
        );

        self.texture = texture;
        self.texture_view = texture_view;
        self.texture_size = (width, height);
        self.texture_format = format;
    }

    /// Tell the shader how the texture maps onto the widget.
    ///
    /// The ratio is 1:1 whenever the page has caught up, which is the steady
    /// state; it only departs from that for the frame or two after a resize,
    /// where anchoring beats stretching.
    /// Returns whether the texture covers the widget, i.e. the page is being
    /// shown at the size it was rendered for.
    fn set_anchor(&mut self, queue: &wgpu::Queue, bounds: &Rectangle, scale: f32) -> bool {
        let (tex_w, tex_h) = self.texture_size;
        if tex_w == 0 || tex_h == 0 {
            return false;
        }
        let bounds_w = (bounds.width * scale).round().max(1.0);
        let bounds_h = (bounds.height * scale).round().max(1.0);
        let uv_scale = [bounds_w / tex_w as f32, bounds_h / tex_h as f32];

        // Rounding between logical and physical pixels can leave the two a
        // pixel apart on a fractional scale, which is not a mismatch worth
        // reacting to.
        let fits = uv_scale.iter().all(|s| (s - 1.0).abs() < 0.01);

        // Writing every frame would queue a buffer copy per frame for a value
        // that almost never changes.
        if uv_scale != self.last_uv_scale {
            self.last_uv_scale = uv_scale;
            let mut data = [0u8; 16];
            data[0..4].copy_from_slice(&uv_scale[0].to_ne_bytes());
            data[4..8].copy_from_slice(&uv_scale[1].to_ne_bytes());
            queue.write_buffer(&self.layout_buffer, 0, &data);
        }

        fits
    }

    /// Point the bind group at a texture we did not allocate.
    ///
    /// Used for GPU-delivered frames, where the texture belongs to the
    /// browser surface rather than to this pipeline.
    #[cfg(feature = "cef")]
    fn bind_external(&mut self, device: &wgpu::Device, texture: &wgpu::Texture) {
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.bind_group = make_bind_group(
            device,
            &self.bind_group_layout,
            &view,
            &self.sampler,
            &self.layout_buffer,
        );
        self.texture_size = (texture.width(), texture.height());
        self.texture_format = texture.format();
    }
}

/// Maps our `PixelFormat` to the matching wgpu texture format.
///
/// The texture's sRGB-ness MUST match the render target's. CEF delivers sRGB
/// display bytes (what a screenshot contains). When the iced surface is an _Srgb
/// format the sampler decodes sRGB→linear and the target re-encodes linear→sRGB,
/// so the bytes round-trip. But iced's `web-colors` mode (which this app enables
/// via icetron) sets `GAMMA_CORRECTION=false`, making iced pick a LINEAR (_Unorm)
/// surface and upload its OWN image atlas as `Rgba8Unorm` — i.e. it passes sRGB
/// bytes straight through with no decode. If we used an _Srgb texture against
/// that linear target, the sampler's sRGB→linear decode is never re-encoded, so
/// midtones crush toward black (a `#212429` page renders ~`#040506` — the "way
/// too dark" webview bug). So we mirror the target: _Srgb texture for an _Srgb
/// surface, plain _Unorm texture for a linear surface — exactly like the atlas.
fn to_wgpu_format(pf: &PixelFormat, target_is_srgb: bool) -> wgpu::TextureFormat {
    match (pf, target_is_srgb) {
        (PixelFormat::Bgra, true) => wgpu::TextureFormat::Bgra8UnormSrgb,
        (PixelFormat::Bgra, false) => wgpu::TextureFormat::Bgra8Unorm,
        (PixelFormat::Rgba, true) => wgpu::TextureFormat::Rgba8UnormSrgb,
        (PixelFormat::Rgba, false) => wgpu::TextureFormat::Rgba8Unorm,
    }
}

fn make_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    layout_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("webview_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: layout_buffer.as_entire_binding(),
            },
        ],
    })
}

fn create_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("webview_texture"),
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
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

// -- Primitive ----------------------------------------------------------------

impl shader::Primitive for WebViewPrimitive {
    type Pipeline = WebViewPipeline;

    fn prepare(
        &self,
        pipeline: &mut Self::Pipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &Rectangle,
        viewport: &shader::Viewport,
    ) {
        // Store the display scale factor so the WebView can pick it up
        // and inform the engine (e.g. CEF's device_scale_factor).
        self.detected_scale
            .store(viewport.scale_factor().to_bits(), Ordering::Relaxed);

        // A GPU-delivered frame is already in a texture: there is nothing to
        // upload, only a bind group to keep pointed at the right one. This is
        // also where the surface first learns which device to import into —
        // it has no other way to find out, and it drops frames until it does.
        #[cfg(feature = "cef")]
        if let Some(surface) = &self.accelerated {
            surface.attach_gpu(
                device,
                queue,
                to_wgpu_format(&self.pixel_format, pipeline.target_is_srgb),
            );

            if let Some((texture, generation)) = surface.current() {
                if pipeline.accel_generation != Some(generation) {
                    pipeline.bind_external(device, &texture);
                    pipeline.accel_generation = Some(generation);
                    // Any CPU frame that follows must re-upload: the bind group
                    // no longer points at this pipeline's own texture.
                    pipeline.last_pixels_ptr = 0;
                }
                let fits = pipeline.set_anchor(queue, bounds, viewport.scale_factor());
                if fits || surface.settle_deadline_passed() {
                    surface.mark_settled();
                }
            }
            return;
        }

        // Coming back from a GPU-delivered frame, the bind group is still
        // pointed at the surface's texture, so it has to be aimed at this
        // pipeline's own one again before anything is uploaded into it.
        #[cfg(feature = "cef")]
        if pipeline.accel_generation.take().is_some() {
            pipeline.recreate_texture(
                device,
                self.width.max(1),
                self.height.max(1),
                to_wgpu_format(&self.pixel_format, pipeline.target_is_srgb),
            );
            pipeline.last_pixels_ptr = 0;
        }

        let needed_format = to_wgpu_format(&self.pixel_format, pipeline.target_is_srgb);
        if (self.width, self.height) != pipeline.texture_size
            || needed_format != pipeline.texture_format
        {
            pipeline.recreate_texture(device, self.width, self.height, needed_format);
            // Force re-upload after texture recreation
            pipeline.last_pixels_ptr = 0;
        }

        // Before the early-out below: the uniform starts zeroed, and a zero
        // ratio collapses every sample onto the top-left texel, so this has to
        // run on the CPU path too — not only when a frame is uploaded.
        pipeline.set_anchor(queue, bounds, viewport.scale_factor());

        // Skip upload when the pixel buffer hasn't changed (same Arc allocation).
        let current_ptr = Arc::as_ptr(&self.pixels) as usize;
        if current_ptr == pipeline.last_pixels_ptr {
            return;
        }

        let expected_len = 4 * self.width as usize * self.height as usize;
        if self.pixels.len() == expected_len && self.width > 0 && self.height > 0 {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &pipeline.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &self.pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * self.width),
                    rows_per_image: Some(self.height),
                },
                wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            );
            pipeline.last_pixels_ptr = current_ptr;
        }
    }

    fn draw(&self, pipeline: &Self::Pipeline, render_pass: &mut wgpu::RenderPass<'_>) -> bool {
        if self.width == 0 || self.height == 0 {
            return true;
        }

        // Nothing to show until the page has been rendered at the size it is
        // being displayed at — otherwise the opening frames appear small in the
        // corner and then jump to fill. `true` means "handled"; drawing nothing
        // leaves whatever is behind the widget.
        #[cfg(feature = "cef")]
        if self
            .accelerated
            .as_ref()
            .is_some_and(|surface| !surface.is_settled())
        {
            return true;
        }
        render_pass.set_pipeline(&pipeline.render_pipeline);
        render_pass.set_bind_group(0, &pipeline.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
        true
    }
}

// -- Pipeline -----------------------------------------------------------------

impl shader::Pipeline for WebViewPipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        // Match the texture's sRGB-ness to the iced render target so CEF's sRGB
        // bytes reach the screen unchanged (see `to_wgpu_format`).
        let target_is_srgb = format.is_srgb();
        let initial_tex_format = if target_is_srgb {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };
        let (texture, texture_view) = create_texture(device, 1, 1, initial_tex_format);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("webview_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("webview_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Starts at 1:1 — the widget and the page agree until a resize.
        let layout_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("webview_layout"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = make_bind_group(
            device,
            &bind_group_layout,
            &texture_view,
            &sampler,
            &layout_buffer,
        );

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("webview_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("webview_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("webview_render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            texture,
            texture_view,
            sampler,
            bind_group_layout,
            bind_group,
            render_pipeline,
            layout_buffer,
            last_uv_scale: [0.0, 0.0],
            texture_size: (1, 1),
            texture_format: initial_tex_format,
            target_is_srgb,
            last_pixels_ptr: 0,
            #[cfg(feature = "cef")]
            accel_generation: None,
        }
    }
}

// -- Program ------------------------------------------------------------------

impl<'a> shader::Program<Action> for WebViewShaderProgram<'a> {
    type State = ShaderState;
    type Primitive = WebViewPrimitive;

    fn update(
        &self,
        state: &mut Self::State,
        event: &Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<shader::Action<Action>> {
        let size = Size::new(bounds.width as u32, bounds.height as u32);
        if state.bounds != size {
            state.bounds = size;
            return Some(shader::Action::publish(Action::Resize(size)));
        }

        match event {
            Event::Keyboard(event) => {
                // Track modifier state from all keyboard events.
                match event {
                    keyboard::Event::KeyPressed { modifiers, .. }
                    | keyboard::Event::KeyReleased { modifiers, .. } => {
                        state.modifiers = *modifiers;
                    }
                    keyboard::Event::ModifiersChanged(modifiers) => {
                        state.modifiers = *modifiers;
                    }
                }
                // Only forward keyboard events when the webview is focused
                // (user clicked inside it). This prevents keystrokes meant
                // for the chat input from also reaching the webview.
                if !state.focused {
                    return None;
                }
                if let keyboard::Event::KeyPressed {
                    key: keyboard::Key::Character(c),
                    modifiers,
                    ..
                } = event
                {
                    if modifiers.command() && c.as_str() == "c" {
                        return Some(shader::Action::publish(Action::CopySelection));
                    }
                }
                Some(shader::Action::publish(Action::SendKeyboardEvent(
                    event.clone(),
                )))
            }
            Event::Mouse(event) => {
                // Track focus: clicking inside grants focus, clicking outside loses it.
                if matches!(event, mouse::Event::ButtonPressed(_)) {
                    let inside = cursor.position_in(bounds).is_some();
                    state.focused = inside;
                    if inside {
                        state.dragging = true;
                    }
                }
                let was_dragging = state.dragging;
                if matches!(event, mouse::Event::ButtonReleased(_)) {
                    state.dragging = false;
                }

                if let Some(point) = cursor.position_in(bounds) {
                    Some(shader::Action::publish(Action::SendMouseEvent(
                        *event,
                        point,
                        state.modifiers,
                    )))
                } else if was_dragging {
                    // Mouse captured: button was pressed inside and is still
                    // held (or just released). Forward events to CEF with
                    // clamped coordinates so drag operations complete properly.
                    if matches!(event, mouse::Event::CursorLeft) {
                        // Cursor left the window while dragging. Send a
                        // synthetic release so CEF cleans up its drag state.
                        state.dragging = false;
                        Some(shader::Action::publish(Action::SendMouseEvent(
                            mouse::Event::ButtonReleased(mouse::Button::Left),
                            Point::ORIGIN,
                            state.modifiers,
                        )))
                    } else if let Some(pos) = cursor.position() {
                        let clamped = Point::new(
                            (pos.x - bounds.x).clamp(0.0, bounds.width),
                            (pos.y - bounds.y).clamp(0.0, bounds.height),
                        );
                        Some(shader::Action::publish(Action::SendMouseEvent(
                            *event,
                            clamped,
                            state.modifiers,
                        )))
                    } else {
                        // No cursor position available (e.g. pointer left the
                        // window entirely on Wayland). Forward button releases
                        // so CEF ends its drag state; swallow move events since
                        // sending them at ORIGIN would jump the scroll position.
                        if matches!(event, mouse::Event::ButtonReleased(_)) {
                            Some(shader::Action::publish(Action::SendMouseEvent(
                                *event,
                                Point::ORIGIN,
                                state.modifiers,
                            )))
                        } else {
                            None
                        }
                    }
                } else if matches!(event, mouse::Event::CursorLeft) {
                    Some(shader::Action::publish(Action::SendMouseEvent(
                        *event,
                        Point::ORIGIN,
                        state.modifiers,
                    )))
                } else {
                    None
                }
            }
            Event::Touch(touch_event) => {
                // Emulate mouse input for the engine (tap = click, drag = move).
                let synthetic = touch_to_mouse(touch_event);
                let released = is_touch_release(touch_event);

                if matches!(touch_event, touch::Event::FingerPressed { .. }) {
                    let inside = cursor.position_in(bounds).is_some();
                    state.focused = inside;
                    if inside {
                        state.dragging = true;
                    }
                }
                let was_dragging = state.dragging;
                if released {
                    state.dragging = false;
                }

                if let Some(point) = cursor.position_in(bounds) {
                    Some(shader::Action::publish(Action::SendMouseEvent(
                        synthetic,
                        point,
                        state.modifiers,
                    )))
                } else if was_dragging && released {
                    // Finger lifted/cancelled outside the view while dragging —
                    // send a release so the engine ends its drag state.
                    Some(shader::Action::publish(Action::SendMouseEvent(
                        mouse::Event::ButtonReleased(mouse::Button::Left),
                        Point::ORIGIN,
                        state.modifiers,
                    )))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn draw(
        &self,
        _state: &Self::State,
        _cursor: mouse::Cursor,
        _bounds: Rectangle,
    ) -> Self::Primitive {
        WebViewPrimitive {
            pixels: self.image_info.pixels(),
            width: self.image_info.image_width(),
            height: self.image_info.image_height(),
            pixel_format: self.image_info.pixel_format().clone(),
            detected_scale: self.detected_scale.clone(),
            #[cfg(feature = "cef")]
            accelerated: self.image_info.accelerated().cloned(),
        }
    }

    fn mouse_interaction(
        &self,
        _state: &Self::State,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Interaction {
        if cursor.position_in(bounds).is_some() {
            self.cursor
        } else {
            Interaction::None
        }
    }
}

// -- WGSL Shader --------------------------------------------------------------

const SHADER_SOURCE: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    // Fullscreen triangle: 3 vertices covering [-1,3] in clip space
    var out: VertexOutput;
    let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
    let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@group(0) @binding(0) var t_texture: texture_2d<f32>;
@group(0) @binding(1) var t_sampler: sampler;

struct Anchor {
    // widget size / texture size, in physical pixels. 1.0 once the page has
    // caught up with the widget, which is the steady state.
    uv_scale: vec2<f32>,
    _pad: vec2<f32>,
};
@group(0) @binding(2) var<uniform> anchor: Anchor;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Anchor the page at the top-left at 1:1 rather than stretching it over
    // the widget. While a resize settles the page is briefly a different size
    // than the widget, and scaling it to fit makes the whole view rubber-band
    // on every drag; anchoring keeps the content still and only the newly
    // exposed strip is wrong for a frame. The sampler clamps to edge, so that
    // strip repeats the border pixels instead of showing a hole.
    var color = textureSample(t_texture, t_sampler, in.uv * anchor.uv_scale);
    color.a = 1.0;
    return color;
}
"#;
