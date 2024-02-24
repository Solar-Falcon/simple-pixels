#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]
#![warn(missing_docs)]

pub use rgb;
pub use simple_blit;

use miniquad::{
    window, Backend, Bindings, BufferLayout, BufferSource, BufferType, BufferUsage, EventHandler,
    MipmapFilterMode, PassAction, Pipeline, RenderingBackend, ShaderMeta, ShaderSource,
    TextureFormat, TextureId, TextureKind, TextureParams, TextureWrap, UniformBlockLayout,
    VertexAttribute, VertexFormat,
};
use rgb::{ComponentBytes, RGBA8};
use rustc_hash::FxHashMap;
use simple_blit::GenericSurface;
use std::{
    future,
    sync::{Arc, Mutex},
    task::Poll,
    time::Duration,
};

#[doc(no_inline)]
pub use miniquad::{
    conf::Conf as Config, fs::Error as FsError, CursorIcon, FilterMode, KeyCode, KeyMods,
    MouseButton,
};

#[repr(C)]
struct Vec2 {
    x: f32,
    y: f32,
}

impl Vec2 {
    #[inline]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

#[repr(C)]
struct Vertex {
    pos: Vec2,
    uv: Vec2,
}

const SHADER_VERT: &str = r#"#version 100
attribute vec2 pos;
attribute vec2 uv;

varying lowp vec2 texcoord;

void main() {
    gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);
    texcoord = uv;
}"#;

const SHADER_FRAG: &str = r#"#version 100
varying lowp vec2 texcoord;

uniform sampler2D tex;

void main() {
    gl_FragColor = texture2D(tex, texcoord);
}"#;

const SHADER_METAL: &str = r#"
#include <metal_stdlib>

using namespace metal;

struct Vertex {
    float2 pos   [[attribute(0)]];
    float2 uv    [[attribute(1)]];
};

struct FragData {
    float4 position [[position]];
    float2 uv       [[user(locn0)]];
};

vertex FragData vertexShader(
    Vertex v [[stage_in]]
) {
    FragData out;

    out.position = float4(v.pos.x, v.pos.y, 0.0, 1.0);
    out.uv = v.uv;

    return out;
}

fragment float4 fragmentShader(
    FragData in             [[stage_in]],
    texture2d<float> tex    [[texture(0)]],
    sampler texSmplr        [[sampler(0)]]
) {
    return tex.sample(texSmplr, in.uv);
}
"#;

/// Input state of a mouse/keyboard button
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputState {
    /// The button has just been pressed.
    Pressed,
    /// The button is being held down.
    Down,
    /// The button has just been released.
    Released,
}

/// An object that holds the app's global state.
pub struct Context {
    ctx: Box<dyn RenderingBackend>,

    pipeline: Pipeline,
    bindings: Bindings,

    instant: f64,
    delta_time: f64,

    clear_color: RGBA8,
    buffer: Vec<RGBA8>,
    buf_width: u32,
    buf_height: u32,

    keys: FxHashMap<KeyCode, InputState>,
    key_mods: KeyMods,
    mouse_pos: (f32, f32),
    mouse_wheel: (f32, f32),
    mouse_buttons: FxHashMap<MouseButton, InputState>,
}

impl Context {
    #[inline]
    const fn texture_params(width: u32, height: u32) -> TextureParams {
        TextureParams {
            kind: TextureKind::Texture2D,
            format: TextureFormat::RGBA8,
            wrap: TextureWrap::Clamp,
            min_filter: FilterMode::Nearest,
            mag_filter: FilterMode::Nearest,
            mipmap_filter: MipmapFilterMode::None,
            width,
            height,
            allocate_mipmaps: false,
        }
    }

    fn new() -> Self {
        let mut ctx = window::new_rendering_backend();

        let (win_width, win_height) = window::screen_size();
        let (win_width, win_height) = (win_width as u32, win_height as u32);

        #[rustfmt::skip]
        let verices: [Vertex; 4] = [
            Vertex { pos: Vec2::new(-1., -1.), uv: Vec2::new(0., 1.) },
            Vertex { pos: Vec2::new( 1., -1.), uv: Vec2::new(1., 1.) },
            Vertex { pos: Vec2::new( 1.,  1.), uv: Vec2::new(1., 0.) },
            Vertex { pos: Vec2::new(-1.,  1.), uv: Vec2::new(0., 0.) },
        ];
        let vertex_buffer = ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&verices),
        );

        let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
        let index_buffer = ctx.new_buffer(
            BufferType::IndexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&indices),
        );

        let texture = ctx.new_render_texture(Self::texture_params(win_width, win_height));

        let bindings = Bindings {
            vertex_buffers: vec![vertex_buffer],
            index_buffer,
            images: vec![texture],
        };

        let shader_meta = ShaderMeta {
            images: vec!["tex".to_string()],
            uniforms: UniformBlockLayout { uniforms: vec![] },
        };

        let shader = ctx
            .new_shader(
                match ctx.info().backend {
                    Backend::OpenGl => ShaderSource::Glsl {
                        vertex: SHADER_VERT,
                        fragment: SHADER_FRAG,
                    },
                    Backend::Metal => ShaderSource::Msl {
                        program: SHADER_METAL,
                    },
                },
                shader_meta,
            )
            .unwrap_or_else(|err| panic!("{err}"));

        let pipeline = ctx.new_pipeline(
            &[BufferLayout::default()],
            &[
                VertexAttribute::new("pos", VertexFormat::Float2),
                VertexAttribute::new("uv", VertexFormat::Float2),
            ],
            shader,
        );

        Self {
            ctx,

            pipeline,
            bindings,

            instant: miniquad::date::now(),
            delta_time: 0.,

            clear_color: RGBA8::new(0, 0, 0, 255),
            buffer: vec![RGBA8::new(0, 0, 0, 255); (win_width * win_height) as usize],
            buf_width: win_width,
            buf_height: win_height,

            keys: FxHashMap::default(),
            key_mods: KeyMods {
                shift: false,
                ctrl: false,
                alt: false,
                logo: false,
            },
            mouse_pos: (0., 0.),
            mouse_wheel: (0., 0.),
            mouse_buttons: FxHashMap::default(),
        }
    }

    #[inline]
    fn texture(&self) -> TextureId {
        self.bindings.images[0]
    }

    #[inline]
    fn set_texture(&mut self, tex: TextureId) {
        self.bindings.images[0] = tex;
    }

    /// Load file from the path and block until its loaded.
    /// Will use filesystem on PC and do a HTTP request on web.
    pub fn load_file<F>(&self, path: impl AsRef<str>, on_loaded: F)
    where
        F: Fn(Result<Vec<u8>, FsError>) + 'static,
    {
        miniquad::fs::load_file(path.as_ref(), on_loaded);
    }

    /// Load file from the path and block until its loaded.
    /// Will use filesystem on PC and do a HTTP request on web.
    pub async fn load_file_async(&self, path: impl AsRef<str>) -> Result<Vec<u8>, FsError> {
        let contents = Arc::new(Mutex::new(None));

        {
            let contents = contents.clone();

            miniquad::fs::load_file(path.as_ref(), move |result| {
                *contents.lock().unwrap() = Some(result);
            });
        }

        future::poll_fn(move |_ctx| {
            let mut result = contents.lock().unwrap();

            if let Some(result) = result.take() {
                Poll::Ready(result)
            } else {
                Poll::Pending
            }
        })
        .await
    }

    /// Display width.
    ///
    /// Accounts for dpi scale.
    #[inline]
    pub fn display_width(&self) -> f32 {
        window::screen_size().0
    }

    /// Display height.
    ///
    /// Accounts for dpi scale.
    #[inline]
    pub fn display_height(&self) -> f32 {
        window::screen_size().1
    }

    /// Draw buffer width.
    #[inline]
    pub fn buffer_width(&self) -> u32 {
        self.buf_width
    }

    /// Draw buffer height.
    #[inline]
    pub fn buffer_height(&self) -> u32 {
        self.buf_height
    }

    /// The dpi scaling factor (window pixels to framebuffer pixels).
    /// See <https://docs.rs/miniquad/0.3.16/miniquad/conf/index.html#high-dpi-rendering> for details.
    ///
    /// Always 1.0 if `high_dpi` in `Config` is set to `false`.
    #[inline]
    pub fn dpi_scale(&self) -> f32 {
        window::dpi_scale()
    }

    /// Time passed between previous and current frame (in seconds).
    #[inline]
    pub fn delta_time_secs(&self) -> f64 {
        self.delta_time
    }

    /// Time passed between previous and current frame (as std::time::Duration).
    #[inline]
    pub fn delta_time(&self) -> Duration {
        Duration::from_secs_f64(self.delta_time)
    }

    /// Set clear/background color.
    ///
    /// The buffer isn't cleared automatically, use [`Context::clear()`] for that.
    #[inline]
    pub fn clear_color(&mut self, color: RGBA8) {
        self.clear_color = color;
    }

    /// Returns current input state of a key or `None` if it isn't held.
    ///
    /// Note that [`InputState::Released`] means that the key has **just** been released, **not** that it isn't held.
    #[inline]
    pub fn get_key_state(&self, key: KeyCode) -> Option<InputState> {
        self.keys.get(&key).copied()
    }

    /// Returns `true` if a key is down.
    #[inline]
    pub fn is_key_down(&self, key: KeyCode) -> bool {
        self.get_key_state(key)
            .map_or(false, |state| state != InputState::Released)
    }

    /// Returns `true` if a key has just been pressed.
    #[inline]
    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.get_key_state(key)
            .map_or(false, |state| state == InputState::Pressed)
    }

    /// Returns `true` if a key has just been released.
    #[inline]
    pub fn is_key_released(&self, key: KeyCode) -> bool {
        self.get_key_state(key)
            .map_or(false, |state| state == InputState::Released)
    }

    /// Returns currently held key modifiers.
    #[inline]
    pub fn get_key_mods(&self) -> KeyMods {
        self.key_mods
    }

    /// Returns current mouse position in the window.
    #[inline]
    pub fn get_real_mouse_pos(&self) -> (i32, i32) {
        let (x, y) = self.mouse_pos;

        (x as _, y as _)
    }

    /// Returns the point of the buffer where the mouse is.
    /// Differs from `get_real_mouse_pos()` when the window and the buffer have different sizes.
    #[inline]
    pub fn get_buffer_mouse_pos(&self) -> (i32, i32) {
        let (x, y) = self.mouse_pos;
        let (win_width, win_height) = window::screen_size();

        (
            (x / win_width * self.buf_width as f32) as _,
            (y / win_height * self.buf_height as f32) as _,
        )
    }

    /// Get current mouse wheel movement.
    #[inline]
    pub fn get_mouse_wheel(&self) -> (f32, f32) {
        self.mouse_wheel
    }

    /// Returns current input state of a mouse button or `None` if it isn't held.
    ///
    /// Note that [`InputState::Released`] means that the key has **just** been released, **not** that it isn't held.
    #[inline]
    pub fn get_mouse_button_state(&self, button: MouseButton) -> Option<InputState> {
        self.mouse_buttons.get(&button).copied()
    }

    /// Returns `true` if a mouse button is down.
    #[inline]
    pub fn is_mouse_button_down(&self, button: MouseButton) -> bool {
        self.get_mouse_button_state(button)
            .map_or(false, |state| state != InputState::Released)
    }

    /// Returns `true` if a mouse button has just been pressed.
    #[inline]
    pub fn is_mouse_button_pressed(&self, button: MouseButton) -> bool {
        self.get_mouse_button_state(button)
            .map_or(false, |state| state == InputState::Pressed)
    }

    /// Returns `true` if a mouse button has just been released.
    #[inline]
    pub fn is_mouse_button_released(&self, button: MouseButton) -> bool {
        self.get_mouse_button_state(button)
            .map_or(false, |state| state == InputState::Released)
    }

    /// Quit the application.
    #[inline]
    pub fn quit(&self) {
        window::request_quit();
    }

    /// Show or hide the mouse cursor.
    #[inline]
    pub fn show_mouse(&self, shown: bool) {
        window::show_mouse(shown);
    }

    /// Show or hide onscreen keyboard.
    #[inline]
    pub fn show_keyboard(&self, shown: bool) {
        window::show_keyboard(shown);
    }

    /// Set the mouse cursor icon.
    #[inline]
    pub fn set_mouse_cursor(&self, cursor_icon: CursorIcon) {
        window::set_mouse_cursor(cursor_icon);
    }

    /// Set window to fullscreen or not.
    #[inline]
    pub fn set_fullscreen(&self, fullscreen: bool) {
        window::set_fullscreen(fullscreen);
    }

    /// Get current OS clipboard value.
    #[inline]
    pub fn get_clipboard(&self) -> Option<String> {
        window::clipboard_get()
    }

    /// Save value to OS clipboard.
    #[inline]
    pub fn set_clipboard(&self, data: impl AsRef<str>) {
        window::clipboard_set(data.as_ref());
    }

    /// Set the application’s window size.
    ///
    /// Note: resizing the window does not resize the drawing buffer.
    /// It will be scaled to the whole window.
    /// You can use [`Context::set_buffer_size()`] for resizing the buffer.
    #[inline]
    pub fn set_window_size(&mut self, new_width: u32, new_height: u32) {
        window::set_window_size(new_width, new_height);
    }

    /// Set the drawing buffer size. The buffer will be cleared.
    ///
    /// This doesn't change the window size.
    /// The buffer will be scaled to the whole window.
    pub fn set_buffer_size(&mut self, new_width: u32, new_height: u32) {
        self.ctx.delete_texture(self.texture());

        let new_texture = self
            .ctx
            .new_render_texture(Self::texture_params(new_width, new_height));
        self.set_texture(new_texture);

        self.buf_width = new_width;
        self.buf_height = new_height;

        self.buffer.fill(self.clear_color);
        self.buffer
            .resize((new_width * new_height) as usize, self.clear_color);
    }

    /// Clear the screen buffer with the current [`Context::clear_color()`].
    #[inline]
    pub fn clear(&mut self) {
        for pix in self.buffer.iter_mut() {
            *pix = self.clear_color;
        }
    }

    /// Draw a pixels at (x, y).
    ///
    /// Does nothing if the position is outside the screen.
    #[inline]
    pub fn draw_pixel(&mut self, x: i32, y: i32, color: RGBA8) {
        if let Some(pix) = self
            .buffer
            .get_mut(y as usize * self.buf_width as usize + x as usize)
        {
            *pix = color;
        }
    }

    /// Draw a colored rectangle.
    ///
    /// Does not panic if a part of the rectangle isn't on screen, just draws the part that is.
    pub fn draw_rect(&mut self, x: i32, y: i32, width: u32, height: u32, color: RGBA8) {
        simple_blit::blit_whole(
            &mut self.as_mut_surface(),
            simple_blit::point(x as _, y as _),
            &simple_blit::SingleValueSurface::new(color, simple_blit::size(width, height)),
            simple_blit::point(0, 0),
            &[],
        );
    }

    /// Fills a rectangle with provided pixels (row-major order).
    ///
    /// Does not panic if a part of the rectangle isn't on screen, just draws the part that is.
    pub fn draw_pixels(&mut self, x: i32, y: i32, width: u32, height: u32, pixels: &[RGBA8]) {
        if let Some(buffer) =
            simple_blit::GenericSurface::new(pixels, simple_blit::size(width, height))
        {
            simple_blit::blit(
                &mut self.as_mut_surface(),
                simple_blit::point(x as _, y as _),
                &buffer,
                simple_blit::point(0, 0),
                simple_blit::size(width, height),
                &[],
            );
        }
    }

    /// Fills the entire screen buffer at once.
    ///
    /// Does not panic if a part of the rectangle isn't on screen, just draws the part that is.
    pub fn draw_screen(&mut self, pixels: &[RGBA8]) {
        if let Some(buffer) = simple_blit::GenericSurface::new(
            pixels,
            simple_blit::size(self.buf_width, self.buf_height),
        ) {
            simple_blit::blit_whole(
                &mut self.as_mut_surface(),
                simple_blit::point(0, 0),
                &buffer,
                simple_blit::point(0, 0),
                &[],
            );
        }
    }

    /// Returns the screen buffer.
    #[inline]
    pub fn get_draw_buffer(&self) -> &[RGBA8] {
        &self.buffer
    }

    /// Returns the screen buffer.
    ///
    /// Can be used for drawing.
    #[inline]
    pub fn get_mut_draw_buffer(&mut self) -> &mut [RGBA8] {
        &mut self.buffer
    }

    /// Get the draw buffer as a `simple_blit::GenericSurface`.
    #[inline]
    pub fn as_surface(&self) -> GenericSurface<&[RGBA8], RGBA8> {
        GenericSurface::new(
            &self.buffer[..],
            simple_blit::size(self.buf_width, self.buf_height),
        )
        .unwrap()
    }

    /// Get the draw buffer as a mutable `simple_blit::GenericSurface`.
    #[inline]
    pub fn as_mut_surface(&mut self) -> GenericSurface<&mut [RGBA8], RGBA8> {
        GenericSurface::new(
            &mut self.buffer[..],
            simple_blit::size(self.buf_width, self.buf_height),
        )
        .unwrap()
    }

    /// Set the filter for the texture that is used for rendering.
    #[inline]
    pub fn set_texture_filter(&mut self, filter: FilterMode) {
        self.ctx
            .texture_set_filter(self.texture(), filter, MipmapFilterMode::None);
    }
}

/// Application state.
pub trait App {
    /// Called every frame.
    fn update(&mut self, ctx: &mut Context);

    /// Called every frame after `update()`.
    /// See <https://docs.rs/miniquad/0.3.16/miniquad/trait.EventHandler.html#tymethod.update> for specifics.
    ///
    /// Note that when using `simple-pixels` it's still safe to draw in `update()`.
    fn draw(&mut self, ctx: &mut Context);
}

struct Handler<S: App> {
    ctx: Context,
    state: S,
}

impl<S> EventHandler for Handler<S>
where
    S: App,
{
    fn update(&mut self) {
        let new_instant = miniquad::date::now();
        self.ctx.delta_time = new_instant - self.ctx.instant;
        self.ctx.instant = new_instant;

        self.state.update(&mut self.ctx);

        self.ctx.mouse_wheel = (0., 0.);

        self.ctx.keys.retain(|_, state| match state {
            InputState::Down => true,
            InputState::Pressed => {
                *state = InputState::Down;
                true
            }
            InputState::Released => false,
        });

        self.ctx.mouse_buttons.retain(|_, state| match state {
            InputState::Down => true,
            InputState::Pressed => {
                *state = InputState::Down;
                true
            }
            InputState::Released => false,
        });
    }

    fn draw(&mut self) {
        self.state.draw(&mut self.ctx);

        self.ctx
            .ctx
            .texture_update(self.ctx.texture(), self.ctx.buffer.as_bytes());

        self.ctx.ctx.begin_default_pass(PassAction::Nothing);

        self.ctx.ctx.apply_pipeline(&self.ctx.pipeline);
        self.ctx.ctx.apply_bindings(&self.ctx.bindings);

        self.ctx.ctx.draw(0, 6, 1);

        self.ctx.ctx.end_render_pass();

        self.ctx.ctx.commit_frame();
    }

    fn key_down_event(&mut self, key_code: KeyCode, key_mods: KeyMods, repeat: bool) {
        if !repeat {
            self.ctx.keys.insert(key_code, InputState::Pressed);
        }

        self.ctx.key_mods = key_mods;
    }

    fn key_up_event(&mut self, key_code: KeyCode, key_mods: KeyMods) {
        self.ctx.keys.insert(key_code, InputState::Released);
        self.ctx.key_mods = key_mods;
    }

    fn mouse_button_down_event(&mut self, button: MouseButton, _x: f32, _y: f32) {
        self.ctx.mouse_buttons.insert(button, InputState::Pressed);
    }

    fn mouse_button_up_event(&mut self, button: MouseButton, _x: f32, _y: f32) {
        self.ctx.mouse_buttons.insert(button, InputState::Pressed);
    }

    fn mouse_motion_event(&mut self, x: f32, y: f32) {
        self.ctx.mouse_pos = (x, y);
    }

    fn mouse_wheel_event(&mut self, x: f32, y: f32) {
        self.ctx.mouse_wheel = (x, y);
    }

    fn char_event(&mut self, _character: char, key_mods: KeyMods, _repeat: bool) {
        self.ctx.key_mods = key_mods;
    }
}

/// Start the application using provided config and state.
#[inline]
pub fn start(config: Config, state: impl App + 'static) {
    miniquad::start(config, move || {
        Box::new(Handler {
            ctx: Context::new(),
            state,
        })
    })
}
