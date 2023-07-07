#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

extern crate fnv;
extern crate miniquad;
pub extern crate rgb;

use std::cmp::min;
use std::time::{Duration, Instant};

use fnv::FnvHashMap;

use miniquad::conf::Conf;
use miniquad::*;
#[doc(no_inline)]
pub use miniquad::{KeyCode, KeyMods, MouseButton, FilterMode, CursorIcon};

use rgb::{ComponentBytes, RGBA8};

/// Application window settings.
#[derive(Debug)]
pub struct Config {
    /// Title of the window
    pub window_title: String,
    /// Width of the window
    pub window_width: u32,
    /// Height of the window
    pub window_height: u32,
    /// Whether the window should be created in fullscreen mode
    pub fullscreen: bool,
    /// Whether the rendering canvas is full-resolution on HighDPI displays.
    /// See <https://docs.rs/miniquad/0.3.16/miniquad/conf/index.html#high-dpi-rendering> for details.
    pub high_dpi: bool,
    /// An optional icon for the window taskbar
    /// Only works on Windows as of currently used version of `miniquad`.
    pub icon: Option<Box<Icon>>,
}

/// Icon image in three levels of detail.
#[derive(Debug)]
pub struct Icon {
    /// 16x16 image (RGBA, row-major order)
    pub small: [RGBA8; 16*16],
    /// 32x32 image (RGBA, row-major order)
    pub medium: [RGBA8; 32*32],
    /// 64x64 image (RGBA, row-major order)
    pub large: [RGBA8; 64*64],
}

impl Icon {
    fn into_miniquad_icon(self) -> miniquad::conf::Icon {
        assert_eq!(std::mem::size_of::<RGBA8>(), 4);

        miniquad::conf::Icon {
            small: unsafe { std::mem::transmute(self.small) },
            medium: unsafe { std::mem::transmute(self.medium) },
            big: unsafe { std::mem::transmute(self.large) },
        }
    }
}

#[repr(C)]
struct Vec2 {
    x: f32,
    y: f32,
}

#[repr(C)]
struct Vertex {
    pos: Vec2,
    uv: Vec2,
}

const SHADER_VERT: &str = r#"#version 100
attribute vec2 pos;
attribute vec2 uv;

varying highp vec2 texcoord;

void main() {
    gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);
    texcoord = uv;
}"#;

const SHADER_FRAG: &str = r#"#version 100
varying highp vec2 texcoord;

uniform sampler2D tex;

void main() {
    gl_FragColor = texture2D(tex, texcoord);
}"#;

/// Input state of a mouse/keyboard button
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum InputState {
    /// The button has just been pressed.
    Pressed,
    /// The button is being held down.
    Down,
    /// The button has just been released.
    Released,
}

#[derive(Debug)]
struct Window {
    width: u32,
    height: u32,

    pipeline: Pipeline,
    bindings: Bindings,

    instant: Instant,
    delta_time: Duration,

    clear_color: RGBA8,
    buffer: Vec<RGBA8>,

    keys: FnvHashMap<KeyCode, InputState>,
    key_mods: KeyMods,

    mouse_pos: (f32, f32),
    mouse_buttons: FnvHashMap<MouseButton, InputState>,
}

impl Window {
    fn init(ctx: &mut GraphicsContext, width: u32, height: u32) -> Self {
        let vertices: [Vertex; 4] = [
            Vertex { pos: Vec2 { x: -1., y: -1. }, uv: Vec2 { x: 0., y: 1. } },
            Vertex { pos: Vec2 { x:  1., y: -1. }, uv: Vec2 { x: 1., y: 1. } },
            Vertex { pos: Vec2 { x:  1., y:  1. }, uv: Vec2 { x: 1., y: 0. } },
            Vertex { pos: Vec2 { x: -1., y:  1. }, uv: Vec2 { x: 0., y: 0. } },
        ];
        let vertex_buffer = Buffer::immutable(ctx, BufferType::VertexBuffer, &vertices);

        let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
        let index_buffer = Buffer::immutable(ctx, BufferType::IndexBuffer, &indices);

        let texture = Texture::new_render_texture(ctx, TextureParams {
            format: TextureFormat::RGBA8,
            wrap: TextureWrap::Clamp,
            filter: FilterMode::Nearest,
            width,
            height,
        });

        let bindings = Bindings {
            vertex_buffers: vec![vertex_buffer],
            index_buffer,
            images: vec![texture],
        };

        let shader_meta = ShaderMeta {
            images: vec!["tex".to_string()],
            uniforms: UniformBlockLayout {
                uniforms: vec![],
            },
        };

        let shader = Shader::new(ctx, SHADER_VERT, SHADER_FRAG, shader_meta).unwrap_or_else(|err| panic!("{}", err));

        let pipeline = Pipeline::new(
            ctx,
            &[BufferLayout::default()],
            &[
                VertexAttribute::new("pos", VertexFormat::Float2),
                VertexAttribute::new("uv", VertexFormat::Float2),
            ],
            shader
        );

        let black = RGBA8 { r: 0, g: 0, b: 0, a: 255 };

        Self {
            width,
            height,

            pipeline,
            bindings,

            instant: Instant::now(),
            delta_time: Duration::ZERO,

            clear_color: black,
            buffer: vec![black; (width * height) as usize],

            keys: FnvHashMap::default(),
            key_mods: KeyMods {
                shift: false,
                ctrl: false,
                alt: false,
                logo: false
            },

            mouse_pos: (0., 0.),
            mouse_buttons: FnvHashMap::default(),
        }
    }
}

/// An object that holds the app's global state.
pub struct Context<'a> {
    win: &'a mut Window,
    ctx: &'a mut GraphicsContext,
}

impl<'a> Context<'a> {
    /// Display width (returns the value provided in [`Config`])
    /// 
    /// Does not account for dpi scale.
    #[inline]
    pub fn width(&self) -> u32 {
        self.win.width
    }

    /// Display height (returns the value provided in [`Config`])
    /// 
    /// Does not account for dpi scale.
    #[inline]
    pub fn height(&self) -> u32 {
        self.win.height
    }

    /// The dpi scaling factor (window pixels to framebuffer pixels).
    /// See <https://docs.rs/miniquad/0.3.16/miniquad/conf/index.html#high-dpi-rendering> for details.
    /// 
    /// Always 1.0 if `high_dpi` in [`Config`] is set to `false`.
    #[inline]
    pub fn dpi_scale(&self) -> f32 {
        self.ctx.dpi_scale()
    }

    /// The current framebuffer size in pixels.
    /// Accounts for dpi scale. See <https://docs.rs/miniquad/0.3.16/miniquad/conf/index.html#high-dpi-rendering> for details.
    #[inline]
    pub fn screen_size(&self) -> (f32, f32) {
        self.ctx.screen_size()
    }

    /// Time passed between previous and current frame.
    #[inline]
    pub fn delta_time(&self) -> Duration {
        self.win.delta_time
    }

    /// Set clear/background color.
    /// 
    /// The buffer isn't cleared automatically, use [`Context::clear()`] for that.
    #[inline]
    pub fn clear_color(&mut self, color: RGBA8) {
        self.win.clear_color = color;
    }

    /// Returns current input state of a key or `None` if it isn't held.
    /// 
    /// Note that [`InputState::Released`] means that the key has **just** been released, **not** that it isn't held.
    #[inline]
    pub fn get_key_state(&self, key: KeyCode) -> Option<InputState> {
        self.win.keys.get(&key).copied()
    }

    /// Returns `true` if a key is down.
    #[inline]
    pub fn is_key_down(&self, key: KeyCode) -> bool {
        self.get_key_state(key).map_or(false, |state| state != InputState::Released)
    }

    /// Returns `true` if a key has just been pressed.
    #[inline]
    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.get_key_state(key).map_or(false, |state| state == InputState::Pressed)
    }

    /// Returns `true` if a key has just been released.
    #[inline]
    pub fn is_key_released(&self, key: KeyCode) -> bool {
        self.get_key_state(key).map_or(false, |state| state == InputState::Released)
    }

    /// Returns currently held key modifiers.
    #[inline]
    pub fn get_key_mods(&self) -> KeyMods {
        self.win.key_mods
    }

    /// Returns current mouse position.
    /// 
    /// Note that it does not account for dpi scale.
    #[inline]
    pub fn get_mouse_pos(&self) -> (f32, f32) {
        self.win.mouse_pos
    }

    /// Returns current mouse position rounded to the nearest integer.
    /// 
    /// Note that it does not account for dpi scale.
    #[inline]
    pub fn get_mouse_pos_int(&self) -> (i32, i32) {
        let (x, y) = self.win.mouse_pos;
        
        (x.round() as i32, y.round() as i32)
    }

    /// Returns current input state of a mouse button or `None` if it isn't held.
    /// 
    /// Note that [`InputState::Released`] means that the key has **just** been released, **not** that it isn't held.
    #[inline]
    pub fn get_mouse_button_state(&self, button: MouseButton) -> Option<InputState> {
        self.win.mouse_buttons.get(&button).copied()
    }

    /// Returns `true` if a mouse button is down.
    #[inline]
    pub fn is_mouse_button_down(&self, button: MouseButton) -> bool {
        self.get_mouse_button_state(button).map_or(false, |state| state != InputState::Released)
    }

    /// Returns `true` if a mouse button has just been pressed.
    #[inline]
    pub fn is_mouse_button_pressed(&self, button: MouseButton) -> bool {
        self.get_mouse_button_state(button).map_or(false, |state| state == InputState::Pressed)
    }

    /// Returns `true` if a mouse button has just been released.
    #[inline]
    pub fn is_mouse_button_released(&self, button: MouseButton) -> bool {
        self.get_mouse_button_state(button).map_or(false, |state| state == InputState::Released)
    }

    /// Quit the application.
    #[inline]
    pub fn quit(&mut self) {
        self.ctx.quit();
    }

    /// Show or hide the mouse cursor.
    #[inline]
    pub fn show_mouse(&mut self, shown: bool) {
        self.ctx.show_mouse(shown);
    }

    /// Set the mouse cursor icon.
    #[inline]
    pub fn set_mouse_cursor(&mut self, cursor_icon: CursorIcon) {
        self.ctx.set_mouse_cursor(cursor_icon);
    }

    /// Set window to fullscreen or not.
    #[inline]
    pub fn set_fullscreen(&mut self, fullscreen: bool) {
        self.ctx.set_fullscreen(fullscreen);
    }

    /// Get current OS clipboard value.
    #[inline]
    pub fn get_clipboard(&mut self) -> Option<String> {
        self.ctx.clipboard_get()
    }

    /// Save value to OS clipboard.
    #[inline]
    pub fn set_clipboard(&mut self, data: &str) {
        self.ctx.clipboard_set(data);
    }

    /// Clear the buffer with the current [`Context::clear_color()`].
    #[inline]
    pub fn clear(&mut self) {
        for pix in self.win.buffer.iter_mut() {
            *pix = self.win.clear_color;
        }
    }

    /// Draw a pixels at (x, y).
    /// 
    /// Does nothing if the position is outside the screen.
    #[inline]
    pub fn draw_pixel(&mut self, x: u32, y: u32, color: RGBA8) {
        if let Some(pix) = self.win.buffer.get_mut(y as usize * self.win.width as usize + x as usize) {
            *pix = color;
        }
    }

    /// Draw a colored rectangle.
    /// 
    /// Does not panic if a part of the rectangle isn't on screen, just draws the part that is.
    pub fn draw_rect(&mut self, x: u32, y: u32, width: u32, height: u32, color: RGBA8) {
        for y in y..min(y + height, self.win.height) {
            for x in x..min(x + width, self.win.width) {
                self.win.buffer[(y * self.win.width + x) as usize] = color;
            }
        }
    }

    /// Fills a rectangle with provided pixels (row-major order).
    /// 
    /// Does not panic if a part of the rectangle isn't on screen, just draws the part that is.
    pub fn draw_pixels(&mut self, x: u32, y: u32, width: u32, height: u32, pixels: &[RGBA8]) {
        let max_width = min(width, self.win.width.saturating_sub(x)) as usize;
        let max_height = min(height, self.win.height.saturating_sub(y)) as usize;

        if max_width == 0 || max_height == 0 {
            return;
        }

        for (iy, line) in pixels.chunks_exact(width as usize).enumerate().take(max_height) {
            let offset = (iy + y as usize) * self.win.width as usize + x as usize;

            self.win.buffer[offset..(offset + max_width)].copy_from_slice(&line[..max_width]);
        }
    }
    
    /// Fills the entire screen buffer at once.
    /// 
    /// Does not panic if a part of the rectangle isn't on screen, just draws the part that is.
    pub fn draw_screen(&mut self, pixels: &[RGBA8]) {
        for (iy, line) in pixels.chunks_exact(self.win.width as usize).enumerate().take(self.win.height as usize) {
            let offset = iy * self.win.width as usize;

            self.win.buffer[offset..(offset + self.win.width as usize)].copy_from_slice(&line[..(self.win.width as usize)]);
        }
    }

    /// Returns the screen buffer.
    #[inline]
    pub fn get_draw_buffer(&self) -> &[RGBA8] {
        &self.win.buffer
    }

    /// Returns the screen buffer.
    /// 
    /// Can be used for drawing.
    #[inline]
    pub fn get_mut_draw_buffer(&mut self) -> &mut [RGBA8] {
        &mut self.win.buffer
    }

    /// Sets the filter mode.
    /// 
    /// The default one is `nearest`.
    #[inline]
    pub fn set_filter_mode(&mut self, filter: FilterMode) {
        let texture = &self.win.bindings.images[0];
        texture.set_filter(self.ctx, filter);
    }
}

/// Application state.
pub trait State: 'static {
    /// Called every frame.
    fn update(&mut self, ctx: &mut Context);
    /// Called every frame after `update()`.
    /// See <https://docs.rs/miniquad/0.3.16/miniquad/trait.EventHandler.html#tymethod.update> for specifics.
    /// 
    /// Note that in `simple-pixels` it's still safe to draw in `update()`.
    fn draw(&mut self, ctx: &mut Context);
}

struct Handler {
    win: Window,
    state: Box<dyn State>,
}

impl EventHandler for Handler {
    fn update(&mut self, ctx: &mut GraphicsContext) {
        self.win.delta_time = self.win.instant.elapsed();
        self.win.instant = Instant::now();

        let mut context = Context {
            win: &mut self.win,
            ctx,
        };

        self.state.update(&mut context);

        self.win.keys.retain(|_, state| {
            match state {
                InputState::Down => true,
                InputState::Pressed => {
                    *state = InputState::Down;
                    true
                },
                InputState::Released => false,
            }
        });

        self.win.mouse_buttons.retain(|_, state| {
            match state {
                InputState::Down => true,
                InputState::Pressed => {
                    *state = InputState::Down;
                    true
                },
                InputState::Released => false,
            }
        });
    }

    fn draw(&mut self, ctx: &mut GraphicsContext) {
        let mut context = Context {
            win: &mut self.win,
            ctx,
        };

        self.state.draw(&mut context);

        let texture = &self.win.bindings.images[0];
        texture.update(ctx, self.win.buffer.as_bytes());

        ctx.begin_default_pass(PassAction::Nothing);

        ctx.apply_pipeline(&self.win.pipeline);
        ctx.apply_bindings(&self.win.bindings);

        ctx.draw(0, 6, 1);

        ctx.end_render_pass();
        ctx.commit_frame();
    }

    fn key_down_event(
        &mut self,
        _ctx: &mut GraphicsContext,
        key_code: KeyCode,
        key_mods: KeyMods,
        repeat: bool,
    ) {
        if !repeat {
            self.win.keys.insert(key_code, InputState::Pressed);
        }
        self.win.key_mods = key_mods;
    }

    fn key_up_event(
        &mut self,
        _ctx: &mut GraphicsContext,
        key_code: KeyCode,
        key_mods: KeyMods,
    ) {
        self.win.keys.insert(key_code, InputState::Released);
        self.win.key_mods = key_mods;
    }

    fn mouse_motion_event(&mut self, _ctx: &mut GraphicsContext, x: f32, y: f32) {
        self.win.mouse_pos = (x, y);
    }

    fn mouse_button_down_event(
        &mut self,
        _ctx: &mut GraphicsContext,
        button: MouseButton,
        _x: f32,
        _y: f32,
    ) {
        self.win.mouse_buttons.insert(button, InputState::Pressed);
    }

    fn mouse_button_up_event(
        &mut self,
        _ctx: &mut GraphicsContext,
        button: MouseButton,
        _x: f32,
        _y: f32,
    ) {
        self.win.mouse_buttons.insert(button, InputState::Released);
    }
}

/// Start the application using provided config and state.
pub fn start(config: Config, state: impl State) {
    let conf = Conf {
        window_title: config.window_title,
        window_width: config.window_width as i32,
        window_height: config.window_height as i32,
        fullscreen: config.fullscreen,
        high_dpi: config.high_dpi,
        window_resizable: false,
        icon: config.icon.map(|icon| icon.into_miniquad_icon()),
        ..Default::default()
    };

    miniquad::start(conf, move |ctx| Box::new(
        Handler {
            win: Window::init(ctx, config.window_width, config.window_height),
            state: Box::new(state),
        }
    ));
}
