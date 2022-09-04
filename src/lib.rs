extern crate fnv;
extern crate miniquad;
pub extern crate rgb;

use std::time::{Duration, Instant};

use fnv::FnvHashSet;

use miniquad::conf::Conf;
use miniquad::*;
pub use miniquad::conf::Icon;
pub use miniquad::{KeyCode, KeyMods, MouseButton, FilterMode};

use rgb::{ComponentBytes, RGBA8};

#[derive(Debug)]
pub struct Config {
    pub window_title: String,
    pub window_width: u32,
    pub window_height: u32,
    pub fullscreen: bool,
    pub icon: Option<Icon>,
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

    key_codes: FnvHashSet<KeyCode>,
    key_mods: KeyMods,
    key_repeated: bool,

    mouse_pos: (f32, f32),
    mouse_buttons: FnvHashSet<MouseButton>,
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

            key_codes: FnvHashSet::default(),
            key_mods: KeyMods {
                shift: false,
                ctrl: false,
                alt: false,
                logo: false
            },
            key_repeated: false,

            mouse_pos: (0., 0.),
            mouse_buttons: FnvHashSet::default(),
        }
    }
}

pub struct Context<'a> {
    win: &'a mut Window,
    ctx: &'a mut GraphicsContext,
}

impl<'a> Context<'a> {
    #[inline]
    pub fn width(&self) -> u32 {
        self.win.width
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.win.height
    }

    #[inline]
    pub fn delta_time(&self) -> Duration {
        self.win.delta_time
    }

    #[inline]
    pub fn clear_color(&mut self, color: RGBA8) {
        self.win.clear_color = color;
    }

    #[inline]
    pub fn is_key_down(&self, key: KeyCode) -> bool {
        self.win.key_codes.contains(&key)
    }

    #[inline]
    pub fn get_key_mods(&self) -> KeyMods {
        self.win.key_mods
    }

    #[inline]
    pub fn is_key_repeated(&self) -> bool {
        self.win.key_repeated
    }

    #[inline]
    pub fn get_mouse_pos(&self) -> (f32, f32) {
        self.win.mouse_pos
    }

    #[inline]
    pub fn get_mouse_pos_int(&self) -> (i32, i32) {
        let (x, y) = self.win.mouse_pos;
        
        (x.round() as i32, y.round() as i32)
    }

    #[inline]
    pub fn is_mouse_button_down(&self, button: MouseButton) -> bool {
        self.win.mouse_buttons.contains(&button)
    }

    #[inline]
    pub fn quit(&mut self) {
        self.ctx.quit();
    }
    
    #[inline]
    pub fn show_mouse(&mut self, shown: bool) {
        self.ctx.show_mouse(shown);
    }
    
    #[inline]
    pub fn set_fullscreen(&mut self, fullscreen: bool) {
        self.ctx.set_fullscreen(fullscreen);
    }
    
    #[inline]
    pub fn get_clipboard(&mut self) -> Option<String> {
        self.ctx.clipboard_get()
    }
    
    #[inline]
    pub fn set_clipboard(&mut self, data: &str) {
        self.ctx.clipboard_set(data);
    }

    pub fn clear(&mut self) {
        for pix in self.win.buffer.iter_mut() {
            *pix = self.win.clear_color;
        }
    }

    #[inline]
    pub fn draw_pixel(&mut self, x: i32, y: i32, color: RGBA8) {
        self.win.buffer[y as usize * self.win.width as usize + x as usize] = color;
    }

    pub fn draw_rect(&mut self, x: i32, y: i32, width: u32, height: u32, color: RGBA8) {
        for y in (y as usize)..(y as usize + height as usize) {
            for x in (x as usize)..(x as usize + width as usize) {
                self.win.buffer[y * self.win.width as usize + x] = color;
            }
        }
    }
    
    pub fn draw_pixels(&mut self, x: i32, y: i32, width: u32, height: u32, pixels: &[RGBA8]) {
        for iy in 0..(height as usize) {
            for ix in 0..(width as usize) {
                self.win.buffer[(iy + y as usize) * self.win.width as usize + ix + x as usize] = pixels[iy * width as usize + ix];
            }
        }
    }

    #[inline]
    pub fn draw_screen(&mut self, pixels: &[RGBA8]) {
        self.win.buffer.copy_from_slice(pixels);
    }

    #[inline]
    pub fn get_draw_buffer(&self) -> &[RGBA8] {
        &self.win.buffer
    }

    #[inline]
    pub fn get_mut_draw_buffer(&mut self) -> &mut [RGBA8] {
        &mut self.win.buffer
    }

    pub fn set_filter_mode(&mut self, filter: FilterMode) {
        let texture = &self.win.bindings.images[0];
        texture.set_filter(self.ctx, filter);
    }
}

pub trait State: 'static {
    fn update(&mut self, ctx: &mut Context);
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
        self.win.key_codes.insert(key_code);
        self.win.key_mods = key_mods;
        self.win.key_repeated = repeat;
    }

    fn key_up_event(
        &mut self,
        _ctx: &mut GraphicsContext,
        key_code: KeyCode,
        key_mods: KeyMods,
    ) {
        self.win.key_codes.remove(&key_code);
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
        self.win.mouse_buttons.insert(button);
    }

    fn mouse_button_up_event(
        &mut self,
        _ctx: &mut GraphicsContext,
        button: MouseButton,
        _x: f32,
        _y: f32,
    ) {
        self.win.mouse_buttons.remove(&button);
    }
}

pub fn start(config: Config, state: impl State) {
    let conf = Conf {
        window_title: config.window_title,
        window_width: config.window_width as i32,
        window_height: config.window_height as i32,
        fullscreen: config.fullscreen,
        window_resizable: false,
        icon: config.icon,
        ..Default::default()
    };

    miniquad::start(conf, move |ctx| Box::new(
        Handler {
            win: Window::init(ctx, config.window_width, config.window_height),
            state: Box::new(state),
        }
    ));
}
