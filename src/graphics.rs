pub struct Scene {
    pub timecode: f64,
}
/*
pub trait Renderer {
    fn render(&mut self, scene: &Scene, dst: &mut [u8]) -> Result<(), RendererError>;
    fn device_name(&self) -> &str;
}

pub enum Hardware {
    CPU,
    GPU,
}

pub fn new_renderer(
    parameters: SurfaceParameters,
    hardware: Hardware,
) -> Result<Box<dyn Renderer>, RendererError> {
    let device: Box<dyn Renderer> = match hardware {
        Hardware::CPU => Box::new(SoftwareRenderer::new(parameters)?),
        Hardware::GPU => Box::new(GpuRenderer::new(parameters)?),
    };
    Ok(device)
}

pub struct SoftwareRenderer {
    width: usize,
    height: usize,
    stride: usize,
}

impl SoftwareRenderer {
    fn new(parameters: SurfaceParameters) -> Result<SoftwareRenderer, RendererError> {
        Ok(SoftwareRenderer {
            width: parameters.width as usize,
            height: parameters.height as usize,
            stride: parameters.stride as usize,
        })
    }
}

impl Renderer for SoftwareRenderer {
    fn render(&mut self, scene: &Scene, dst: &mut [u8]) -> Result<(), RendererError> {
        let start_color = scene.timecode.round() as usize % 256;

        /*
        let fraction = scene.timecode * 4.0;
        let mut color = (fraction - (fraction / 255.0).floor() * 255.0) as u8;
        let (w, h) = (self.width, self.height);
        let (x_c, y_c) = (w / 2, h / 2);
        let max_l = x_c * x_c + y_c * y_c;
        // TODO: run in parallel?
        for y in 0..h {
            for x in 0..w {
                let rgb = (color + 1, color + 2, color);
                self.set_pixel(dst, x, y, rgb);
            }
            color += 1;
        }
        */

        let (w, h) = (self.width, self.height);
        let (x_c, y_c) = (w / 2, h / 2);
        let max_l = x_c * x_c + y_c * y_c;
        // TODO: run in parallel?
        for y in 0..h {
            let y_d = y.max(y_c) - y.min(y_c);
            let y_d = y_d * y_d;
            for x in 0..w {
                let x_d = x.max(x_c) - x.min(x_c);
                let x_d = x_d * x_d;
                let dist = y_d + x_d;

                let h = (dist * 256 * 128 / max_l + start_color) % 256;
                let s = 255;
                let v = 255;
                // Shows a tunnel vision effect
                /*
                let v = (256 - dist * 256 / max_l).min(255) as u8;
                let s = (256 - dist * 256 / max_l).min(255) as u8;
                */
                let rgb = SoftwareRenderer::hsl_to_rgb(h as u8, s, v);
                self.set_pixel(dst, x, y, rgb);
            }
        }
        Ok(())
    }

    fn device_name(&self) -> &str {
        "CPU"
    }
}

impl SoftwareRenderer {
    fn hsl_to_rgb(h: u8, s: u8, v: u8) -> (u8, u8, u8) {
        // Based on https://rasmithuk.org.uk/entry/fixed-hsv-rgb.
        const DIV_INTO_6: u16 = 0x0600;
        let c = ((s as u32 * v as u32) >> 8) as u16;
        let t = (h as u32 + 1) * DIV_INTO_6 as u32;
        let fiddle = (t & 0x00010000) != 0;

        let t = if fiddle {
            0x00020000 - (t & 0x0001ffff)
        } else {
            t & 0x0001ffff
        };

        let mut x = (t * c as u32) >> 16;
        if !fiddle {
            if x > 6 {
                x -= (6 * c as u32) >> 8;
            }
        } else {
            if x < 249 {
                x += (6 * c as u32) >> 8;
            }
        }

        let x = x as u16;
        let m = (v as u16 - c) as u16;
        let c = c as u16;

        let quad = (((h as u32 + 1) as u32 * DIV_INTO_6 as u32) >> 16) as u8 % 6;
        let (r, g, b) = match quad {
            0 => (c + m, x + m, m),
            1 => (x + m, c + m, m),
            2 => (m, c + m, x + m),
            3 => (m, x + m, c + m),
            4 => (x + m, m, c + m),
            5 => (c + m, m, x + m),
            _ => (0, 0, 0),
        };
        (r.min(255) as u8, g.min(255) as u8, b.min(255) as u8)
    }

    fn set_pixel(&mut self, mapping: &mut [u8], x: usize, y: usize, rgb: (u8, u8, u8)) {
        let start = (y * self.stride as usize + x) * 4;
        let target = &mut mapping[start..start + 4];
        let (r, g, b) = rgb;
        // Data is stored as a little-endian u32.
        target[0] = b;
        target[1] = g;
        target[2] = r;
        target[3] = 255;
    }
}

pub struct GpuRenderer {
    gpu: Gpu,
}

impl GpuRenderer {
    fn new(parameters: SurfaceParameters) -> Result<GpuRenderer, RendererError> {
        let gpu = Gpu::init(parameters)?;

        Ok(GpuRenderer { gpu })
    }
}

impl Renderer for GpuRenderer {
    fn render(&mut self, scene: &Scene, dst: &mut [u8]) -> Result<(), RendererError> {
        unsafe { self.gpu.render(scene, dst)? };
        Ok(())
    }

    fn device_name(&self) -> &str {
        self.gpu.device_name()
    }
}

#[derive(Debug)]
pub enum RendererError {
    InternalError(String),
    SurfaceError(SurfaceError),
    GpuError(GpuError),
    IoError(String, io::Error),
}

impl fmt::Display for RendererError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            RendererError::InternalError(ref msg) => f.write_str(msg),
            RendererError::SurfaceError(ref e) => e.fmt(f),
            RendererError::GpuError(ref e) => e.fmt(f),
            RendererError::IoError(ref msg, ref e) => {
                if !msg.is_empty() {
                    write!(f, "{}: {}", msg, e)
                } else {
                    e.fmt(f)
                }
            }
        }
    }
}

impl error::Error for RendererError {
    fn cause(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            RendererError::InternalError(ref _msg) => None,
            RendererError::SurfaceError(ref e) => Some(e),
            RendererError::GpuError(ref e) => Some(e),
            RendererError::IoError(ref _msg, ref e) => Some(e),
        }
    }
}

impl From<SurfaceError> for RendererError {
    fn from(e: SurfaceError) -> RendererError {
        RendererError::SurfaceError(e)
    }
}

impl From<GpuError> for RendererError {
    fn from(e: GpuError) -> RendererError {
        RendererError::GpuError(e)
    }
}

impl From<(&str, io::Error)> for RendererError {
    fn from(e: (&str, io::Error)) -> RendererError {
        RendererError::IoError(e.0.to_string(), e.1)
    }
}

impl From<&str> for RendererError {
    fn from(msg: &str) -> RendererError {
        RendererError::InternalError(msg.to_string())
    }
}
*/
