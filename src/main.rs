use std::io::{self, Write};
use std::time::Instant;

use hal::Surface;

mod hal;

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

fn main() {
    let mut surface = Surface::open("/dev/dri/card1").expect("Failed to open surface");
    let mut start_color = 0usize;
    let mut start = Instant::now();
    let mut frames = 0usize;

    loop {
        let mut map = surface.map_buffer().expect("Failed to map buffer");

        let (w, h) = (map.width(), map.height());
        let (x_c, y_c) = (w / 2, h / 2);
        let max_l = x_c * x_c + y_c * y_c;
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
                let rgb = hsl_to_rgb(h as u8, s, v);
                map.set_pixel(x, y, rgb);
            }
        }
        start_color += 1;

        frames += 1;
        if frames > 30 {
            let elapsed = start.elapsed();
            let fps = frames as f32 * 1000.0 / elapsed.as_millis() as f32;
            print!("Average FPS: {:.2}   \r", fps);
            let _ = io::stdout().flush();
            frames = 0;
            start = Instant::now();
        }
    }
}
