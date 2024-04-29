use std::{
    io::{self, Write},
    time::Instant,
};

use crate::graphics::Scene;

mod gpu;
mod graphics;

fn main() {
    let renderer = gpu::Gpu::init().expect("Failed to init GPU");

    println!("Using device: {}", renderer.device_name());

    let start = Instant::now();
    let mut framecounter_start = Instant::now();
    let mut framecounter_frames = 0usize;
    loop {
        let scene = Scene {
            timecode: start.elapsed().as_secs_f64() * 4.0,
        };

        let result = match renderer.render(&scene) {
            Ok(res) => res,
            Err(err) => {
                eprintln!("Failed to render scene: {}", err);
                continue;
            }
        };
        if result.swapchain_suboptimal {
            eprintln!("Swapchain is suboptimal");
        }
        if result.queue_suboptimal {
            eprintln!("Queue is suboptimal");
        }

        framecounter_frames += 1;
        if framecounter_frames > 100 {
            let elapsed = framecounter_start.elapsed();
            let fps = framecounter_frames as f32 / elapsed.as_secs_f32() as f32;

            print!("Average FPS: {:.2}   \r", fps);
            let _ = io::stdout().flush();

            framecounter_frames = 0;
            framecounter_start = Instant::now();
        }
    }
}
