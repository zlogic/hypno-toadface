use std::{
    io::{self, Write},
    thread,
    time::{self, Instant},
};

use crate::graphics::Scene;

mod gpu;
mod graphics;
mod sighandler;
mod sound;

// TODO: make this configurable.
const ANIMATION_SPEED: f64 = 4.0 / 100.0;
// TODO: make this configurable.
const SOUND_DEVICE: &str = "/dev/snd/pcmC0D3p";

fn main() {
    let renderer = gpu::Gpu::init().expect("Failed to init GPU");
    let player = sound::Player::new(SOUND_DEVICE).expect("Failed to init audio device");

    println!("Using video device: {}", renderer.device_name());

    let start = Instant::now();
    let mut framecounter_start = Instant::now();
    let mut framecounter_frames = 0usize;
    unsafe {
        sighandler::listen_to_sigint();
    }
    loop {
        if sighandler::stop_requested() {
            break;
        }
        let scene = Scene {
            timecode: start.elapsed().as_secs_f64() * ANIMATION_SPEED,
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
            let fps = framecounter_frames as f32 / elapsed.as_secs_f32();

            print!("Average FPS: {:.2}   \r", fps);
            let _ = io::stdout().flush();

            framecounter_frames = 0;
            framecounter_start = Instant::now();
        }
    }
    drop(renderer);
    drop(player);

    println!("\nGoodbye.");
    // TODO: remove this code
    thread::sleep(time::Duration::from_secs(5));
}
